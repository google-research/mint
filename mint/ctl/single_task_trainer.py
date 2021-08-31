# Copyright 2021, Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A trainer object that can train models with a single output."""

from absl import logging
from third_party.tf_models import orbit
import tensorflow as tf


class IdentityMetric(tf.keras.metrics.Metric):
  """Keras metric to report value at any instant."""

  def __init__(self, name, aggregation):
    """Constructor.

    Args:
      name: Name of the metric.
      aggregation: A tf.VariableAggregation method that indicates how to
        aggregate values across replicas.
    """
    super(IdentityMetric, self).__init__(name=name)
    self.value = self.add_weight(
        name='/'.join([name, 'value']),
        initializer='zeros',
        aggregation=aggregation)

  def update_state(self, current_value):
    """Update metrics.

    Args:
      current_value: A scalar value for the metric.
    """
    self.value.assign(current_value)

  def result(self):
    return self.value


class SingleTaskTrainer(orbit.StandardTrainer):
  """Trains a single-output model on a given dataset.

  This trainer will handle running a model with one output on a single
  dataset. It will apply the provided loss function to the model's output
  to calculate gradients and will apply them via the provided optimizer. It will
  also supply the output of that model to one or more `tf.keras.metrics.Metric`
  objects.
  """

  def __init__(self,
               train_dataset,
               label_key,
               model,
               loss_fn,
               optimizer,
               metrics=None,
               trainer_options=None,
               summary_fn=None,
               grad_clip_norm=0.):
    """Initializes a `SingleTaskTrainer` instance.

    If the `SingleTaskTrainer` should run its model under a distribution
    strategy, it should be created within that strategy's scope.

    This trainer will also calculate metrics during training. The loss metric
    is calculated by default, but other metrics can be passed to the `metrics`
    arg.

    Arguments:
      train_dataset: A `tf.data.Dataset` or `DistributedDataset` that contains a
        string-keyed dict of `Tensor`s.
      label_key: The key corresponding to the label value in feature
        dictionaries dequeued from `train_dataset`. This key will be removed
        from the dictionary before it is passed to the model.
      model: A `tf.Module` or Keras `Model` object to evaluate. It must accept a
        `training` kwarg.
      loss_fn: A per-element loss function of the form (target, output). The
        output of this loss function will be reduced via `tf.reduce_mean` to
        create the final loss. We recommend using the functions in the
        `tf.keras.losses` package or `tf.keras.losses.Loss` objects with
        `reduction=tf.keras.losses.reduction.NONE`.
      optimizer: A `tf.keras.optimizers.Optimizer` instance.
      metrics: A single `tf.keras.metrics.Metric` object, or a list of
        `tf.keras.metrics.Metric` objects.
      trainer_options: An optional `orbit.utils.StandardTrainerOptions` object.
      summary_fn: A function that adds tf.summary on model input and output
        tensors.
      grad_clip_norm: A float to clip the gradients by global norm.
    """
    self.label_key = label_key
    self.model = model
    self.loss_fn = loss_fn
    self.optimizer = optimizer
    self.summary_fn = summary_fn
    self.grad_clip_norm = grad_clip_norm

    # Capture the strategy from the containing scope.
    self.strategy = tf.distribute.get_strategy()

    self.train_loss = IdentityMetric('training_loss',
                                     tf.VariableAggregation.SUM)
    self.task_loss = IdentityMetric('task_loss', tf.VariableAggregation.SUM)
    self.regularization_loss = IdentityMetric('regularization_loss',
                                              tf.VariableAggregation.SUM)
    self.learning_rate = IdentityMetric(
        'learning_rate', tf.VariableAggregation.ONLY_FIRST_REPLICA)

    # We need self.metrics to be an iterable later, so we handle that here.
    if metrics is None:
      self.metrics = []
    elif isinstance(metrics, list):
      self.metrics = metrics
    else:
      self.metrics = [metrics]

    super(SingleTaskTrainer, self).__init__(
        train_dataset=train_dataset, options=trainer_options)

  def train_loop_begin(self):
    """Actions to take once, at the beginning of each train loop."""
    self.train_loss.reset_states()
    self.task_loss.reset_states()
    self.regularization_loss.reset_states()
    self.learning_rate.reset_states()
    for metric in self.metrics:
      metric.reset_states()

  def train_step(self, iterator):
    """A train step. Called multiple times per train loop by the superclass."""

    def train_fn(inputs):
      with tf.GradientTape() as tape:
        # Extract the target value and delete it from the input dict, so that
        # the model never sees it.
        target = inputs.pop(self.label_key)

        # Get the outputs of the model.
        logging.info('*** Features ***')
        for name in sorted(inputs.keys()):
          logging.info('  name = %s', name)
        output = self.model(inputs, training=True)

        # Get the average per-batch loss and scale it down by the number of
        # replicas. This ensures that we don't end up multiplying our loss by
        # the number of workers - gradients are summed, not averaged, across
        # replicas during the apply_gradients call.
        loss = tf.reduce_mean(self.loss_fn(target, output))
        loss = loss / self.strategy.num_replicas_in_sync

        # Since we don't use compile/fit api for training, the only losses added
        # to the model are regularization losses.
        regularization_loss = 0
        if self.model.losses:
          regularization_loss = tf.add_n(self.model.losses)
        regularization_loss = (
            regularization_loss / self.strategy.num_replicas_in_sync)
        total_loss = loss + regularization_loss

        loss_dict = {
            'total_loss': total_loss,
            'loss:': loss,
            'reg_loss': regularization_loss,
        }
        if self.summary_fn:
          self.summary_fn(loss_dict, self.optimizer.iterations)
        # Get the gradients by applying the loss to the model's trainable
        # variables.
        gradients = tape.gradient(total_loss, self.model.trainable_variables)

        if self.grad_clip_norm > 0.:
          logging.info('Clipping gradient by norm: {:.3f}'.format(
              self.grad_clip_norm))
          gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)

        # Apply the gradients via the optimizer.
        self.optimizer.apply_gradients(
            list(zip(gradients, self.model.trainable_variables)))

        # Update metrics.
        self.train_loss.update_state(total_loss)
        self.task_loss.update_state(loss)
        self.regularization_loss.update_state(regularization_loss)
        self.learning_rate.update_state(
            self.optimizer.learning_rate(self.optimizer.iterations))
        for metric in self.metrics:
          metric.update_state(target, output)

    # This is needed to handle distributed computation.
    self.strategy.run(train_fn, args=(next(iterator),))

  def train_loop_end(self):
    """Actions to take once after a training loop."""
    with self.strategy.scope():
      # Export the metrics.
      metrics = {metric.name: metric.result() for metric in self.metrics}
      metrics[self.train_loss.name] = self.train_loss.result()
      metrics[self.task_loss.name] = self.task_loss.result()
      metrics[self.regularization_loss.name] = self.regularization_loss.result()
      metrics[self.learning_rate.name] = self.learning_rate.result()

    return metrics
