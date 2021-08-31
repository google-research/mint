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
"""An evaluator object that can evaluate models with a single output."""
from third_party.tf_models import orbit
import tensorflow as tf
import numpy as np
import os


class SingleTaskEvaluator(orbit.StandardEvaluator):
  """Evaluates a single-output model on a given dataset.

  This evaluator will handle running a model with one output on a single
  dataset, and will apply the output of that model to one or more
  `tf.keras.metrics.Metric` objects.
  """

  def __init__(self,
               eval_dataset,
               model,
               metrics,
               output_dir=None,
               evaluator_options=None):
    """Initializes a `SingleTaskEvaluator` instance.

    If the `SingleTaskEvaluator` should run its model under a distribution
    strategy, it should be created within that strategy's scope.

    Arguments:
      eval_dataset: A `tf.data.Dataset` or `DistributedDataset` that contains a
        string-keyed dict of `Tensor`s.
      model: A `tf.Module` or Keras `Model` object to evaluate.
      metrics: A single `tf.keras.metrics.Metric` object, or a list of
        `tf.keras.metrics.Metric` objects.
      evaluator_options: An optional `orbit.StandardEvaluatorOptions` object.
    """

    self.model = model
    self.metrics = metrics if isinstance(metrics, list) else [metrics]
    self.output_dir = output_dir

    # Capture the strategy from the containing scope.
    self.strategy = tf.distribute.get_strategy()

    super(SingleTaskEvaluator, self).__init__(
        eval_dataset=eval_dataset, options=evaluator_options)

  def eval_begin(self):
    """Actions to take once before every eval loop."""
    for metric in self.metrics:
      metric.reset_states()

  def eval_step(self, iterator):
    """One eval step. Called multiple times per eval loop by the superclass."""

    def step_fn(inputs):
      # [batch_size, steps, motion_feature_dimension]
      outputs = self.model.infer_auto_regressive(inputs, steps=1200)
      # [batch_size, motion_seq_length + steps, motion_feature_dimension]
      outputs = tf.concat([inputs["motion_input"], outputs], axis=1)
      batch_size = tf.shape(outputs)[0]
      if self.output_dir is not None:
        os.makedirs(self.output_dir, exist_ok=True)
        # save each batch instance seperately
        for i in range(batch_size):
          output = outputs[i].numpy()
          save_path = os.path.join(self.output_dir, "%s_%s.npy" % (
              inputs["motion_name"][i].numpy().decode("utf-8"),
              inputs["audio_name"][i].numpy().decode("utf-8"),
          ))
          print ("Saving results to %s" % save_path)
          np.save(save_path, output)  # [steps, motion_feature_dimension]
      # calculate metrics
      for metric in self.metrics:
        metric.update_state(inputs, outputs)
      
    # This is needed to handle distributed computation.
    self.strategy.run(step_fn, args=(next(iterator),))

  def eval_end(self):
    """Actions to take once after an eval loop."""
    with self.strategy.scope():
      # Export the metrics.
      metrics = {metric.name: metric.result() for metric in self.metrics}

    return metrics
