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
"""Module to train a condititonal flow prediction model."""

from absl import app
from absl import flags
from mint.core import inputs
from mint.core import learning_schedules
from mint.core import model_builder
from mint.ctl import single_task_trainer
from mint.utils import config_util
from third_party.tf_models import orbit
import tensorflow as tf


TRAIN_STRATEGY = ['tpu', 'gpu']

FLAGS = flags.FLAGS
flags.DEFINE_enum('train_strategy', TRAIN_STRATEGY[1], TRAIN_STRATEGY,
                  'Whether to train with TPUs or Mirrored GPUs.')
flags.DEFINE_string('master', None, 'BNS name of the TensorFlow tpu to use.')
flags.DEFINE_string('config_path', None, 'Path to the config file.')
flags.DEFINE_string('model_dir', None,
                    'Directory to write training checkpoints and logs')
flags.DEFINE_float('initial_learning_rate', 0.1,
                   'Initial learning rate for cosine decay schedule')
flags.DEFINE_string(
    'head_initializer', 'he_normal',
    'Initializer for prediction head. Valid options are any '
    'of the tf.keras.initializers.')
flags.DEFINE_integer('steps', 2400000, 'Number of training steps')
flags.DEFINE_integer('warmup_steps', 1000,
                     'Number of learning rate warmup steps')
flags.DEFINE_float('weight_decay', None, 'L2 regularization penalty to apply.')
flags.DEFINE_float('grad_clip_norm', 0., 'Clip gradients by norm.')


def _create_learning_rate(learning_rate_config):
  """Create optimizer learning rate based on config.

  Args:
    learning_rate_config: A LearningRate proto message.

  Returns:
    A learning rate schedule.

  Raises:
    ValueError: when using an unsupported input data type.
  """
  lr_schedule = None
  learning_rate_type = learning_rate_config.WhichOneof('learning_rate')

  if learning_rate_type == 'exponential_decay_learning_rate':
    config = learning_rate_config.exponential_decay_learning_rate
    lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
        FLAGS.initial_learning_rate,
        decay_steps=config.decay_steps,
        end_learning_rate=config.min_learning_rate,
        power=config.decay_factor)
    if FLAGS.warmup_steps:
      lr_schedule = learning_schedules.WarmUp(
          FLAGS.initial_learning_rate,
          decay_schedule_fn=lr_schedule,
          warmup_steps=FLAGS.warmup_steps)

  if learning_rate_type == 'manual_step_learning_rate':
    config = learning_rate_config.manual_step_learning_rate
    if not config.schedule:
      raise ValueError('Empty learning rate schedule.')
    learning_rate_step_boundaries = [x.step for x in config.schedule]
    learning_rate_sequence = [config.initial_learning_rate]
    learning_rate_sequence += [x.learning_rate for x in config.schedule]
    lr_schedule = learning_schedules.ManualStepping(
        learning_rate_step_boundaries, learning_rate_sequence,
        config.warmup)

  if learning_rate_type == 'cosine_decay_learning_rate':
    config = learning_rate_config.cosine_decay_learning_rate
    lr_schedule = learning_schedules.CosineDecayWithWarmup(
        FLAGS.initial_learning_rate, config.total_steps, FLAGS.warmup_steps)

  if lr_schedule is None:
    raise ValueError('Learning_rate %s not supported.' % learning_rate_type)

  return lr_schedule


def get_dataset_fn(configs):
  """Returns tf dataset."""

  def dataset_fn(input_context=None):
    del input_context
    train_config = configs['train_config']
    train_dataset_config = configs['train_dataset']
    use_tpu = (FLAGS.train_strategy == TRAIN_STRATEGY[0])
    dataset = inputs.create_input(
        train_config, train_dataset_config, use_tpu=use_tpu)
    return dataset

  return dataset_fn


def summary_fn(loss_dict, global_step):
  """Function to summarize model input and output tensors.

  Args:
    loss_dict: A dictionary of the losses.
    global_step: Global step value.
  """
  for loss_type in loss_dict:
    tf.summary.scalar(loss_type, loss_dict[loss_type], step=global_step)


def distribution_strategy():
  """Returns strategy for distributed training."""
  if FLAGS.train_strategy == TRAIN_STRATEGY[0]:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.master)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
  else:
    strategy = tf.distribute.MirroredStrategy()
  return strategy


def train():
  """Trains model."""
  configs = config_util.get_configs_from_pipeline_file(FLAGS.config_path)

  model_config = configs['model']
  train_config = configs['train_config']

  strategy = distribution_strategy()
  dataset = strategy.distribute_datasets_from_function(get_dataset_fn(configs))
  with strategy.scope():
    model_ = model_builder.build(model_config, True)
    lr_schedule = _create_learning_rate(train_config.learning_rate)
    optimizer = tf.keras.optimizers.Adam(lr_schedule)
    model_.global_step = optimizer.iterations
    summaryfn = None
    if FLAGS.train_strategy == TRAIN_STRATEGY[1]:
      summaryfn = summary_fn
    trainer = single_task_trainer.SingleTaskTrainer(
        dataset,
        label_key='target',
        model=model_,
        loss_fn=model_.loss,
        optimizer=optimizer,
        summary_fn=summaryfn,
        grad_clip_norm=FLAGS.grad_clip_norm)

  controller = orbit.Controller(
      trainer=trainer,
      strategy=strategy,
      steps_per_loop=10,
      checkpoint_manager=tf.train.CheckpointManager(
          tf.train.Checkpoint(optimizer=optimizer, model=model_),
          directory=FLAGS.model_dir,
          checkpoint_interval=1000,
          step_counter=trainer.optimizer.iterations,
          max_to_keep=5),
      summary_dir=FLAGS.model_dir,
      summary_interval=10,
      global_step=trainer.optimizer.iterations)
  controller.train(1)
  controller.train(FLAGS.steps - 1)


def main(_):
  train()


if __name__ == '__main__':
  app.run(main)
