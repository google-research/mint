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
"""Custom learning schedules."""

import tensorflow as tf


class ManualStepping(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Manual stepping learning rate schedule."""

  def __init__(self, lr_step_boundaries, lr_sequence, warmup, name=None):
    super().__init__()
    if any([b < 0 for b in lr_step_boundaries]) or any(
        [not isinstance(b, int) for b in lr_step_boundaries]):
      raise ValueError("boundaries must be a list of positive integers")
    if any([
        bnext <= b
        for bnext, b in zip(lr_step_boundaries[1:], lr_step_boundaries[:-1])
    ]):
      raise ValueError("Entries in boundaries must be strictly increasing.")
    if any([not isinstance(r, float) for r in lr_sequence]):
      raise ValueError("Learning rates must be floats")
    if len(lr_sequence) != len(lr_step_boundaries) + 1:
      raise ValueError("Number of provided learning rates must exceed "
                       "number of boundary points by exactly 1.")

    if lr_step_boundaries and lr_step_boundaries[0] == 0:
      raise ValueError("First step cannot be zero.")

    if warmup and lr_step_boundaries:
      slope = (lr_sequence[1] -
               lr_sequence[0]) * 1.0 / lr_step_boundaries[0]
      warmup_steps = list(range(lr_step_boundaries[0]))
      warmup_rates = [
          lr_sequence[0] + slope * step for step in warmup_steps
      ]
      lr_step_boundaries = warmup_steps + lr_step_boundaries
      lr_sequence = warmup_rates + lr_sequence[1:]
    else:
      lr_step_boundaries = [0] + lr_step_boundaries

    self.num_boundaries = len(lr_step_boundaries)
    self.lr_step_boundaries = lr_step_boundaries
    self.lr_sequence = lr_sequence
    self.warmup = warmup
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "ManualStepping") as name:
      rate_index = tf.reduce_max(
          tf.where(
              tf.greater_equal(step, self.lr_step_boundaries),
              list(range(self.num_boundaries)), [0] * self.num_boundaries))
      return tf.reduce_sum(
          self.lr_sequence * tf.one_hot(rate_index, depth=self.num_boundaries),
          name=name)


class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Applies a warmup schedule on a given learning rate decay schedule.

  Args:
      initial_learning_rate (:obj:`float`): The initial learning rate for the
        schedule after the warmup (so this will be the learning rate at the end
        of the warmup).
      decay_schedule_fn (:obj:`Callable`): The schedule function to apply after
        the warmup for the rest of training.
      warmup_steps (:obj:`int`): The number of steps for the warmup part of
        training.
      power (:obj:`float`, `optional`, defaults to 1): The power to use for the
        polynomial warmup (defaults is a linear warmup).
      name (:obj:`str`, `optional`): Optional name prefix for the returned
        tensors during the schedule.
  """

  def __init__(
      self,
      initial_learning_rate,
      decay_schedule_fn,
      warmup_steps,
      power=1.0,
      name=None,
  ):
    super().__init__()
    self.initial_learning_rate = initial_learning_rate
    self.warmup_steps = warmup_steps
    self.power = power
    self.decay_schedule_fn = decay_schedule_fn
    self.name = name

  def __call__(self, step):
    with tf.name_scope(self.name or "WarmUp") as name:
      # Implements polynomial warmup. i.e., if global_step < warmup_steps, the
      # learning rate will be `global_step/num_warmup_steps * init_lr`.
      global_step_float = tf.cast(step, tf.float32)
      warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
      warmup_percent_done = global_step_float / warmup_steps_float
      warmup_learning_rate = self.initial_learning_rate * tf.math.pow(
          warmup_percent_done, self.power)
      return tf.cond(
          global_step_float < warmup_steps_float,
          lambda: warmup_learning_rate,
          lambda: self.decay_schedule_fn(step - self.warmup_steps),
          name=name,
      )

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "decay_schedule_fn": self.decay_schedule_fn,
        "warmup_steps": self.warmup_steps,
        "power": self.power,
        "name": self.name,
    }


class CosineDecayWithWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Cosine Decay with linear warmup.

  Extends Keras Cosine Decay Schedule to include a linear warmup at the
  beginning.
  """

  def __init__(self, initial_learning_rate, steps, warmup=0, alpha=0.0):
    """Applies cosine decay to the learning rate.

    Args:
      initial_learning_rate: A scalar `float32` Tensor or a Python number. The
        initial learning rate.
      steps: A scalar `int32` or `int64` `Tensor` or a Python number. Number of
        steps to decay over including warmup.
      warmup: Number of steps to perform linear warmup before starting cosine
        decay with initial learning rate.
      alpha: A scalar `float32` or `float64` Tensor or a Python number. Minimum
        learning rate value as a fraction of initial_learning_rate.
    """
    super(CosineDecayWithWarmup, self).__init__()

    self.initial_learning_rate = initial_learning_rate
    self.steps = steps
    self.warmup = warmup
    self.alpha = alpha

  def __call__(self, step):
    with tf.name_scope(self.name or "CosineDecayWithWarmUp"):
      initial_learning_rate = tf.convert_to_tensor(
          self.initial_learning_rate, name="initial_learning_rate")
      dtype = initial_learning_rate.dtype
      global_step = tf.cast(step, dtype)
      warmup = tf.cast(self.warmup, dtype)
      warmup_indicator = tf.cast(global_step < warmup, dtype=dtype)
      one = tf.constant(1, dtype=dtype)
      warmup_lr = (global_step * self.initial_learning_rate / (warmup - one))
      return warmup_indicator * warmup_lr + (
          one - warmup_indicator) * super().__call__(global_step - warmup + one)

  def get_config(self):
    return {
        "initial_learning_rate": self.initial_learning_rate,
        "steps": self.steps,
        "warmup": self.warmup,
        "alpha": self.alpha,
        "name": self.name
    }
