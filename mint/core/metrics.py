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
"""Metrics to measure the sequence generation accuracy."""
from mint.core import base_model_util
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg


class EulerAnglesError(tf.keras.metrics.Metric):
  """Metric to measure positional accuracy.

  This metric measures the difference between two poses in terms of
  euler angles. The metrics have been used in Motion Prediction work such as
  http://arxiv.org/abs/2004.08692.
  """

  def __init__(self, num_joints):
    super(EulerAnglesError, self).__init__(name='EulerAnglesError')
    self.num_joints = num_joints
    self.euler_errors = self.add_weight(
        name='euler_errors', initializer='zeros')

  def update_state(self, inputs, pred):
    """Update metrics.

    Args:
      target: float32 tensor of shape [batch, sequence_length, (num_joints+1)*9]
        the groundtruth motion vector with the first 9 dim as the translation.
      pred: float32 tensor of shape [batch, sequence_length, (num_joints+1)*9]
        the predicted motion vector with the first 9 dim as the translation.
    """
    target = inputs["target"]
    _, target_seq_len, _ = base_model_util.get_shape_list(target)
    euler_preds = tfg.euler.from_rotation_matrix(
        tf.reshape(pred[:, :target_seq_len, 9:], [-1, 3, 3]))
    euler_targs = tfg.euler.from_rotation_matrix(
        tf.reshape(target[:, :, 9:], [-1, 3, 3]))

    # Euler conversion might have NANs, here replace nans with zeros.
    euler_preds = tf.where(
        tf.math.is_nan(euler_preds), tf.zeros_like(euler_preds), euler_preds)
    euler_targs = tf.where(
        tf.math.is_nan(euler_targs), tf.zeros_like(euler_targs), euler_targs)
    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = tf.reshape(euler_preds, [-1, self.num_joints * 3])
    euler_targs = tf.reshape(euler_targs, [-1, self.num_joints * 3])

    euler_diff = tf.norm(euler_targs - euler_preds, axis=-1)
    self.euler_errors.assign_add(tf.reduce_mean(euler_diff))

  def result(self):
    return self.euler_errors