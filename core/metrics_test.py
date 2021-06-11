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
"""Tests for metrics."""
from mint.core import metrics
import tensorflow as tf


class MetricsTest(tf.test.TestCase):

  def test_euler_angles_error(self):
    num_joints = 3
    metrics_fn = metrics.EulerAnglesError(num_joints)
    target_rotmat = tf.reshape(
        tf.eye(3, batch_shape=[4, 10, num_joints + 1]),
        [4, 10, (num_joints + 1) * 9])
    pred_rotmat = tf.reshape(
        tf.eye(3, batch_shape=[4, 10, num_joints + 1]),
        [4, 10, (num_joints + 1) * 9])
    euler_angles_error = metrics_fn(target_rotmat, pred_rotmat)
    self.assertAllClose(0., euler_angles_error)


if __name__ == '__main__':
  tf.test.main()
