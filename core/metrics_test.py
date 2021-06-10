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
