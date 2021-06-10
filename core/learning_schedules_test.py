"""Tests for mint.core.learning_schedules."""

from mint.core import learning_schedules
import tensorflow as tf


class LearningSchedulesTest(tf.test.TestCase):

  def test_cosine_with_warmup(self):
    lr = learning_schedules.CosineDecayWithWarmup(
        initial_learning_rate=1.0, steps=10, warmup=4, alpha=1e-4)
    lrs = []
    for i in range(10):
      lrs.append(lr(i).numpy())
    self.assertAllClose(
        lrs, [0.0, 0.33, 0.66, 1.0, 0.933, 0.750, 0.500, 0.25, 0.067, 1e-04],
        1e-2, 1e-2)

  def test_cosine_with_warmup_int64(self):
    lr = learning_schedules.CosineDecayWithWarmup(
        initial_learning_rate=1.0, steps=10, warmup=4, alpha=1e-4)
    lrs = []
    for i in range(10):
      lrs.append(lr(tf.cast(i, tf.int64)).numpy())
    self.assertAllClose(
        lrs, [0.0, 0.33, 0.66, 1.0, 0.933, 0.750, 0.500, 0.25, 0.067, 1e-04],
        1e-2, 1e-2)


if __name__ == '__main__':
  tf.test.main()
