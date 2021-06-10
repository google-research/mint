"""Tests for the single_task_trainer."""
from mint.ctl import single_task_trainer
import orbit

import tensorflow as tf
import tensorflow_datasets as tfds


class SingleTaskTrainerTest(tf.test.TestCase):

  def test_single_task_training(self):
    iris = tfds.load('iris')
    train_ds = iris['train'].batch(32).repeat()

    model = tf.keras.Sequential([
        tf.keras.Input(shape=(4,), name='features'),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.relu),
        tf.keras.layers.Dense(3)
    ])

    trainer = single_task_trainer.SingleTaskTrainer(
        train_ds,
        label_key='label',
        model=model,
        loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.SGD(
            learning_rate=tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                [0], [0.01, 0.01])))

    controller = orbit.Controller(
        trainer=trainer,
        steps_per_loop=100,
        global_step=trainer.optimizer.iterations)

    controller.train(1)
    start_loss = trainer.train_loss.result().numpy()
    controller.train(500)
    end_loss = trainer.train_loss.result().numpy()

    # Assert that the model has trained 'significantly' - that the loss
    # has dropped by over 50%.
    self.assertLess(end_loss, start_loss / 2)


if __name__ == '__main__':
  tf.test.main()
