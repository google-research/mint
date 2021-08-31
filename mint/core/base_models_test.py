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
"""Tests for mint.core.base_models."""

from mint.core import base_models
import tensorflow as tf


class TransformerModelTest(tf.test.TestCase):

  def test_run(self):
    transformer = base_models.Transformer(
        hidden_size=20, num_attention_heads=10)
    input_tensor = tf.ones([4, 128, 20])
    out_tensor = transformer(input_tensor)
    self.assertEqual(out_tensor.numpy().shape[0], 4)
    self.assertEqual(out_tensor.numpy().shape[1], 128)
    self.assertEqual(out_tensor.numpy().shape[2], 20)


class PositionEmbeddingLayerTest(tf.test.TestCase):

  def test_3d_input(self):
    embedding_layer = base_models.PositionEmbedding(128, 219)
    input_tensor = tf.ones([4, 128, 219])
    out_tensor = embedding_layer(input_tensor)
    self.assertEqual(out_tensor.numpy().shape[0], 4)
    self.assertEqual(out_tensor.numpy().shape[1], 128)
    self.assertEqual(out_tensor.numpy().shape[2], 219)

if __name__ == "__main__":
  tf.test.main()
