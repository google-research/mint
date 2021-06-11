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
"""Basic building blocks for the multi-modal model."""

from einops.layers.tensorflow import Rearrange
from mint.core import base_model_util
from mint.protos import model_pb2
import tensorflow as tf


class Norm(tf.keras.Model):
  """Layer normalization."""

  def __init__(self, fn):
    super().__init__()
    self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
    self.fn = fn

  def call(self, x):
    return self.fn(self.norm(x))


class Residual(tf.keras.Model):
  """Residual layer."""

  def __init__(self, fn):
    super().__init__()
    self.fn = fn

  def call(self, x):
    return self.fn(x) + x


class MLP(tf.keras.Model):
  """Feedforward layer."""

  def __init__(self, out_dim, hidden_dim):
    super().__init__()
    self.net = tf.keras.Sequential([
        tf.keras.layers.Dense(
            hidden_dim, activation=base_model_util.get_activation("gelu")),
        tf.keras.layers.Dense(out_dim)
    ])

  def call(self, x):
    return self.net(x)


class Attention(tf.keras.Model):
  """Attention layer."""

  def __init__(self, dim, heads=8):
    super().__init__()
    self.heads = heads
    self.scale = dim**-0.5

    self.to_qkv = tf.keras.layers.Dense(dim * 3, use_bias=False)
    self.to_out = tf.keras.layers.Dense(dim)

    self.rearrange_qkv = Rearrange(
        "b n (qkv h d) -> qkv b h n d", qkv=3, h=self.heads)
    self.rearrange_out = Rearrange("b h n d -> b n (h d)")

  def call(self, x):
    qkv = self.to_qkv(x)
    qkv = self.rearrange_qkv(qkv)
    q = qkv[0]
    k = qkv[1]
    v = qkv[2]

    dots = tf.einsum("bhid,bhjd->bhij", q, k) * self.scale
    attn = tf.nn.softmax(dots, axis=-1)

    out = tf.einsum("bhij,bhjd->bhid", attn, v)
    out = self.rearrange_out(out)
    out = self.to_out(out)
    return out


class Transformer(tf.keras.Model):
  """Transformer Encoder."""

  def __init__(self,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               initializer_range=0.02):
    super().__init__()
    blocks = []
    for _ in range(num_hidden_layers):
      blocks.extend([
          Residual(Norm(Attention(hidden_size, heads=num_attention_heads))),
          Residual(Norm(MLP(hidden_size, intermediate_size)))
      ])
    self.net = tf.keras.Sequential(blocks)

  def call(self, x):
    return self.net(x)


class PatchEmbedding(tf.keras.Model):
  """Images patch embedding layer."""

  def __init__(self, config):
    super().__init__()
    self.patch_embed_layer = tf.keras.layers.Dense(config.hidden_size)
    self.rearrange = Rearrange(
        "b (h p1) (w p2) c -> b (h w) (p1 p2 c)",
        p1=config.patch_size,
        p2=config.patch_size,
        c=config.num_channel)

  def call(self, x):
    x = self.rearrange(x)
    return self.patch_embed_layer(x)


class LinearEmbedding(tf.keras.Model):
  """Linear projection."""

  def __init__(self, dim):
    super().__init__()
    self.net = tf.keras.layers.Dense(dim)

  def call(self, x):
    return self.net(x)


class PositionEmbedding(tf.keras.Model):
  """Position Embedding layer."""

  def __init__(self, seq_length, dim):
    super().__init__()

    pos_initializer = base_model_util.create_initializer(0.02)
    self.pos_embedding = self.add_weight(
        "position_embedding",
        shape=[seq_length, dim],
        initializer=pos_initializer,
        dtype=tf.float32)

  def call(self, x):
    """Call embedding layer."""
    return x + self.pos_embedding


class CrossModalLayer(tf.keras.layers.Layer):
  """Cross-modal layer."""

  def __init__(self, config, is_training):
    super().__init__()
    self.config = config
    self.is_training = is_training
    self.model_type = self.config.WhichOneof("model")
    model_config = self.config.transformer
    self.transformer_layer = Transformer(
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        initializer_range=model_config.initializer_range)

    output_layer_config = self.config.output_layer
    self.cross_output_layer = tf.keras.layers.Dense(
        units=output_layer_config.out_dim,
        activation=None,
        kernel_initializer=base_model_util.create_initializer(
            output_layer_config.initializer_range))

  def call(self, modal_a_sequences, modal_b_sequences):
    """Get loss for the cross-modal tasks."""
    _, _, modal_a_width = base_model_util.get_shape_list(modal_a_sequences)
    _, _, modal_b_width = base_model_util.get_shape_list(modal_b_sequences)
    if modal_a_width != modal_b_width:
      raise ValueError(
          "The modal_a hidden size (%d) should be the same with the modal_b "
          "hidden size (%d)" % (modal_a_width, modal_b_width))
    if self.config.cross_modal_concat_dim == model_pb2.CrossModalModel.CrossModalConcatDim.SEQUENCE_WISE:
      # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
      merged_sequences = tf.concat([modal_a_sequences, modal_b_sequences],
                                   axis=1)
    else:
      raise NotImplementedError("cross_modal_concat_dim %s is not supported." %
                                self.config.cross_modal_concat_dim)

    # [batch_size, modal_a_seq_len + modal_b_seq_len, width]
    merged_sequences = self.transformer_layer(merged_sequences)
    logits = self.cross_output_layer(merged_sequences)

    return logits
