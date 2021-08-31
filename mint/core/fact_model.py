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
"""The main FACT model and related functions."""

import copy
import os
from mint.core import base_model_util
from mint.core import base_models
from mint.core import metrics
from mint.core import multi_modal_model
from mint.core import multi_modal_model_util
import tensorflow as tf


class FACTModel(multi_modal_model.MultiModalModel):
  """Audio Motion Multi-Modal model."""

  def __init__(self, config, is_training):
    """Initializer for FACTModel.

    Args:
      config: `FACTConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
    """
    super().__init__(is_training)
    self.config = copy.deepcopy(config)
    self.is_training = is_training
    (self.feature_to_model, self.feature_to_params, self.feature_to_preprocessor
    ) = multi_modal_model_util.build_modalities_model(self.config.modality)

    self.cross_modal_layer = base_models.CrossModalLayer(
        self.config.cross_modal_model, is_training=self.is_training)
    motion_transformer_config = self.feature_to_model["motion"][
        "transformer_layer"]
    audio_transformer_config = self.feature_to_model["audio"][
        "transformer_layer"]
    self.motion_transformer = base_models.Transformer(
        hidden_size=motion_transformer_config.hidden_size,
        num_hidden_layers=motion_transformer_config.num_hidden_layers,
        num_attention_heads=motion_transformer_config.num_attention_heads,
        intermediate_size=motion_transformer_config.intermediate_size,
        initializer_range=motion_transformer_config.initializer_range)
    self.motion_pos_embedding = base_models.PositionEmbedding(
        self.feature_to_params["motion"]["sequence_length"],
        motion_transformer_config.hidden_size)
    self.motion_linear_embedding = base_models.LinearEmbedding(
        motion_transformer_config.hidden_size)
    self.audio_transformer = base_models.Transformer(
        hidden_size=audio_transformer_config.hidden_size,
        num_hidden_layers=audio_transformer_config.num_hidden_layers,
        num_attention_heads=audio_transformer_config.num_attention_heads,
        intermediate_size=audio_transformer_config.intermediate_size,
        initializer_range=audio_transformer_config.initializer_range)
    self.audio_pos_embedding = base_models.PositionEmbedding(
        self.feature_to_params["audio"]["sequence_length"],
        audio_transformer_config.hidden_size)
    self.audio_linear_embedding = base_models.LinearEmbedding(
        audio_transformer_config.hidden_size)

  def call(self, inputs):
    """Predict sequences from inputs. 

    This is a single forward pass that been used during training. 

    Args:
      inputs: Input dict of tensors. The dict should contains 
        `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension]) and
        `audio_input` ([batch_size, audio_seq_length, audio_feature_dimension]).

    Returns:
      Final output after the cross modal transformer. A tensor with shape 
      [batch_size, motion_seq_length + audio_seq_length, motion_feature_dimension]
      will be return. **Be aware only the first N-frames are supervised during training**
    """
    # Computes motion features.
    motion_features = self.motion_linear_embedding(inputs["motion_input"])
    # `motion_features` shape = [batch_size, seq_length, hidden_size].
    motion_features = self.motion_pos_embedding(motion_features)
    motion_features = self.motion_transformer(motion_features)

    # Computes audio features.
    audio_features = self.audio_linear_embedding(inputs["audio_input"])
    audio_features = self.audio_pos_embedding(audio_features)
    audio_features = self.audio_transformer(audio_features)

    # Computes cross modal output.
    output = self.cross_modal_layer(motion_features, audio_features)

    return output

  def infer_auto_regressive(self, inputs, steps=1200):
    """Predict sequences from inputs in an auto-regressive manner. 

    This function should be used only during inference. During each forward step, 
    only the first frame was kept. Inputs are shifted by 1 frame after each forward.


    Args:
      inputs: Input dict of tensors. The dict should contains 
        `motion_input` ([batch_size, motion_seq_length, motion_feature_dimension]) and
        `audio_input` ([batch_size, audio_seq_length, audio_feature_dimension]).

    Returns:
      Final output after the auto-regressive inference. A tensor with shape 
      [batch_size, steps, motion_feature_dimension]
      will be return.
    """
    audio_seq_length = self.feature_to_params["audio"]["sequence_length"]
    outputs = []
    motion_input = inputs["motion_input"]
    for i in range(steps):
      audio_input = inputs["audio_input"][:, i: i + audio_seq_length]
      if tf.shape(audio_input)[1] < audio_seq_length:
        break
      output = self.call({"motion_input": motion_input, "audio_input": audio_input})
      output = output[:, 0:1, :]  # only keep the first frame
      outputs.append(output)
      # update motion input
      motion_input = tf.concat([motion_input[:, 1:, :], output], axis=1)
    return tf.concat(outputs, axis=1)
      
  def loss(self, target, pred):
    motion_generation_loss = self.compute_motion_generation_loss(pred, target)
    return motion_generation_loss

  def get_metrics(self, eval_config):
    """Computes metrics."""
    # Currently we do off-line metrics calculation.
    return []

  def compute_motion_generation_loss(self, pred_tensors, target_tensors):
    """Compute motion generation loss from layer output."""
    _, target_seq_len, _ = base_model_util.get_shape_list(target_tensors)
    diff = target_tensors - pred_tensors[:, :target_seq_len]
    l2_loss = tf.reduce_mean(tf.square(diff))
    return l2_loss
