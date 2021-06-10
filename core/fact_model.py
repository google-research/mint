"""The main FACT model and related functions."""

import copy
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

    Args:
      inputs: Input dict of tensors, the output from the provide_inputs().

    Returns:
      motion_sequence_output: Tensor of shape
        [batch_size, seq_length, motion_feature_dimension]
      motion_last_output: Tensor of shape [batch_size, motion_feature_dimension]
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

  def loss(self, target, pred):
    motion_generation_loss = self.compute_motion_generation_loss(pred, target)
    return motion_generation_loss

  def get_metrics(self, eval_config):
    """Computes metrics."""
    eval_metric_config = eval_config.eval_metric.motion_generation_metrics
    eval_metrics = [metrics.EulerAnglesError(eval_metric_config.num_joints)]
    return eval_metrics

  def compute_motion_generation_loss(self, pred_tensors, target_tensors):
    """Compute motion generation loss from layer output."""
    _, target_seq_len, _ = base_model_util.get_shape_list(target_tensors)
    diff = target_tensors - pred_tensors[:, :target_seq_len]
    l2_loss = tf.reduce_mean(tf.square(diff))
    return l2_loss
