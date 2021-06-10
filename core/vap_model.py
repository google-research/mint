"""The video-audio multi-modal model."""

import copy
from mint.core import base_models
from mint.core import multi_modal_model
from mint.core import multi_modal_model_util
from mint.protos import model_pb2
import tensorflow as tf


class VAPModel(multi_modal_model.MultiModalModel):
  """Multi-Modal Video Audio Pose model."""

  def __init__(self, config, is_training):
    """Initializer for VAPModel."""
    super(VAPModel, self).__init__(is_training)
    self.config = copy.deepcopy(config)
    self.is_training = is_training

    self.cross_modal_layer = None
    self.pose_output_layer = None
    (self.feature_to_model, self.feature_to_params, self.feature_to_preprocessor
    ) = multi_modal_model_util.build_modalities_model(self.config.modality)

    transformer_config = self.feature_to_model["visual"]["transformer_layer"]
    patch_embedding_config = self.feature_to_model["visual"][
        "patch_embed_layer"]
    self.visual_patch_embedding = base_models.PatchEmbedding(
        patch_embedding_config)
    self.visual_pos_embedding = base_models.PositionEmbedding(
        self.feature_to_params["visual"]["sequence_length"],
        transformer_config.hidden_size)
    self.visual_transformer = base_models.Transformer(
        hidden_size=transformer_config.hidden_size,
        num_hidden_layers=transformer_config.num_hidden_layers,
        num_attention_heads=transformer_config.num_attention_heads,
        intermediate_size=transformer_config.intermediate_size,
        initializer_range=transformer_config.initializer_range)
    if self.config.HasField("cross_modal_model"):
      self.cross_modal_layer = base_models.CrossModalLayer(
          self.config.cross_modal_model, is_training=self.is_training)

    if self.config.task == model_pb2.VAPModel.TaskType.GENRE_CLASSIFICATION:
      self.cls_token = self.add_weight(
          "cls_token",
          shape=[1, 1, transformer_config.hidden_size],
          initializer=tf.keras.initializers.RandomNormal(),
          dtype=tf.float32)
      self.softmax_layer = tf.keras.layers.Softmax()
      self.output_classification_layer = tf.keras.layers.Dense(
          self.config.target_num_categories)

  def call(self, inputs):
    """Predict outputs from inputs."""
    # Run preprocessing.
    for feature_name in self.feature_to_preprocessor:
      preprocessed_output = inputs[f"{feature_name}_input"]
      process_layers = self.feature_to_preprocessor[feature_name]
      for layer in process_layers:
        preprocessed_output = layer(preprocessed_output)
      # Overwrite the input with preprocessed output
      inputs[f"{feature_name}_input"] = preprocessed_output

    visual_features = self.visual_patch_embedding(inputs["visual_input"])
    feature_shape = tf.shape(visual_features)
    cls_token = tf.repeat(self.cls_token, repeats=feature_shape[0], axis=0)
    visual_features = tf.concat((cls_token, visual_features), axis=1)
    visual_features = self.visual_pos_embedding(visual_features)
    visual_features = self.visual_transformer(visual_features)

    if self.config.task == model_pb2.VAPModel.TaskType.GENRE_CLASSIFICATION:
      output = self.output_classification_layer(visual_features[:, 0])
      output = self.softmax_layer(output)

    return output

  def loss(self, target, pred):
    loss = None
    if self.config.task == model_pb2.VAPModel.TaskType.GENRE_CLASSIFICATION:
      loss = tf.keras.losses.categorical_crossentropy(target, pred)
    return loss

  def get_metrics(self):
    """Compute metrics based on the eval results and metrics."""
    metrics = []
    if self.config.task == model_pb2.VAPModel.TaskType.GENRE_CLASSIFICATION:
      metrics.append(tf.keras.metrics.CategoricalAccuracy())
      metrics.append(tf.keras.metrics.TopKCategoricalAccuracy(k=5))
    return metrics
