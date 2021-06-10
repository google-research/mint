"""Tests for mint.core.vap_model."""

from mint.core import vap_model
from mint.protos import model_pb2
import tensorflow as tf


class VAPModelTest(tf.test.TestCase):

  def test_run_genre_classification_model(self):
    is_training = True
    config = model_pb2.VAPModel()
    visual_modality = model_pb2.Modality()
    visual_modality.sequence_length = 197
    visual_modality.feature_name = "visual"
    visual_model = model_pb2.ModalityModel()
    visual_model.transformer.hidden_size = 768
    patch_layer = model_pb2.ModalityModel()
    patch_layer.patch_embedding.hidden_size = 768
    visual_modality.model.append(visual_model)
    visual_modality.model.append(patch_layer)
    config.modality.append(visual_modality)

    config.task = model_pb2.VAPModel.TaskType.GENRE_CLASSIFICATION
    config.target_num_categories = 5
    # Build model
    model = vap_model.VAPModel(config, is_training)
    features = {
        "visual_input": tf.ones([2, 224, 224, 3], dtype=tf.float32),
        "visual_mask": tf.ones([2, 20], dtype=tf.float32),
        "target": tf.ones([2, 5], dtype=tf.float32),
    }
    output = model(features)
    self.assertAllEqual(output.shape, (2, 5))


if __name__ == "__main__":
  tf.test.main()
