"""Tests for google3.research.vision.couchpotato.choreo_generation.models.multi_modal.utils.inputs_util."""


from mint.protos import dataset_pb2
from mint.utils import inputs_util
import tensorflow as tf


class InputsUtilTest(tf.test.TestCase):

  def test_get_modality_to_params(self):
    dataset_config = dataset_pb2.Dataset()
    dataset_config.window_type = dataset_pb2.Dataset.BEGINNING
    dataset_config.input_length_sec = 1.0
    dataset_config.target_length_sec = 0.5
    dataset_config.target_shift_sec = 0.2
    motion = dataset_pb2.GeneralModality()
    motion.feature_name = "motion"
    motion.dimension = 34
    motion.sample_rate = 10
    visual = dataset_pb2.GeneralModality()
    visual.feature_name = "visual"
    visual.dimension = 1024
    visual.sample_rate = 20
    motion_modality = dataset_config.modality.add()
    motion_modality.general_modality.CopyFrom(motion)
    visual_modality = dataset_config.modality.add()
    visual_modality.general_modality.CopyFrom(visual)

    modality_to_params = inputs_util.get_modality_to_param_dict(dataset_config)
    self.assertIn("motion", modality_to_params)
    self.assertEqual(modality_to_params["motion"]["input_length"], 10)
    self.assertEqual(modality_to_params["motion"]["target_length"], 5)
    self.assertEqual(modality_to_params["motion"]["target_shift"], 2)
    self.assertIn("visual", modality_to_params)
    self.assertEqual(modality_to_params["visual"]["input_length"], 20)
    self.assertEqual(modality_to_params["visual"]["target_length"], 10)
    self.assertEqual(modality_to_params["visual"]["target_shift"], 4)


if __name__ == "__main__":
  tf.test.main()
