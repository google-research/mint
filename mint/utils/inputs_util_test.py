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
