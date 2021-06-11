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
"""Tests for fact_model."""

from mint.core import fact_model
from mint.protos import model_pb2
import tensorflow as tf


class FactModelTest(tf.test.TestCase):

  def test_run_fact_model(self):
    is_training = True
    config = model_pb2.FACTModel()
    motion_modality = model_pb2.Modality()
    motion_modality.sequence_length = 120
    motion_modality.feature_dim = 225
    motion_modality.feature_name = "motion"
    motion_model = model_pb2.ModalityModel()
    motion_model.transformer.num_hidden_layers = 2
    motion_modality.model.append(motion_model)
    config.modality.append(motion_modality)
    audio_modality = model_pb2.Modality()
    audio_modality.sequence_length = 240
    audio_modality.feature_name = "audio"
    audio_modality.feature_dim = 35
    audio_model = model_pb2.ModalityModel()
    audio_model.transformer.num_hidden_layers = 2
    audio_modality.model.append(audio_model)
    config.modality.append(audio_modality)
    config.cross_modal_model.modality_a = "motion"
    config.cross_modal_model.modality_b = "audio"
    config.cross_modal_model.transformer.num_hidden_layers = 12
    config.cross_modal_model.output_layer.out_dim = 225
    model = fact_model.FACTModel(config, is_training)
    features = {
        "motion_input": tf.ones([2, 120, 225], dtype=tf.float32),
        "motion_mask": tf.ones([2, 120], dtype=tf.float32),
        "audio_input": tf.ones([2, 240, 35], dtype=tf.float32),
        "audio_mask": tf.ones([2, 240], dtype=tf.float32)
    }
    output = model(features)
    self.assertAllEqual(output.shape, (2, 360, 225))


if __name__ == "__main__":
  tf.test.main()
