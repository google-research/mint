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
"""Abstract multi-modal model."""
import abc

import tensorflow as tf


class MultiModalModel(tf.keras.Model):
  """Abstract base class for multi-modal video understanding."""

  def __init__(self, is_training):
    """Constructor."""
    super(MultiModalModel, self).__init__()
    self.is_training = is_training
    # Input, output, model, parameter, preprocessor for each modality.
    self.feature_to_output = {}
    self.feature_to_model = {}
    self.feature_to_params = {}
    self.feature_to_preprocessor = {}

  @abc.abstractmethod
  def call(self, inputs):
    """Execute the model with inputs."""
    raise NotImplementedError

  @abc.abstractmethod
  def restore_from_objects(self, checkpoint_type):
    """Restore model from trackable objects."""
    raise NotImplementedError

  @abc.abstractmethod
  def loss(self, features, pred_dict, target_dict):
    """Compute losses."""
    raise NotImplementedError

  @abc.abstractmethod
  def predict(self, features, dataset_config):
    """Produce predictions given the features."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_metrics(self, eval_config):
    """Get model specific metrics given the eval_config."""
    raise NotImplementedError

  @abc.abstractmethod
  def compute_metrics(self, eval_dict, eval_metrics, **kwargs):
    """Compute metrics."""
    raise NotImplementedError

  def visualization(self, eval_config, eval_dataset_config, **kwargs):
    """Model specific visualization."""
    pass
