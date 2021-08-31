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
"""Multi-modal model util functions."""

from mint.core import base_models


def build_preprocessing_layer(preprocessor_config, feature_to_params):
  """Build preprocessing layers."""
  pass


def build_modalities_model(modality_config):
  """Process the parameters in the modality config."""
  # Initialize the dictionaries.
  feature_to_model = {}
  feature_to_params = {}
  feature_to_preprocessor = {}

  for modality in modality_config:
    feature_name = modality.feature_name
    feature_to_params[feature_name] = {}
    feature_to_model[feature_name] = {}
    feature_to_preprocessor[feature_name] = []

    # Process parameters.
    feature_to_params[feature_name][
        "sequence_length"] = modality.sequence_length
    feature_to_params[feature_name]["feature_dim"] = modality.feature_dim

    # Process preprocessors.
    for preprocessor in modality.preprocessor:
      feature_to_preprocessor[feature_name].append(
          build_preprocessing_layer(preprocessor, feature_to_params))

    # Process models configs.
    for model in modality.model:
      model_type = model.WhichOneof("model")
      if model_type == "transformer":
        model_config = model.transformer
        feature_to_model[feature_name]["transformer_layer"] = model_config
      if model_type == "patch_embedding":
        model_config = model.patch_embedding
        feature_to_model[feature_name]["patch_embed_layer"] = model_config
  return feature_to_model, feature_to_params, feature_to_preprocessor
