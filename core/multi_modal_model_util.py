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


def build_audio_preprocessing(preprocessing_config, feature_to_params):
  """Builds audio preprocessing layer based on config."""
  audio_preprocess_layer = None
  # Preprocessing may change the length.
  if preprocessing_config.WhichOneof("options") == "mel_spec_options":
    options = preprocessing_config.mel_spec_options
    audio_preprocess_layer = base_models.AudioMelSpecPreprocessor(options)
    # Update sequence length.
    audio_input_ms = feature_to_params["audio"][
        "sequence_length"] / options.sample_rate_hz * 1000
    audio_seq_length = int(audio_input_ms / options.fft_frame_step_ms)
    feature_to_params["audio"]["sequence_length"] = audio_seq_length
  return audio_preprocess_layer


def build_preprocessing_layer(preprocessor_config, feature_to_params):
  """Build preprocessing layers."""
  preprocessor_type = preprocessor_config.WhichOneof("preprocessor")
  if preprocessor_type == "audio_preprocessor":
    return build_audio_preprocessing(preprocessor_config.audio_preprocessor,
                                     feature_to_params)


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


# TODO(shanyang) deal with mask, some layers before the transformer changes
# the sequence_length.
def run_modalities_model(features, feature_to_preprocessor, feature_to_model):
  """Run the modalities models."""
  feature_to_output = {}
  # Run preprocessing.
  for feature_name in feature_to_preprocessor:
    preprocessed_output = features[f"{feature_name}_input"]
    process_layers = feature_to_preprocessor[feature_name]
    for layer in process_layers:
      preprocessed_output = layer(preprocessed_output)
    # Overwrite the input with preprocessed output
    features[f"{feature_name}_input"] = preprocessed_output

  # Embed each modality feature using configed model
  for feature_name in feature_to_model:
    output = features[f"{feature_name}_input"]
    model = feature_to_model[feature_name]
    output = model["patch_embed_layer"](output)
    output = model["pos_embed_layer"](output)
    output, _ = model["transformer_layer"](output)
    feature_to_output[feature_name] = output
  return output, feature_to_output
