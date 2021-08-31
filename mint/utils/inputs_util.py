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
"""Util functions for creating inputs."""
import tensorflow as tf


def get_modality_to_param_dict(dataset_config):
  """Creates a map from modality name to modality parameters."""

  modality_to_param_dict = {}
  for modality in dataset_config.modality:
    modality_type = modality.WhichOneof("modality")
    if modality_type == "general_modality":
      modality = modality.general_modality
      modality_to_param_dict[modality.feature_name] = {}
      modality_to_param_dict[
          modality.feature_name]["feature_dim"] = modality.dimension
      modality_to_param_dict[modality.feature_name]["input_length"] = int(
          dataset_config.input_length_sec * modality.sample_rate)
      modality_to_param_dict[modality.feature_name]["target_length"] = int(
          dataset_config.target_length_sec * modality.sample_rate)
      modality_to_param_dict[modality.feature_name]["target_shift"] = int(
          dataset_config.target_shift_sec * modality.sample_rate)
      modality_to_param_dict[
          modality.feature_name]["sample_rate"] = modality.sample_rate
      # Raw image specific parameters.
      modality_to_param_dict[modality.feature_name]["resize"] = modality.resize
      modality_to_param_dict[
          modality.feature_name]["crop_size"] = modality.crop_size
    elif modality_type == "raw_text":
      modality_to_param_dict[modality.feature_name] = {}
    else:
      raise ValueError("Unknown modality type:", modality_type)
  return modality_to_param_dict


def preprocess_labels(example, dataset_config):
  """Preprocess labels to one_hot encoding."""
  target = example.pop(dataset_config.data_target_field)
  example["target"] = tf.reduce_max(
      tf.one_hot(
          tf.sparse.to_dense(target),
          depth=dataset_config.target_num_categories),
      axis=0)
  return example


def fact_preprocessing(example, modality_to_params, is_training):
  """Preprocess data for FACT model."""
  motion_seq_length = tf.shape(example["motion_sequence"])[0]
  motion_input_length = modality_to_params["motion"]["input_length"]
  motion_target_length = modality_to_params["motion"]["target_length"]
  motion_target_shift = modality_to_params["motion"]["target_shift"]
  audio_input_length = modality_to_params["audio"]["input_length"]

  motion_dim = modality_to_params["motion"]["feature_dim"]
  audio_dim = modality_to_params["audio"]["feature_dim"]

  # Pad the input motion translation from 3-dim to 9-dim.
  motion_dim += 6
  example["motion_sequence"] = tf.pad(example["motion_sequence"],
                                      [[0, 0], [6, 0]])
  if is_training:
    windows_size = tf.maximum(motion_input_length,
                              motion_target_shift + motion_target_length)
    windows_size = tf.maximum(windows_size, audio_input_length)
    # the start frame id for this window.
    start = tf.random.uniform([],
                              0,
                              motion_seq_length - windows_size + 1,
                              dtype=tf.int32)
  else:
    start = 0

  # motion input: [start, start + motion_input_length)
  example["motion_input"] = example["motion_sequence"][start:start +
                                                       motion_input_length, :]
  example["motion_input"].set_shape([motion_input_length, motion_dim])
  if is_training:
    # motion target: [start + shift, start + shift + motion_target_length)
    example["target"] = example["motion_sequence"][start +
                                                   motion_target_shift:start +
                                                   motion_target_shift +
                                                   motion_target_length, :]
    example["target"].set_shape([motion_target_length, motion_dim])
  del example["motion_sequence"]

  if is_training:
    # audio input: [start, start + audio_input_length)
    example["audio_input"] = example["audio_sequence"][start:start +
                                                      audio_input_length, :]
    example["audio_input"].set_shape([audio_input_length, audio_dim])
  else:
    example["audio_input"] = example["audio_sequence"]
  del example["audio_sequence"]
  return example

