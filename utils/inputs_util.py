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

  windows_size = tf.maximum(motion_input_length,
                            motion_target_shift + motion_target_length)
  windows_size = tf.maximum(windows_size, audio_input_length)

  # Pad the input motion translation from 3-dim to 9-dim.
  motion_dim += 6
  example["motion_sequence"] = tf.pad(example["motion_sequence"],
                                      [[0, 0], [6, 0]])

  if is_training:
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
  example["motion_mask"] = tf.ones([motion_input_length], dtype=tf.float32)
  example["motion_mask"].set_shape([motion_input_length])
  example["motion_input"].set_shape([motion_input_length, motion_dim])
  # motion target: [start + shift, start + shift + motion_target_length)
  example["target"] = example["motion_sequence"][start +
                                                 motion_target_shift:start +
                                                 motion_target_shift +
                                                 motion_target_length, :]
  example["target"].set_shape([motion_target_length, motion_dim])
  del example["motion_sequence"]

  # audio input: [start, start + audio_input_length)
  example["audio_input"] = example["audio_sequence"][start:start +
                                                     audio_input_length, :]
  example["audio_input"].set_shape([audio_input_length, audio_dim])
  example["audio_mask"] = tf.ones([audio_input_length], dtype=tf.float32)
  example["audio_mask"].set_shape([audio_input_length])
  del example["audio_sequence"]
  return example


def image_preprocessing(example, modality_to_params, is_training):
  """Dataset preprocess function."""
  modality_name = "visual"
  input_length = modality_to_params[modality_name]["input_length"]
  sequence = example.pop(f"{modality_name}_sequence")
  image = tf.io.decode_image(sequence[0])
  resize = modality_to_params[modality_name]["resize"]
  crop_size = modality_to_params[modality_name]["crop_size"]

  image = tf.image.convert_image_dtype(image, dtype=tf.float32)
  if is_training:
    image = tf.image.resize_with_pad(image, resize, resize)
    image = tf.image.random_crop(image, [crop_size, crop_size, 3])
    image = tf.image.random_flip_left_right(image)
  else:
    image = tf.image.resize_with_pad(image, crop_size, crop_size)

  sequence = (image - 127.5) / 127.5

  example[f"{modality_name}_input"] = sequence
  example[f"{modality_name}_mask"] = tf.ones([input_length], dtype=tf.float32)
  example[f"{modality_name}_mask"].set_shape([input_length])

  return example
