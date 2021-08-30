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
"""Data provider."""
import functools
from mint.utils import inputs_util
import tensorflow as tf


def create_input(train_eval_config,
                 dataset_config,
                 num_cpu_threads=2,
                 is_training=True,
                 use_tpu=False):
  """Create batched input data.

  Args:
    train_eval_config: A train or eval config.
    dataset_config: A dataset config.
    num_cpu_threads: Number of cpu threads for dataset reading.
    is_training: Whether this is training stage.
    use_tpu: Whether or not provide inputs for TPU.

  Returns:
    ds: A tf.data.Dataset, with the following features:
      features_{audio, motion}, masked_features_{audio, motion},
      masked_positions_{audio, motion}, mask_{audio, motion}.
  """
  batch_size = train_eval_config.batch_size
  data_files = tf.io.gfile.glob(dataset_config.data_files)
  name_to_features = {}
  modality_to_params = inputs_util.get_modality_to_param_dict(dataset_config)

  for modality in modality_to_params:
    if modality == "visual":
      name_to_features.update({
          f"{modality}_sequence": tf.io.VarLenFeature(tf.string),
          f"{modality}_sequence_shape": tf.io.FixedLenFeature([1], tf.int64),
      })
    else:
      name_to_features.update({
          f"{modality}_sequence": tf.io.VarLenFeature(tf.float32),
          f"{modality}_sequence_shape": tf.io.FixedLenFeature([2], tf.int64),
          f"{modality}_name": tf.io.FixedLenFeature([], tf.string),
      })

  if dataset_config.data_target_field:
    name_to_features.update(
        {dataset_config.data_target_field: tf.io.VarLenFeature(tf.int64)})

  # For training, we want a lot of parallel reading and shuffling.
  # For eval, we want no shuffling and parallel reading doesn't matter.
  if is_training:
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(data_files))
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=tf.data.AUTOTUNE,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False)
    ds = ds.shuffle(100).repeat()
  else:
    ds = tf.data.TFRecordDataset(data_files)
    # Since we evaluate for a fixed number of steps we don't want to encounter
    # out-of-range exceptions.
    ds = ds.repeat(1)

  # Function to decode a record
  def _decode_and_reshape_record(record):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(record, name_to_features)
    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.dtypes.cast(t, tf.int32)
      example[name] = t

    # Sparse to dense
    for modality in modality_to_params:
      example[f"{modality}_sequence"] = tf.reshape(
          tf.sparse.to_dense(example[f"{modality}_sequence"]),
          example[f"{modality}_sequence_shape"])
    return example

  ds = ds.map(_decode_and_reshape_record, num_parallel_calls=num_cpu_threads)

  # Data preprocessing and augmentation
  for da_step_config in dataset_config.data_augmentation_options:
    da_step_type = da_step_config.WhichOneof("preprocessor")
    if da_step_type == "fact_preprocessor":
      ds = ds.map(
          functools.partial(
              inputs_util.fact_preprocessing,
              modality_to_params=modality_to_params,
              is_training=is_training),
          num_parallel_calls=num_cpu_threads)

  if dataset_config.data_target_field:
    ds = ds.map(
        functools.partial(
            inputs_util.preprocess_labels, dataset_config=dataset_config),
        num_parallel_calls=num_cpu_threads)

  # We must `drop_remainder` on training because the TPU requires fixed
  # size dimensions.
  # If not using TPU, we *don't* want to drop the remainder when eval.
  if use_tpu:
    ds = ds.batch(batch_size, drop_remainder=True)
  else:
    ds = ds.batch(batch_size, drop_remainder=is_training)
  ds = ds.prefetch(1)
  return ds
