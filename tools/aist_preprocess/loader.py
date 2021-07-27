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
"""A library for loading AIST++ dataset."""
import glob
import json
import os
import pickle

from absl import logging
import numpy as np


_LOG_FILE_JSON_EMPTY = 'json_empty_list.txt'
_LOG_FILE_JSON_MISSING = 'json_missing_list.txt'
_LOG_FILE_JSON_ZERO_PERSON = 'json_zero_person_list.txt'
_LOG_FILE_JSON_MULTI_PERSON = 'json_multi_person_list.txt'


def write_log(log_dir, filename, content):
  if log_dir is None:
    return
  os.makedirs(log_dir, exist_ok=True)
  path = os.path.join(log_dir, filename)
  with open(path, 'a') as f:
    f.write(content + '\n')


def array_nan(shape, dtype=np.float32):
  array = np.empty(shape, dtype=dtype)
  array[:] = np.nan
  return array


def get_video_name(video_name, view):
  """Get AIST video name for a specific view."""
  splits = video_name.split('_')
  splits[2] = view
  return '_'.join(splits)


def get_audio_name(video_name):
  """Get AIST audio name for a specific video."""
  splits = video_name.split('_')
  return splits[-2]


def get_tempo(audio_name):
  """Get tempo (BPM) for a audio by parsing audio name."""
  assert len(audio_name) == 4
  if audio_name[0:3] in [
      'mBR', 'mPO', 'mLO', 'mMH', 'mLH', 'mWA', 'mKR', 'mJS', 'mJB'
  ]:
    return int(audio_name[3]) * 10 + 80
  elif audio_name[0:3] == 'mHO':
    return int(audio_name[3]) * 5 + 110
  else:
    assert False, audio_name


def load_pkl(path, keys=None):
  """Load AIST++ annotations from pkl file."""
  with open(path, 'rb') as f:
    data = pickle.load(f)
    annos = data['pred_results']
  assert annos, f'data {path} has empty annotations'
  out = {}

  # smpl related
  if 'smpl_loss' in data and ('smpl_loss' in keys if keys else True):
    # a single float
    out.update({'smpl_loss': data['smpl_loss']})

  if 'smpl_joints' in annos[0] and ('smpl_joints' in keys if keys else True):
    # [nframes, 24, 3]
    out.update({
        'smpl_joints':
            np.stack([anno['smpl_joints'] for anno in annos])
            [:, :24, :].astype(np.float32)
    })
  if 'smpl_pose' in annos[0] and ('smpl_poses' in keys if keys else True):
    # [nframes, 24, 3]
    out.update({
        'smpl_poses':
            np.stack([anno['smpl_pose'] for anno in annos]
                    ).reshape(-1, 24, 3).astype(np.float32)
    })
  if 'smpl_shape' in annos[0] and ('smpl_shape' in keys if keys else True):
    # [nframes, 10]
    out.update({
        'smpl_shape':
            np.stack([anno['smpl_shape'] for anno in annos]).astype(np.float32)
    })
  if 'scaling' in annos[0] and ('smpl_scaling' in keys if keys else True):
    # [nframes, 1]
    out.update({
        'smpl_scaling':
            np.stack([anno['scaling'] for anno in annos]).astype(np.float32)
    })
  if 'transl' in annos[0] and ('smpl_trans' in keys if keys else True):
    # [nframes, 3]
    out.update({
        'smpl_trans':
            np.stack([anno['transl'] for anno in annos]).astype(np.float32)
    })
  if 'verts' in annos[0] and ('smpl_verts' in keys if keys else True):
    # [nframes, 6890, 3]
    out.update({
        'smpl_verts':
            np.stack([anno['verts'] for anno in annos]).astype(np.float32)
    })

  # 2D and 3D keypoints
  if 'keypoints2d' in annos[0] and ('smpl_verts' in keys if keys else True):
    # [9, nframes, 17, 3]
    out.update({
        'keypoints2d':
            np.stack([anno['keypoints2d'] for anno in annos],
                     axis=1).astype(np.float32)
    })
  if 'keypoints3d' in annos[0] and ('keypoints3d' in keys if keys else True):
    # [nframes, 17, 3]
    out.update({
        'keypoints3d':
            np.stack([anno['keypoints3d'] for anno in annos]).astype(np.float32)
    })
  if 'keypoints3d_optim' in annos[0] and ('keypoints3d_optim' in keys
                                          if keys else True):
    # [nframes, 17, 3]
    out.update({
        'keypoints3d_optim':
            np.stack([anno['keypoints3d_optim'] for anno in annos]
                    ).astype(np.float32)
    })

  # timestamps for each frame, in ms.
  if 'timestamp' in annos[0] and ('timestamps' in keys if keys else True):
    # [nframes,]
    out.update({
        'timestamps':
            np.stack([anno['timestamp'] for anno in annos]).astype(np.int32)
    })

  # human detection score
  if 'det_scores' in annos[0] and ('det_scores' in keys if keys else True):
    # [9, nframes]
    out.update({
        'det_scores':
            np.stack([anno['det_scores'] for anno in annos],
                     axis=1).astype(np.int32)
    })

  return out


def load(path):
  """Load AIST++ annotations."""
  if path[-4:] == '.pkl':
    return load_pkl(path)
  else:
    assert False, f'{path} should be a pkl file'


def load_keypoints2d_file(path, n_joints=17, log_dir=None):
  """load 2D keypoints file from centernet results.

  Only one person is extracted from the results. If there are multiple
  persons in the prediction results, we select the one with the highest
  detection score.

  Args:
    path: the json file path.
    n_joints: number of joints. e.g., 17.
    log_dir: dictionary to store the log.

  Returns:
    A `np.array` with the shape of [n_joints, 3].
  """
  with open(path, 'r') as f:
    try:
      data = json.load(f)
    except Exception as e:  # pylint: disable=broad-except
      logging.warning(e)
      write_log(log_dir, _LOG_FILE_JSON_EMPTY, content=path)
      keypoint = array_nan((n_joints, 3), dtype=np.float32)
      return keypoint
  detection_scores = np.array(data['detection_scores'])
  keypoints = np.array(data['keypoints']).reshape((-1, n_joints, 3))

  # the detection results may contain zero person or multiple people.
  if detection_scores.shape[0] == 0:
    write_log(log_dir, _LOG_FILE_JSON_ZERO_PERSON, content=path)
    keypoint = array_nan((n_joints, 3), dtype=np.float32)
    return keypoint, 0.0
  elif detection_scores.shape[0] == 1:
    keypoint = keypoints[0]
    det_score = detection_scores[0]
    return keypoint, det_score
  else:
    write_log(log_dir, _LOG_FILE_JSON_MULTI_PERSON, content=path)
    idx = np.argmax(detection_scores)
    keypoint = keypoints[idx]
    det_score = detection_scores[idx]
    return keypoint, det_score


def load_keypoints2d(data_dir, video_name, views=None, log_dir=None):
  """Load 2D keypoints predictions from centernet."""

  # load single view or all views
  if views is None:
    video_names = [video_name]
  else:
    video_names = [get_video_name(video_name, view) for view in views]

  # in case frames are missing, we first scan all views to get a union
  # of timestamps.
  paths_cache = {}
  timestamps = []
  for video_name in video_names:
    paths = sorted(glob.glob(os.path.join(data_dir, video_name, '*.json')))
    paths_cache[video_name] = paths
    timestamps += [int(p.split('.')[0].split('_')[-1]) for p in paths]
  timestamps = sorted(list(set(timestamps)))

  # then we load all frames according to timestamps.
  # if some frames do not exist, we mark them as np.nan
  keypoints2d = []
  det_scores = []
  for video_name in video_names:
    paths = [
        os.path.join(data_dir, video_name, f'{video_name}_{ts}.json')
        for ts in timestamps
    ]
    keypoints2d_per_view = []
    det_scores_per_view = []
    for path in paths:
      if path in paths_cache[video_name]:
        keypoint, det_score = load_keypoints2d_file(
            path, n_joints=17, log_dir=log_dir)
        keypoints2d_per_view.append(keypoint)
        det_scores_per_view.append(det_score)
      else:
        write_log(log_dir, _LOG_FILE_JSON_MISSING, content=path)
        keypoints2d_per_view.append(array_nan((17, 3), dtype=np.float32))
        det_scores_per_view.append(0.0)
    keypoints2d.append(keypoints2d_per_view)
    det_scores.append(det_scores_per_view)

  keypoints2d = np.array(keypoints2d, dtype=np.float32)
  det_scores = np.array(det_scores, dtype=np.float32)
  return keypoints2d, det_scores, timestamps
