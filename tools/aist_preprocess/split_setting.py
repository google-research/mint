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
"""AIST++ train/val/test set split setting."""
import re

MUSIC_ID_TESTVAL = '(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)'
MOTION_GEN_ID_TESTVAL = '.*_sBM_.*_(mBR|mPO|mLO|mMH|mLH|mHO|mWA|mKR|mJS|mJB).*_(ch01|ch02)'
MOTION_GEN_ID_TESTVAL_PAIRED = '.*_sBM_.*_(mBR0|mPO1|mLO2|mMH3|mLH4|mHO5|mWA0|mKR2|mJS3|mJB5)_(ch01|ch02)'
MOTION_GEN_ID_TESTVAL_UNPAIRED = '.*_sBM_.*_(mBR5|mPO5|mLO5|mMH0|mLH0|mHO0|mWA5|mKR5|mJS0|mJB0)_(ch01|ch02)'

MOTION_PRED_ID_VAL = '.*_ch01'
MOTION_PRED_ID_TEST = '.*_ch02'


def get_testval_music_id(video_name):
  """Get the test / val music name for a specific video name."""
  music_id = video_name.split('_')[-2]
  if 'mBR' in music_id:
    return 'mBR0'
  elif 'mPO' in music_id:
    return 'mPO1'
  elif 'mLO' in music_id:
    return 'mLO2'
  elif 'mMH' in music_id:
    return 'mMH3'
  elif 'mLH' in music_id:
    return 'mLH4'
  elif 'mHO' in music_id:
    return 'mHO5'
  elif 'mWA' in music_id:
    return 'mWA0'
  elif 'mKR' in music_id:
    return 'mKR2'
  elif 'mJS' in music_id:
    return 'mJS3'
  elif 'mJB' in music_id:
    return 'mJB5'
  else:
    assert False, video_name


def get_split(video_names, task, subset, **kwargs):
  """Get the subset split of AIST++ dataset."""
  assert task in ['generation', 'prediction']
  assert subset in ['train', 'val', 'test', 'all']

  split = {
      'video_names': [],
      'music_names': [],
      'is_paired':
          kwargs['is_paired'] if 'is_paired' in kwargs else None,
  }

  if task == 'prediction' and subset == 'val':
    split['video_names'] = [
        video_name for video_name in video_names
        if re.match(MOTION_PRED_ID_VAL, video_name)
    ]

  elif task == 'prediction' and subset == 'test':
    split['video_names'] = [
        video_name for video_name in video_names
        if re.match(MOTION_PRED_ID_TEST, video_name)
    ]

  elif task == 'prediction' and subset == 'train':
    split['video_names'] = [
        video_name for video_name in video_names
        if (not re.match(MOTION_PRED_ID_VAL, video_name) and
            not re.match(MOTION_PRED_ID_TEST, video_name))]

  elif task == 'generation' and (subset == 'val' or subset == 'test'):
    assert split['is_paired'] in [True, False]
    if split['is_paired']:
      split['video_names'] = [
          video_name for video_name in video_names
          if re.match(MOTION_GEN_ID_TESTVAL_PAIRED, video_name)
      ]
      split['music_names'] = [
          video_name.split('_')[-2] for video_name in split['video_names']
      ]
    else:
      split['video_names'] = [
          video_name for video_name in video_names
          if re.match(MOTION_GEN_ID_TESTVAL_UNPAIRED, video_name)
      ]
      split['music_names'] = [
          get_testval_music_id(video_name)
          for video_name in split['video_names']
      ]

  elif task == 'generation' and subset == 'train':
    split['video_names'] = [
        video_name for video_name in video_names
        if (not re.match(MOTION_GEN_ID_TESTVAL, video_name) and
            not re.match(MUSIC_ID_TESTVAL, video_name.split('_')[-2]))]
    split['music_names'] = [
        video_name.split('_')[-2] for video_name in split['video_names']
    ]

  else:
    raise NotImplementedError

  return split
