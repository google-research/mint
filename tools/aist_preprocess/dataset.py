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
"""AIST++ Dataset Class."""
import os

from absl import logging
from mint.tools.aist_preprocess import loader
from mint.tools.aist_preprocess import raw_processing


class AISTDataset(object):
  """A dataset class for loading, processing and plotting AIST++."""

  def __init__(self,
               motion_dir,
               audio_dir,
               audio_feature_dir,
               ignore_list_file=None):
    self.motion_dir = motion_dir
    self.audio_dir = audio_dir
    self.audio_feature_dir = audio_feature_dir

    self.video_names = sorted([
        name.split('.')[0] for name in os.listdir(motion_dir)])
    self.audio_names = sorted([
        name.split('.')[0] for name in os.listdir(audio_dir)])

    # ignore video names: those videos have problematic 3D keypoints.
    if ignore_list_file:
      with open(ignore_list_file, 'r') as f:
        ignore_video_names = [l.strip() for l in f.readlines()]
        self.video_names = [
            name for name in self.video_names if name not in ignore_video_names
        ]

  def __len__(self):
    return len(self.video_names)

  def __getitem__(self, index):
    video_name = self.video_names[index]
    return self.get_item(video_name)

  def get_item(self, video_name, only_motion=False, verbose=False):
    """Get a motion-audio pair as a data item."""
    audio_name = loader.get_audio_name(video_name)
    audio_tempo = loader.get_tempo(audio_name)

    motion_path = os.path.join(self.motion_dir, f'{video_name}.pkl')
    audio_path = os.path.join(self.audio_dir, f'{audio_name}.wav')
    audio_feature_path = os.path.join(self.audio_feature_dir,
                                      f'{audio_name}.pkl')
    motion_data = loader.load_pkl(
        motion_path,
        keys=['smpl_poses', 'smpl_scaling', 'smpl_trans'])
    motion_data['smpl_trans'] /= motion_data['smpl_scaling']
    del motion_data['smpl_scaling']

    for k, v in motion_data.items():
      # interpolate 30 FPS motion data into 60 FPS
      motion_data[k] = raw_processing.interpolate2x(v, axis=0)

    if only_motion:  # save time
      audio_data = None
    else:
      audio_data = raw_processing.audio_features_all(
          cache_path=audio_feature_path, tempo=audio_tempo, concat=False)

    if verbose:
      logging.info('---- loading AIST++ data item: %s ----', video_name)
      for k, v in motion_data.items():
        logging.info('[motion] %s: %s', k, v.shape)
      for k, v in audio_data.items():
        logging.info('[audio] %s: %s', k, v.shape)

    return {
        'video_name': video_name,
        'audio_name': audio_name,
        'motion_path': motion_path,
        'audio_path': audio_path,
        'motion_data': motion_data,
        'audio_data': audio_data,
        'audio_tempo': audio_tempo,
    }
