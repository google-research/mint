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
"""Fucntions for processing motion and audio raw data."""
import collections
import pickle

import librosa
import numpy as np
import scipy
import scipy.interpolate


_FPS = 60
_HOP_LENGTH = 512
_SR = _FPS * _HOP_LENGTH
_EPS = 1e-6


# ===========================================================
# Functions for processing audio data.
# ===========================================================
def audio_features_all(path=None, cache_path=None, tempo=120.0, concat=False):
  """Load all the audio features."""
  if cache_path:
    with open(cache_path, 'rb') as f:
      features = pickle.load(f)
  else:
    data = audio_load(path)
    envelope = audio_envelope(data=data)

    # tempogram = audio_tempogram(envelope=envelope)
    mfcc = audio_mfcc(data=data)
    chroma = audio_chroma(data=data)
    _, peak_onehot = audio_peak_onehot(envelope=envelope)
    _, beat_onehot, _ = audio_beat_onehot(envelope=envelope, start_bpm=tempo)

    features = collections.OrderedDict({
        'envelope': envelope[:, None],
        # 'tempogram': tempogram,
        'mfcc': mfcc,
        'chroma': chroma,
        'peak_onehot': peak_onehot,
        'beat_onehot': beat_onehot,
    })
    with open(cache_path, 'wb') as f:
      pickle.dump(features, f, protocol=pickle.HIGHEST_PROTOCOL)

  if concat:
    return np.concatenate([v for k, v in features.items()], axis=1)
  else:
    return features


def audio_load(path):
  """Load raw audio data."""
  data, _ = librosa.load(path, sr=_SR)
  return data


def audio_envelope(path=None, data=None):
  """Calculate raw audio envelope."""
  assert (path is not None) or (data is not None)
  if data is None:
    data = audio_load(path)
  envelope = librosa.onset.onset_strength(data, sr=_SR)
  return envelope  # (seq_len,)


def audio_tempogram(path=None, envelope=None, win_length=384):
  """Calculate audio feature: tempogram."""
  assert (path is not None) or (envelope is not None)
  if envelope is None:
    envelope = audio_envelope(path)
  tempogram = librosa.feature.tempogram(
      onset_envelope=envelope.flatten(), sr=_SR, hop_length=_HOP_LENGTH,
      win_length=win_length)
  return tempogram.T  # (seq_len, 384)


def audio_mfcc(path=None, data=None, m_mfcc=20):
  """Calculate audio feature: mfcc."""
  assert (path is not None) or (data is not None)
  if data is None:
    data = audio_load(path)
  mfcc = librosa.feature.mfcc(data, sr=_SR, n_mfcc=m_mfcc)
  return mfcc.T  # (seq_len, 20)


def audio_chroma(path=None, data=None, n_chroma=12):
  """Calculate audio feature: chroma."""
  assert (path is not None) or (data is not None)
  if data is None:
    data = audio_load(path)
  chroma = librosa.feature.chroma_cens(
      data, sr=_SR, hop_length=_HOP_LENGTH, n_chroma=n_chroma)
  return chroma.T  # (seq_len, 12)


def audio_peak_onehot(path=None, envelope=None):
  """Calculate audio onset peaks."""
  assert (path is not None) or (envelope is not None)
  if envelope is None:
    envelope = audio_envelope(path=path)
  peak_idxs = librosa.onset.onset_detect(
      onset_envelope=envelope.flatten(), sr=_SR, hop_length=_HOP_LENGTH)
  peak_onehot = np.zeros_like(envelope, dtype=bool)
  peak_onehot[peak_idxs] = 1
  return envelope, peak_onehot  # (seq_len,) (seq_len,)


def audio_beat_onehot(path=None, envelope=None, start_bpm=120.0, tightness=100):
  """Calculate audio beats."""
  assert (path is not None) or (envelope is not None)
  if envelope is None:
    envelope = audio_envelope(path=path)
  tempo, beat_idxs = librosa.beat.beat_track(
      onset_envelope=envelope, sr=_SR, hop_length=_HOP_LENGTH,
      start_bpm=start_bpm, tightness=tightness)
  beat_onehot = np.zeros_like(envelope, dtype=bool)
  beat_onehot[beat_idxs] = 1
  return envelope, beat_onehot, tempo  # (seq_len,) (seq_len,) float


# ===========================================================
# Functions for processing motion data.
# ===========================================================
def interpolate2x(data, axis, kind='cubic'):
  """Double the size of a sequence data along a specific axis.

  This function is useful for interpolate the 30 FPS motion sequence into
  60 FPS.

  Args:
    data: np array with any shape.
    axis: along with axis to interpolate.
    kind: interpolation function used by `scipy.interpolate.interp1d`.

  Returns:
    np array of the interpolated data. The size along `axis` has been doubled.
  """
  assert len(data.shape) > axis, (
      f'axis={axis} is out of range! data shape is {data.shape}')
  output_size = data.shape[axis] * 2
  return interpolate(data, axis, output_size, kind=kind)


def interpolate(data, axis, output_size, kind='cubic'):
  """Interpolate a sequence data along a specific axis.

  This function is useful for interpolate the 30 FPS motion sequence into
  60 FPS and vise versa.

  Args:
    data: np array with any shape.
    axis: along with axis to interpolate.
    output_size: the final size of that axis after interpolation.
    kind: interpolation function used by `scipy.interpolate.interp1d`.

  Returns:
    np array of the interpolated data.
  """
  assert len(data.shape) > axis, (
      f'axis={axis} is out of range! data shape is {data.shape}')
  input_size = data.shape[axis]
  x = np.arange(0, input_size)
  fit = scipy.interpolate.interp1d(x, data, axis=axis, kind=kind)
  output = fit(np.linspace(0, input_size-1, output_size))
  return output
