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
"""Metrics to measure the sequence generation accuracy."""
import numpy as np
from scipy import linalg
from mint.core import base_model_util
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg

# See https://github.com/google/aistplusplus_api/ for installation 
from aist_plusplus.features.kinetic import extract_kinetic_features
from aist_plusplus.features.manual import extract_manual_features


class EulerAnglesError(tf.keras.metrics.Metric):
  """Metric to measure positional accuracy.

  This metric measures the difference between two poses in terms of
  euler angles. The metrics have been used in Motion Prediction work such as
  http://arxiv.org/abs/2004.08692.
  """

  def __init__(self, num_joints):
    super(EulerAnglesError, self).__init__(name='EulerAnglesError')
    self.num_joints = num_joints
    self.euler_errors = self.add_weight(
        name='euler_errors', initializer='zeros')

  def update_state(self, target, pred):
    """Update metrics.

    Args:
      target: float32 tensor of shape [batch, sequence_length, (num_joints+1)*9]
        the groundtruth motion vector with the first 9 dim as the translation.
      pred: float32 tensor of shape [batch, sequence_length, (num_joints+1)*9]
        the predicted motion vector with the first 9 dim as the translation.
    """
    _, target_seq_len, _ = base_model_util.get_shape_list(target)
    euler_preds = tfg.euler.from_rotation_matrix(
        tf.reshape(pred[:, :target_seq_len, 9:], [-1, 3, 3]))
    euler_targs = tfg.euler.from_rotation_matrix(
        tf.reshape(target[:, :, 9:], [-1, 3, 3]))

    # Euler conversion might have NANs, here replace nans with zeros.
    euler_preds = tf.where(
        tf.math.is_nan(euler_preds), tf.zeros_like(euler_preds), euler_preds)
    euler_targs = tf.where(
        tf.math.is_nan(euler_targs), tf.zeros_like(euler_targs), euler_targs)
    # reshape to (-1, n_joints*3) to be consistent with previous work
    euler_preds = tf.reshape(euler_preds, [-1, self.num_joints * 3])
    euler_targs = tf.reshape(euler_targs, [-1, self.num_joints * 3])

    euler_diff = tf.norm(euler_targs - euler_preds, axis=-1)
    self.euler_errors.assign_add(tf.reduce_mean(euler_diff))

  def result(self):
    return self.euler_errors


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.

    Code apapted from https://github.com/mseitzer/pytorch-fid

    Copyright 2018 Institute of Bioinformatics, JKU Linz
    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    mu and sigma are calculated through:
    ```
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    ```
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


class FrechetFeatDist(tf.keras.metrics.Metric):

  def __init__(self, mode="kinetic"):
    super(FrechetFeatDist, self).__init__(name='FrechetFeatDist')
    if mode == "kinetic":
      self.extract_func = extract_kinetic_features
    elif mode == "manual":
      self.extract_func = extract_manual_features
    else:
      raise ValueError("%s is not support!" % mode)
    self.dist = 0

  def update_state(self, target_list, pred_list):
    traget_feat_list = np.array([
        self.extract_func(target.numpy()) for target in target_list])
    pred_feat_list = np.array([
        self.extract_func(pred.numpy()) for pred in pred_list])
    self.dist = calculate_frechet_distance(
        mu1=np.mean(traget_feat_list, axis=0),
        sigma1=np.cov(traget_feat_list, rowvar=False),
        mu2=np.mean(pred_feat_list, axis=0),
        sigma2=np.cov(pred_feat_list, rowvar=False),
    )

  def result(self):
    return self.dist