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
from numpy.core.records import array
from scipy import linalg
from scipy.spatial.transform import Rotation as R
import torch

from mint.core import base_model_util
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg

# See https://github.com/google/aistplusplus_api/ for installation 
from aist_plusplus.features.kinetic import extract_kinetic_features
from aist_plusplus.features.manual import extract_manual_features
from smplx import SMPL


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


def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden


def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest
    

def recover_to_axis_angles(motion):
    batch_size, seq_len, _ = motion.shape
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


class FrechetFeatDist(tf.keras.metrics.Metric):

  def __init__(self, smpl_dir, mode="kinetic"):
    super(FrechetFeatDist, self).__init__(name='FrechetFeatDist')
    if mode == "kinetic":
      self.extract_func = extract_kinetic_features
    elif mode == "manual":
      self.extract_func = extract_manual_features
    else:
      raise ValueError("%s is not support!" % mode)
    self.smpl_model = SMPL(
        model_path=smpl_dir, gender='MALE', batch_size=1)
    self.traget_feat_list = []
    self.pred_feat_list = []

  def reset_states(self):
    self.traget_feat_list.clear()
    self.pred_feat_list.clear()

  def update_state(self, target, pred):
    def _warp_extract_func(tensor):
      smpl_poses, smpl_trans = recover_to_axis_angles(tensor.numpy())
      batch_size, seq_len, _24, _3 = smpl_poses.shape
      smpl_poses = smpl_poses.reshape(-1, 24, 3)
      smpl_trans = smpl_trans.reshape(-1, 3)
      keypoints3d = self.smpl_model.forward(
          global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
          body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
          transl=torch.from_numpy(smpl_trans).float(),
      ).joints.detach().numpy()[:, :24, :]
      keypoints3d = keypoints3d.reshape(batch_size, seq_len, 24, 3)
      return np.stack([self.extract_func(pts) for pts in keypoints3d])
    self.traget_feat_list.append(
        tf.py_function(func=_warp_extract_func, inp=[target], Tout=tf.float32))
    self.pred_feat_list.append(
        tf.py_function(func=_warp_extract_func, inp=[pred], Tout=tf.float32))
    
  def result(self):
    def _warp_distance_func(tensors1, tensors2):
      array1, array2 = tensors1.numpy(), tensors2.numpy()
      return calculate_frechet_distance(
          mu1=np.mean(array1, axis=0),
          sigma1=np.cov(array1, rowvar=False),
          mu2=np.mean(array2, axis=0),
          sigma2=np.cov(array2, rowvar=False),
      )
    tensors1 = tf.concat(self.traget_feat_list, axis=0)
    tensors2 = tf.concat(self.pred_feat_list, axis=0)
    frechet_dist = tf.py_function(
        func=_warp_distance_func, inp=[tensors1, tensors2], Tout=tf.float32)
    return frechet_dist
