from absl import app
from absl import flags
from absl import logging

import os
from librosa import beat
import torch
import numpy as np
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.signal as scisignal
from aist_plusplus.loader import AISTDataset


FLAGS = flags.FLAGS
flags.DEFINE_string(
    'anno_dir', '/mnt/data/aist_plusplus_final/',
    'Path to the AIST++ annotation files.')
flags.DEFINE_string(
    'audio_dir', '/mnt/data/AIST/music/',
    'Path to the AIST wav files.')
flags.DEFINE_string(
    'audio_cache_dir', './data/aist_audio_feats/',
    'Path to cache dictionary for audio features.')
flags.DEFINE_enum(
    'split', 'testval', ['train', 'testval'],
    'Whether do training set or testval set.')
flags.DEFINE_string(
    'result_files', '/mnt/data/aist_paper_results/*.pkl',
    'The path pattern of the result files.')
flags.DEFINE_bool(
    'legacy', True,
    'Whether the result files are the legacy version.')


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
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl


def recover_motion_to_keypoints(motion, smpl_model):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
    return keypoints3d


def motion_peak_onehot(joints):
    """Calculate motion beats.
    Kwargs:
        joints: [nframes, njoints, 3]
    Returns:
        - peak_onhot: motion beats.
    """
    # Calculate velocity.
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    velocity_norms = np.linalg.norm(velocity, axis=2)
    envelope = np.sum(velocity_norms, axis=1)  # (seq_len,)

    # Find local minima in velocity -- beats
    peak_idxs = scisignal.argrelextrema(envelope, np.less, axis=0, order=10) # 10 for 60FPS
    peak_onehot = np.zeros_like(envelope, dtype=bool)
    peak_onehot[peak_idxs] = 1

    # # Second-derivative of the velocity shows the energy of the beats
    # peak_energy = np.gradient(np.gradient(envelope)) # (seq_len,)
    # # optimize peaks
    # peak_onehot[peak_energy<0.001] = 0
    return peak_onehot


def alignment_score(music_beats, motion_beats, sigma=3):
    """Calculate alignment score between music and motion."""
    if motion_beats.sum() == 0:
        return 0.0
    music_beat_idxs = np.where(music_beats)[0]
    motion_beat_idxs = np.where(motion_beats)[0]
    score_all = []
    for motion_beat_idx in motion_beat_idxs:
        dists = np.abs(music_beat_idxs - motion_beat_idx).astype(np.float32)
        ind = np.argmin(dists)
        score = np.exp(- dists[ind]**2 / 2 / sigma**2)
        score_all.append(score)
    return sum(score_all) / len(score_all)


def main(_):
    import glob
    import tqdm
    from smplx import SMPL

    # set smpl
    smpl = SMPL(model_path="/mnt/data/smpl/", gender='MALE', batch_size=1)

    # create list
    seq_names = []
    if "train" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_train.txt"), dtype=str
        ).tolist()
    if "val" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_val.txt"), dtype=str
        ).tolist()
    if "test" in FLAGS.split:
        seq_names += np.loadtxt(
            os.path.join(FLAGS.anno_dir, "splits/crossmodal_test.txt"), dtype=str
        ).tolist()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str
    ).tolist()
    seq_names = [name for name in seq_names if name not in ignore_list]

    # calculate score on real data
    dataset = AISTDataset(FLAGS.anno_dir)
    n_samples = len(seq_names)
    beat_scores = []
    for i, seq_name in enumerate(seq_names):
        logging.info("processing %d / %d" % (i + 1, n_samples))
        # get real data motion beats
        smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
            dataset.motion_dir, seq_name)
        smpl_trans /= smpl_scaling
        keypoints3d = smpl.forward(
            global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
            body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
            transl=torch.from_numpy(smpl_trans).float(),
        ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)
        motion_beats = motion_peak_onehot(keypoints3d)
        # get real data music beats
        audio_name = seq_name.split("_")[4]
        audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy"))
        audio_beats = audio_feature[:keypoints3d.shape[0], -1] # last dim is the music beats
        # get beat alignment scores
        beat_score = alignment_score(audio_beats, motion_beats, sigma=3)
        beat_scores.append(beat_score)
    print ("\nBeat score on real data: %.3f\n" % (sum(beat_scores) / n_samples))

    # calculate score on generated motion data
    result_files = sorted(glob.glob(FLAGS.result_files))
    result_files = [f for f in result_files if f[-8:-4] in f[:-8]]
    if FLAGS.legacy:
        # for some reason there are repetitive results. Skip them
        result_files = {f[-34:]: f for f in result_files}
        result_files = result_files.values()
    n_samples = len(result_files)
    beat_scores = []
    for result_file in tqdm.tqdm(result_files):
        if FLAGS.legacy:
            with open(result_file, "rb") as f:
                data = pickle.load(f)
            result_motion = np.concatenate([
                np.pad(data["pred_trans"], ((0, 0), (0, 0), (6, 0))),
                data["pred_motion"].reshape(1, -1, 24 * 9)
            ], axis=-1)  # [1, 120 + 1200, 225]
        else:
            result_motion = np.load(result_file)[None, ...]  # [1, 120 + 1200, 225]
        keypoints3d = recover_motion_to_keypoints(result_motion, smpl)
        motion_beats = motion_peak_onehot(keypoints3d)
        if FLAGS.legacy:
            audio_beats = data["audio_beats"][0] > 0.5
        else:
            audio_name = result_file[-8:-4]
            audio_feature = np.load(os.path.join(FLAGS.audio_cache_dir, f"{audio_name}.npy"))
            audio_beats = audio_feature[:, -1] # last dim is the music beats
        beat_score = alignment_score(audio_beats[120:], motion_beats[120:], sigma=3)
        beat_scores.append(beat_score)
    print ("\nBeat score on generated data: %.3f\n" % (sum(beat_scores) / n_samples))


if __name__ == '__main__':
    app.run(main)
