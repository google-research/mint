import os
import numpy as np
import argparse

from aist_plusplus.loader import AISTDataset
from aist_plusplus.features.kinetic import extract_kinetic_features
from aist_plusplus.features.manual import extract_manual_features
from smplx import SMPL

import torch
import multiprocessing
import functools


parser = argparse.ArgumentParser(
    description='')
parser.add_argument(
    '--anno_dir',
    type=str,
    default='/mnt/data/aist_plusplus_final/',
    help='input local dictionary for AIST++ annotations.')
parser.add_argument(
    '--smpl_dir',
    type=str,
    default='/mnt/data/smpl/',
    help='input local dictionary that stores SMPL data.')
parser.add_argument(
    '--save_dir',
    type=str,
    default='./data/aist_features/',
    help='output local dictionary that stores features.')
FLAGS = parser.parse_args()
    

def main(seq_name, motion_dir):
    # Parsing SMPL 24 joints.
    # Note here we calculate `transl` as `smpl_trans/smpl_scaling` for 
    # normalizing the motion in generic SMPL model scale.    
    smpl = SMPL(model_path=FLAGS.smpl_dir, gender='MALE', batch_size=1)

    print (seq_name)
    smpl_poses, smpl_scaling, smpl_trans = AISTDataset.load_motion(
        motion_dir, seq_name)
    keypoints3d = smpl.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans / smpl_scaling).float(),
        ).joints.detach().numpy()[:, 0:24, :]
    features = extract_kinetic_features(keypoints3d)
    np.save(os.path.join(FLAGS.save_dir, seq_name+"_kinetic.npy"), features)
    features = extract_manual_features(keypoints3d)
    np.save(os.path.join(FLAGS.save_dir, seq_name+"_manual.npy"), features)
    print (seq_name, "is done")


if __name__ == '__main__':
    os.makedirs(FLAGS.save_dir, exist_ok=True)
    
    # Parsing data info.
    aist_dataset = AISTDataset(FLAGS.anno_dir)
    seq_names = aist_dataset.mapping_seq2env.keys()
    ignore_list = np.loadtxt(
        os.path.join(FLAGS.anno_dir, "ignore_list.txt"), dtype=str).tolist()
    seq_names = [n for n in seq_names if n not in ignore_list]

    # processing
    process = functools.partial(main, motion_dir=aist_dataset.motion_dir)
    pool = multiprocessing.Pool(12)
    pool.map(process, seq_names)