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
"""BVH file writer and utils.

BVH is a motion capture file format. Details refer to:
http://www.dcs.shef.ac.uk/intranet/research/public/resmes/CS0111.pdf
It stores the skeletal animation in a hierarchical manner.

Writer needs two inputs: the skeletal definition and motion data.

The motion data is provided in a pkl file. The pkl file contain a dict:
{"pred_results":["model_name_pose": joints angle array,
                 "joints_3d":joints pose array]}.

"""
import csv
import pickle

from absl import logging
import mako.template
from mint.tools import conversion_util
import numpy as np

_SUPPORTED_MODELS = ["smpl"]


class BVHData(object):
  """Struct of a BVH data.

  A container for properties: ${global_trans} the 3d translation of the joint,
  ${axis_angles} the joint rotation represented in axis angles,
  ${euler_angles} the joint rotation represented in euler angles.
  """

  def __init__(self, global_trans, axis_angles):
    """Initialize with joints location and axis angle."""
    self.global_trans = global_trans
    self.axis_angles = axis_angles.reshape([-1, 3])
    self.euler_angles = np.ones_like(self.axis_angles)
    for index, axis_angle in enumerate(self.axis_angles):
      self.euler_angles[index] = conversion_util.aa2rotmat(axis_angle)


class BVHWriter(object):
  """BVH file writer."""

  def __init__(self,
               model_name,
               skeleton_csv_filename,
               motion_pkl_filename,
               joints_to_ignore_csv_filename=None):
    """Initialize the bvh writer with given properties.

    Args:
      model_name: the name of the human shape/pose model.
      skeleton_csv_filename: the skeleton definition file.
      motion_pkl_filename: the pkl file with the motion data.
      joints_to_ignore_csv_filename: the csv file of the joints to ignore.
    """
    self.header_content = ""
    self.motion_content = ""
    if model_name not in _SUPPORTED_MODELS:
      raise ValueError("Model: %s not supported yet." % model_name)
    self.model_name = model_name
    (self.joint_names, self.joint_indices, self.joint_parent_indices,
     rest_pose) = self._read_skeleton_csv_file(skeleton_csv_filename)
    self.joint_names_to_ignore, _, _, _ = self._read_skeleton_csv_file(
        joints_to_ignore_csv_filename)

    self.rest_offsets = self._compute_joint_offsets(np.array(rest_pose))
    self.motion_data = self._read_motion_from_pkl_file(motion_pkl_filename)

  def _compute_joint_offsets(self, joints_3d):
    """Compute the rest pose offsets.

    Offsets is computed as the joint_position_3d - parent_joint_position_3d.

    Args:
      joints_3d: the joints position in 3d read from the skeleton definition.

    Returns:
      offsets: a numpy array with shape [num_joints * 3,] of the offsets.
    """
    # Here constructs the ${joint_index_to_pos} array.
    # The joint_index doesn't necessary correspond to the index of joint in
    # ${joint_names} array
    joint_index_to_pos = {}
    for index, joint_id in enumerate(self.joint_indices):
      joint_index_to_pos[joint_id] = joints_3d[index]

    offsets = []
    for index in range(len(self.joint_names)):
      if index == 0:
        offsets.append(joints_3d[index])
      else:
        parent_pos = joint_index_to_pos[self.joint_parent_indices[index]]
        offsets.append(joints_3d[index] - parent_pos)
    offsets = np.stack(offsets).ravel()
    return offsets

  def _read_skeleton_csv_file(self, csv_filename):
    """Read the skeleton definition from a csv file."""
    joint_names = []
    joint_indices = []
    joint_parent_indices = []
    joints_3d = []
    if csv_filename is not None:
      try:
        with open(csv_filename, "rt") as csv_file:
          csv_file_reader = csv.reader(
              csv_file, skipinitialspace=True, delimiter=",")
          for row in csv_file_reader:
            joint_names.append(row[0])
            if len(row) == 6:
              joint_indices.append(int(row[1]))
              joint_parent_indices.append(int(row[2]))
              joints_3d.append([float(row[3]), float(row[4]), float(row[5])])
      except EOFError as e:
        message = "Skipping reading file %s due to: %s" % (csv_filename, str(e))
        logging.warning(message)
    return joint_names, joint_indices, joint_parent_indices, joints_3d

  def fill_header(self, frame_rate, template_file):
    """Fill the bvh file header."""
    template_text = open(template_file).read()
    template = mako.template.Template(
        template_text, output_encoding="utf-8", encoding_errors="replace")
    self.header_content = template.render_unicode(
        offsets=self.rest_offsets,
        num_frames=len(self.motion_data),
        frame_rate=frame_rate)

  def _read_motion_from_pkl_file(self, pkl_filename):
    """Read motion from a pkl file."""
    try:
      with open(pkl_filename, "rb") as f:
        data = pickle.load(f)
    except EOFError as e:
      message = "Aboring reading file %s due to: %s" % (pkl_filename, str(e))
      logging.warning(message)
      raise ValueError(message)

    if "smpl_poses" in data:
      axis_angles = np.reshape(data["smpl_poses"], [-1, 24, 3])
      if data["smpl_trans"] is None:
        trans = np.zeros((axis_angles.shape[0], 3), dtype=np.float32)
      else:
        trans = np.reshape(data["smpl_trans"], [-1, 3])
    else:
      rotmats = np.reshape(data["pred_motion"], [-1, 24, 3, 3])
      rotmats = conversion_util.get_closest_rotmat(rotmats)
      axis_angles = conversion_util.rotmat2aa(rotmats)
      trans = np.reshape(data["pred_trans"], [-1, 3])

    motion_data = []
    for axis_angle_frame, trans_frame in zip(axis_angles, trans):
      motion_data.append(BVHData(trans_frame, axis_angle_frame))
    return motion_data

  def fill_motion(self, root_index=0, order="zyx"):
    """Fill the bvh motion content."""
    joint_indices_to_write = []
    for index in range(len(self.joint_indices)):
      if (self.joint_names[index]
          not in self.joint_names_to_ignore) and (index != root_index):
        joint_indices_to_write.append(self.joint_indices[index])

    motion_strs = []
    for motion in self.motion_data:
      euler_angles = motion.euler_angles
      root_trans = motion.global_trans
      root_rot = np.rad2deg(euler_angles[root_index])
      if order == "zyx":
        root_rot = root_rot[::-1]
      relative_rot = np.rad2deg(euler_angles[joint_indices_to_write])
      if order == "zyx":
        relative_rot = relative_rot[:, ::-1]
      bvh_motion = np.vstack((root_trans, root_rot, relative_rot)).ravel()
      motion_strs.append(" ".join(["%.5f" % m for m in bvh_motion]))
    self.motion_content = "\n".join(motion_strs)

  def write_to_bvh(self, out_bvh_filename):
    if self.header_content and self.motion_content:
      with open(out_bvh_filename, "w") as f:
        f.write(self.header_content)
        f.write(self.motion_content)
    else:
      message = ("Need to fill both the header and the motion content before "
                 "write to the bvh file.")
      logging.warning(message)
      raise ValueError(message)
