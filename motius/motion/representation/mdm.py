"""MDM HumanML target-location representation, separate from neural models."""

import torch
from motius.motion.representation.humanml import recover_from_ric, recover_root_rot_pos
from motius.motion.representation.humanml import _qrot as qrot

HML_JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]


def recover_root_rot_heading_ang(joints):
    """Get Forward Direction"""
    face_joint_idx = [2, 1, 17, 16]
    # l_hip, r_hip, sdr_r, sdr_l = face_joint_idx
    r_hip, l_hip, sdr_r, sdr_l = face_joint_idx  # Note the bugfix
    across1 = joints[:, r_hip] - joints[:, l_hip]
    across2 = joints[:, sdr_r] - joints[:, sdr_l]
    across = across1 + across2
    across = torch.nn.functional.normalize(across, dim=1)
    # print(across1.shape, across2.shape)

    # forward (batch_size, 3)
    forward = torch.cross(
        torch.tensor([[[0], [1], [0]]], dtype=across.dtype, device=across.device),
        across,
        axis=1,
    )
    forward = torch.nn.functional.normalize(forward, dim=1)

    return torch.atan2(forward[:, 0], forward[:, 2])[:, None]


def get_target_location(
    motion,
    mean,
    std,
    lengths,
    joints_num,
    all_goal_joint_names,
    target_joint_names,
    is_heading,
):
    assert (lengths == lengths[0]).all(), "currently supporting only fixed length"
    batch_size = motion.shape[0]
    extended_goal_joint_names = all_goal_joint_names + [
        "traj",
        "heading",
    ]  # todo: fix hardcoded indexing that assumes traj and heading are last

    # output tensor
    target_loc = torch.zeros(
        (batch_size, len(extended_goal_joint_names), 3, lengths[0]),
        dtype=motion.dtype,
        device=motion.device,
    )  #  n_samples x (n_target_joints+1) x 3 x n_frames

    # hml to abs loc (all joints, not only the requested ones)
    joints_loc = hml_to_abs_loc(motion, mean, std, joints_num)
    pelvis_loc = HML_JOINT_NAMES.index("pelvis")
    joints_loc = torch.concat(
        [joints_loc, joints_loc[:, pelvis_loc : pelvis_loc + 1]], dim=1
    )  # concatenate the pelvis location to be used for traj

    # joint names to indices
    HML_JOINT_NAMES_w_traj = HML_JOINT_NAMES + ["traj"]
    for sample_idx in range(batch_size):
        req_joint_idx_in = [
            HML_JOINT_NAMES_w_traj.index(name)
            for name in target_joint_names[sample_idx]
        ]
        req_joint_idx_out = [
            extended_goal_joint_names.index(name)
            for name in target_joint_names[sample_idx]
        ]

        target_loc[sample_idx, req_joint_idx_out] = joints_loc[
            sample_idx, req_joint_idx_in
        ]  # assign joints loc to output tensor

    target_loc[:, -2, 1] = 0  # zero the y axis for the trajectory

    # last entry is the heading
    heading = recover_root_rot_heading_ang(joints_loc)
    target_loc[:, -1:, 0][is_heading] = heading[is_heading]

    return target_loc[..., -1]


def hml_to_abs_loc(motion, mean, std, joints_num):
    # hml to abs loc (all joints, not only the requested ones)
    unnormed_motion = (motion * std + mean).permute(0, 2, 3, 1).float()
    joints_loc = recover_from_ric(unnormed_motion, joints_num)
    joints_loc = joints_loc.view(-1, *joints_loc.shape[2:]).permute(
        0, 2, 3, 1
    )  # n_samples x n_joints x 3 x n_frames
    return joints_loc
