"""Differentiable SMPL-H joint reconstruction."""
import torch
from motius.motion.representation.flowhmr.geometry import rot6d_to_rotation_matrix

def get_joints_from_smpl_params(smpl_skeleton, batch, joint_num=24):
    rot6d = batch["rot6d"]
    transl = batch["trans"]
    shapes = batch["shapes"]
    rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
    transl_flat = transl.reshape(rot6d.shape[0] * rot6d.shape[1], 3)
    assert shapes.shape[1] == 1, f"shapes shape should be [B,1,16], got {shapes.shape}"
    shapes = shapes.repeat(1, rot6d.shape[1], 1)
    shapes_flat = shapes.reshape(rot6d.shape[0] * rot6d.shape[1], 16)

    local_params = {
        "rot6d": rot6d_flat,
        "trans": torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype),
        "shapes": shapes_flat,
    }
    global_params = {
        "rot6d": rot6d_flat,
        "trans": transl_flat,
        "shapes": shapes_flat,
    }

    output_joints = {
        "local_joints": smpl_skeleton(local_params)["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
        "global_joints": smpl_skeleton(global_params)["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
    }
    for key in output_joints:
        output_joints[key] = output_joints[key][:, :, :joint_num, :]
    return output_joints
