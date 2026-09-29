from __future__ import annotations
import torch
from ..geometry import angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix, rotation_matrix_to_rot6d


def compute_cam_angvel(R_w2c, padding_last=True):
    """
    R_w2c : (F, 3, 3)
    """
    # R @ R0 = R1, so R = R1 @ R0^T
    cam_angvel = rotation_matrix_to_rot6d(R_w2c[1:] @ R_w2c[:-1].transpose(-1, -2))  # (F-1, 6)
    # cam_angvel = (cam_angvel - torch.tensor([[1, 0, 0, 0, 1, 0]])) * FPS
    assert padding_last
    cam_angvel = torch.cat([cam_angvel, cam_angvel[-1:]], dim=0)  # (F, 6)
    return cam_angvel


def get_R_c2gv(R_w2c, axis_gravity_in_w=[0, 0, -1]):
    """
    Args:
        R_w2c: (*, 3, 3)
    Returns:
        R_c2gv: (*, 3, 3)
    """
    if isinstance(axis_gravity_in_w, list):
        axis_gravity_in_w = torch.FloatTensor(axis_gravity_in_w)  # gravity direction in world coord
    axis_z_in_c = torch.FloatTensor([0, 0, 1])

    # get gv-coord axes in in c-coord
    axis_y_of_gv = R_w2c @ axis_gravity_in_w  # (*, 3)
    axis_x_of_gv = axis_y_of_gv.cross(axis_z_in_c.expand_as(axis_y_of_gv), dim=-1)
    # normalize
    axis_x_of_gv_norm = axis_x_of_gv.norm(dim=-1, keepdim=True)
    axis_x_of_gv = axis_x_of_gv / (axis_x_of_gv_norm + 1e-5)
    axis_x_of_gv[axis_x_of_gv_norm.squeeze(-1) < 1e-5] = torch.FloatTensor(
        [1.0, 0.0, 0.0]
    )  # use cam x-axis as axis_x_of_gv
    axis_z_of_gv = axis_x_of_gv.cross(axis_y_of_gv, dim=-1)

    R_gv2c = torch.FloatTensor(torch.stack([axis_x_of_gv, axis_y_of_gv, axis_z_of_gv], dim=-1))  # (*, 3, 3)
    R_c2gv = R_gv2c.transpose(-1, -2)  # (*, 3, 3)
    return R_c2gv


def process_r_t(R_transform, root_rotation, transl, j_shaped):
    root_rotation_new = R_transform[None] @ root_rotation
    if True:
        transl_new = (R_transform[None] @ (j_shaped[..., None] + torch.FloatTensor(transl[..., None]))).reshape(-1, 3) - j_shaped
    else:
        transl_new = (R_transform[None] @ torch.FloatTensor(transl[..., None])).reshape(-1, 3)

    return root_rotation_new, transl_new