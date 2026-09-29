from __future__ import annotations
# 这个脚本用来进行数据增强

import numpy as np
import copy
import torch
from ..datasets.geometry import angle_axis_to_rotation_matrix, rotation_matrix_to_angle_axis

# Permutation of SMPL pose parameters when flipping the shape
_PERMUTATION = {
    "smpl": [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 23, 22],
    "smplh": [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24],
    "smplx": [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20, 24, 25, 23, 24, 26, 28, 27],
    "smplhfull": [
        0,
        2,
        1,
        3,
        5,
        4,
        6,
        8,
        7,
        9,
        11,
        10,
        12,
        14,
        13,
        15,
        17,
        16,
        19,
        18,
        21,
        20,  # body
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
    ],
    "smplxfull": [
        0,
        2,
        1,
        3,
        5,
        4,
        6,
        8,
        7,
        9,
        11,
        10,
        12,
        14,
        13,
        15,
        17,
        16,
        19,
        18,
        21,
        20,  # body
        22,
        24,
        23,  # jaw, left eye, right eye
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,  # right hand
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,  # left hand
    ],
}

PERMUTATION = {}
for key in _PERMUTATION.keys():
    res = []
    for i in _PERMUTATION[key]:
        res.extend([3 * i + j for j in range(3)])
    PERMUTATION[max(res) + 1] = res


def calc_mirror_transform(m):
    coeff_mat = np.eye(4)[None, :, :]
    coeff_mat = coeff_mat.repeat(m.shape[0], 0)
    norm = np.linalg.norm(m[:, :3], keepdims=True, axis=1)
    m[:, :3] /= norm
    coeff_mat[:, 0, 0] = 1 - 2 * m[:, 0] ** 2
    coeff_mat[:, 0, 1] = -2 * m[:, 0] * m[:, 1]
    coeff_mat[:, 0, 2] = -2 * m[:, 0] * m[:, 2]
    coeff_mat[:, 0, 3] = -2 * m[:, 0] * m[:, 3]
    coeff_mat[:, 1, 0] = -2 * m[:, 1] * m[:, 0]
    coeff_mat[:, 1, 1] = 1 - 2 * m[:, 1] ** 2
    coeff_mat[:, 1, 2] = -2 * m[:, 1] * m[:, 2]
    coeff_mat[:, 1, 3] = -2 * m[:, 1] * m[:, 3]
    coeff_mat[:, 2, 0] = -2 * m[:, 2] * m[:, 0]
    coeff_mat[:, 2, 1] = -2 * m[:, 2] * m[:, 1]
    coeff_mat[:, 2, 2] = 1 - 2 * m[:, 2] ** 2
    coeff_mat[:, 2, 3] = -2 * m[:, 2] * m[:, 3]
    return coeff_mat


def swap_left_right(data_dict):
    poses = data_dict["poses"].copy()
    poses = poses[:, PERMUTATION[poses.shape[-1]]]
    if poses.shape[1] in [72, 156, 165]:
        poses[:, 1::3] = -poses[:, 1::3]
        poses[:, 2::3] = -poses[:, 2::3]
    elif poses.shape[1] in [78, 87]:
        poses[:, 1:66:3] = -poses[:, 1:66:3]
        poses[:, 2:66:3] = -poses[:, 2:66:3]
    else:
        import ipdb

        ipdb.set_trace()
    # we also negate the second and the third dimension of the axis-angle
    data_dict_copy = copy.deepcopy(data_dict)
    data_dict_copy["poses"] = poses.reshape(poses.shape[0], -1).copy()
    #
    mirror = np.array([[1.0, 0.0, 0.0, 0.0]])
    M = calc_mirror_transform(mirror)
    Tnew = np.einsum("bmn,bn->bm", M[:, :3, :3], data_dict_copy["trans"]) + M[:, :3, 3]
    data_dict_copy["trans"] = Tnew.copy()
    return data_dict_copy


def process_r_t(R_transform, global_orient, transl, j_shaped):
    global_orient_rot = angle_axis_to_rotation_matrix(torch.FloatTensor(global_orient))
    global_orient_rot_new = R_transform[None] @ global_orient_rot
    if True:
        transl_new = (R_transform[None] @ (j_shaped[..., None] + torch.FloatTensor(transl[..., None]))).reshape(
            -1, 3
        ) - j_shaped
    else:
        transl_new = (R_transform[None] @ torch.FloatTensor(transl[..., None])).reshape(-1, 3)
    global_orient_new = rotation_matrix_to_angle_axis(global_orient_rot_new)

    return global_orient_new.cpu().numpy(), transl_new.cpu().numpy()


def rotation_around_y_axis(data_dict, rotation_angle, j_shaped):
    #
    axis_y = torch.FloatTensor([0, 1, 0]) * rotation_angle
    R_transform = angle_axis_to_rotation_matrix(axis_y.reshape(1, -1))[0]
    global_orient, transl = process_r_t(
        R_transform, data_dict["poses"][:, :3], data_dict["trans"], j_shaped[:, 0].cpu()
    )
    data_dict["poses"][:, :3] = global_orient
    data_dict["trans"] = transl
    return data_dict
