# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"Rotation conversion for the SOMA inference wrapper."

from __future__ import annotations

import torch


def cont6d_to_matrix(cont6d):
    """Convert 6D rotations to rotation matrices."""
    assert cont6d.shape[-1] == 6, "The last dimension must be 6"
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]
    x = x_raw / torch.norm(x_raw, dim=-1, keepdim=True)
    z = torch.cross(x, y_raw, dim=-1)
    z = z / torch.norm(z, dim=-1, keepdim=True)
    y = torch.cross(z, x, dim=-1)
    x = x[..., None]
    y = y[..., None]
    z = z[..., None]
    mat = torch.cat([x, y, z], dim=-1)
    return mat


def matrix_to_cont6d(matrix):
    """Convert rotation matrices to 6D rotations, their first two columns."""
    cont_6d = torch.concat([matrix[..., 0], matrix[..., 1]], dim=-1)
    return cont_6d


def change_t_pose_global_mats_onnx_compatible(
    global_mats: torch.Tensor,
    global_offset_from: torch.Tensor,
    global_offset_to: torch.Tensor,
):
    """Move global joint rotations from one T pose to another.

    Each result is global_mats @ offset_from @ offset_to^T.
    """
    dtype = global_mats.dtype
    device = global_mats.device
    new_global_mats = torch.matmul(
        torch.matmul(
            global_mats, global_offset_from[None, None].to(device=device, dtype=dtype)
        ),
        global_offset_to.transpose(-1, -2)[None, None].to(device=device, dtype=dtype),
    )
    return new_global_mats
