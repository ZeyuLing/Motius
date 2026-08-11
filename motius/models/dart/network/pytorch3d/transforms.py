"""Subset of :mod:`pytorch3d.transforms` used by DART inference.

The A100 runtime used for large evaluations does not always have PyTorch3D
installed. DART only needs rotation conversion helpers for inference, so this
shim mirrors the PyTorch3D 6D convention locally inside the vendored DART
runtime.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from motius.models.hymotion_t2m.network.geometry import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """PyTorch3D 6D -> matrix convention.

    PyTorch3D stores the first two rows of the rotation matrix:
    ``matrix[..., :2, :].reshape(..., 6)``.
    """

    if d6.shape[-1] != 6:
        raise ValueError(f"rotation_6d_to_matrix expects (..., 6), got {tuple(d6.shape)}")
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Matrix -> PyTorch3D 6D convention."""

    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"matrix_to_rotation_6d expects (..., 3, 3), got {tuple(matrix.shape)}")
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"Invalid axis {axis!r}.")
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    if euler_angles.shape[-1] != 3:
        raise ValueError(f"Invalid input shape {euler_angles.shape}.")
    if len(convention) != 3:
        raise ValueError(f"Convention must have 3 letters, got {convention!r}.")
    matrices = [
        _axis_angle_rotation(axis, angle)
        for axis, angle in zip(convention, torch.unbind(euler_angles, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


__all__ = [
    "axis_angle_to_matrix",
    "matrix_to_axis_angle",
    "rotation_6d_to_matrix",
    "matrix_to_rotation_6d",
    "euler_angles_to_matrix",
]
