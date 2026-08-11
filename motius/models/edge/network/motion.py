"""EDGE rotation/FK helpers without a runtime dependency on the source repo."""

from __future__ import annotations

import torch
from torch.nn import functional as F


EDGE_CONTACT_DIM = 4
EDGE_MOTION_DIM = 3 + 24 * 6
EDGE_REPR_DIM = EDGE_CONTACT_DIM + EDGE_MOTION_DIM

EDGE_SMPL24_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)
EDGE_SMPL24_OFFSETS = (
    (0.0, 0.0, 0.0),
    (0.05858135, -0.08228004, -0.01766408),
    (-0.06030973, -0.09051332, -0.01354254),
    (0.00443945, 0.12440352, -0.03838522),
    (0.04345142, -0.38646945, 0.008037),
    (-0.04325663, -0.38368791, -0.00484304),
    (0.00448844, 0.1379564, 0.02682033),
    (-0.01479032, -0.42687458, -0.037428),
    (0.01905555, -0.4200455, -0.03456167),
    (-0.00226458, 0.05603239, 0.00285505),
    (0.04105436, -0.06028581, 0.12204243),
    (-0.03483987, -0.06210566, 0.13032329),
    (-0.0133902, 0.21163553, -0.03346758),
    (0.07170245, 0.11399969, -0.01889817),
    (-0.08295366, 0.11247234, -0.02370739),
    (0.01011321, 0.08893734, 0.05040987),
    (0.12292141, 0.04520509, -0.019046),
    (-0.11322832, 0.04685326, -0.00847207),
    (0.2553319, -0.01564902, -0.02294649),
    (-0.26012748, -0.01436928, -0.03126873),
    (0.26570925, 0.01269811, -0.00737473),
    (-0.26910836, 0.00679372, -0.00602676),
    (0.08669055, -0.01063603, -0.01559429),
    (-0.0887537, -0.00865157, -0.01010708),
)


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Convert PyTorch3D row-major 6D rotations to matrices."""

    a1, a2 = value[..., :3], value[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Return the first two matrix rows, matching PyTorch3D."""

    return matrix[..., :2, :].clone().reshape(*matrix.shape[:-2], 6)


def edge_forward_kinematics(
    rotations: torch.Tensor,
    root_positions: torch.Tensor,
) -> torch.Tensor:
    """Apply the fixed 24-joint skeleton used by the released EDGE model."""

    if rotations.ndim != 5 or rotations.shape[-3:] != (24, 3, 3):
        raise ValueError(f"Expected rotations (B,T,24,3,3), got {tuple(rotations.shape)}")
    if root_positions.shape != rotations.shape[:2] + (3,):
        raise ValueError("root_positions must match the rotation batch and timeline")
    offsets = torch.as_tensor(
        EDGE_SMPL24_OFFSETS, dtype=rotations.dtype, device=rotations.device
    )
    positions = torch.zeros(
        rotations.shape[:3] + (3,), dtype=rotations.dtype, device=rotations.device
    )
    global_rotations = torch.zeros_like(rotations)
    positions[:, :, 0] = root_positions
    global_rotations[:, :, 0] = rotations[:, :, 0]
    for joint in range(1, 24):
        parent = EDGE_SMPL24_PARENTS[joint]
        global_rotations[:, :, joint] = torch.matmul(
            global_rotations[:, :, parent], rotations[:, :, joint]
        )
        positions[:, :, joint] = positions[:, :, parent] + torch.matmul(
            global_rotations[:, :, parent], offsets[joint, :, None]
        ).squeeze(-1)
    return positions


def edge_zup_to_aistpp_yup(points: torch.Tensor) -> torch.Tensor:
    """Undo EDGE's +90 degree X rotation: ``(x,y,z) -> (x,z,-y)``."""

    return torch.stack((points[..., 0], points[..., 2], -points[..., 1]), dim=-1)


def edge_motion_to_motion135(motion: torch.Tensor) -> torch.Tensor:
    """Convert EDGE-151/147 directly to Y-up SMPL ``motion135``.

    EDGE stores the exact SMPL local rotations used by its released skeleton,
    so this route is lossless and does not use position IK.  AIST++ was rotated
    +90 degrees around X before EDGE training; converting back therefore only
    left-multiplies the root rotation by the inverse basis transform.  The
    remaining local joint rotations are unchanged.
    """

    value = torch.as_tensor(motion)
    squeeze = value.ndim == 2
    if squeeze:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] not in (EDGE_REPR_DIM, EDGE_MOTION_DIM):
        raise ValueError(
            f"Expected EDGE motion (B,T,151) or (B,T,147), got {tuple(value.shape)}"
        )
    if value.shape[-1] == EDGE_REPR_DIM:
        value = value[..., EDGE_CONTACT_DIM:]

    root = edge_zup_to_aistpp_yup(value[..., :3])
    rotations = rotation_6d_to_matrix(
        value[..., 3:].reshape(*value.shape[:2], 24, 6)
    ).clone()
    inverse_basis = rotations.new_tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0))
    )
    rotations[..., 0, :, :] = torch.matmul(
        inverse_basis, rotations[..., 0, :, :]
    )
    # Motius motion135 historically stores the first two matrix columns with
    # row-wise flattening: [R00,R01,R10,R11,R20,R21]. EDGE's native PyTorch3D
    # rot6d stores the first two rows, so copying the six values is incorrect.
    rotation6d = rotations[..., :22, :, :2].reshape(*value.shape[:2], 22 * 6)
    result = torch.cat((root, rotation6d), dim=-1)
    return result[0] if squeeze else result


def edge_motion_to_aistpp_joints(motion: torch.Tensor) -> torch.Tensor:
    """Decode unnormalized EDGE-151/147 motion to AIST++ Y-up joints."""

    value = torch.as_tensor(motion)
    if value.ndim == 2:
        value = value.unsqueeze(0)
    if value.ndim != 3 or value.shape[-1] not in (EDGE_REPR_DIM, EDGE_MOTION_DIM):
        raise ValueError(
            f"Expected EDGE motion (B,T,151) or (B,T,147), got {tuple(value.shape)}"
        )
    if value.shape[-1] == EDGE_REPR_DIM:
        value = value[..., EDGE_CONTACT_DIM:]
    root = value[..., :3]
    rotations = rotation_6d_to_matrix(value[..., 3:].reshape(*value.shape[:2], 24, 6))
    return edge_zup_to_aistpp_yup(edge_forward_kinematics(rotations, root))


__all__ = [
    "EDGE_CONTACT_DIM",
    "EDGE_MOTION_DIM",
    "EDGE_REPR_DIM",
    "edge_forward_kinematics",
    "edge_motion_to_aistpp_joints",
    "edge_motion_to_motion135",
    "edge_zup_to_aistpp_yup",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
]
