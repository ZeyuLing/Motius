"""Forward / inverse kinematics utilities for local <-> global rotation conversion.

Provides numpy and torch versions for dataset transforms and inference decode
respectively.

Rotation convention notes:
- Training data uses **row-major** rot6d: [R00, R01, R10, R11, R20, R21].
- ``rotation_convert.py`` uses **column-major** rot6d: [R00, R10, R20, R01, R11, R21].
- ``geometry.py`` (M2M network) uses **row-major** rot6d natively.

Numpy path (dataset):
    row-major -> col-major [0,2,4,1,3,5] -> rotation_6d_to_matrix -> 3x3 -> FK/IFK
    -> matrix_to_rotation_6d -> col-major -> row-major [0,3,1,4,2,5]

Torch path (inference decode):
    row-major -> geometry.rot6d_to_rotation_matrix (row-major native) -> 3x3 -> FK/IFK
    -> geometry.rotation_matrix_to_rot6d -> row-major
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
from torch import Tensor

# SMPL-22 kinematic tree: parent index for each joint (-1 = root)
SMPL22_PARENTS: List[int] = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip
    0,   # 2: R_Hip
    0,   # 3: Spine1
    1,   # 4: L_Knee
    2,   # 5: R_Knee
    3,   # 6: Spine2
    4,   # 7: L_Ankle
    5,   # 8: R_Ankle
    6,   # 9: Spine3
    7,   # 10: L_Foot
    8,   # 11: R_Foot
    9,   # 12: Neck
    9,   # 13: L_Collar
    9,   # 14: R_Collar
    12,  # 15: Head
    13,  # 16: L_Shoulder
    14,  # 17: R_Shoulder
    16,  # 18: L_Elbow
    17,  # 19: R_Elbow
    18,  # 20: L_Wrist
    19,  # 21: R_Wrist
]

NUM_JOINTS = 22

# =====================================================================
# Numpy versions (for dataset transforms)
# =====================================================================

# Reorder indices: row-major <-> column-major for 6D rotation
_ROW_TO_COL = [0, 2, 4, 1, 3, 5]  # row-major -> column-major
_COL_TO_ROW = [0, 3, 1, 4, 2, 5]  # column-major -> row-major


def _rot6d_row_to_matrix_np(rot6d_row: np.ndarray) -> np.ndarray:
    """Convert row-major rot6d (..., 6) to rotation matrix (..., 3, 3) via column-major path."""
    from motius.motion.representation.rotation import (
        rotation_6d_to_matrix,
    )
    rot6d_col = rot6d_row[..., _ROW_TO_COL]
    return rotation_6d_to_matrix(rot6d_col)  # (..., 3, 3)


def _matrix_to_rot6d_row_np(mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix (..., 3, 3) to row-major rot6d (..., 6)."""
    from motius.motion.representation.rotation import (
        matrix_to_rotation_6d,
    )
    rot6d_col = matrix_to_rotation_6d(mat)  # (..., 6) column-major
    return rot6d_col[..., _COL_TO_ROW]


def local_to_global_rot6d(local_rot6d_rowmajor: np.ndarray) -> np.ndarray:
    """Convert local rotation 6D (row-major) to global rotation 6D (row-major).

    Args:
        local_rot6d_rowmajor: (T, 22, 6) row-major local rotations.

    Returns:
        (T, 22, 6) row-major global rotations.
    """
    assert local_rot6d_rowmajor.shape[-2:] == (NUM_JOINTS, 6), (
        f"Expected (..., 22, 6), got {local_rot6d_rowmajor.shape}"
    )
    # Convert to rotation matrices
    local_mat = _rot6d_row_to_matrix_np(local_rot6d_rowmajor)  # (..., 22, 3, 3)

    # Forward kinematics: accumulate rotations along kinematic chain
    leading = local_mat.shape[:-3]  # e.g. (T,)
    global_mat = np.zeros_like(local_mat)
    for j, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            global_mat[..., j, :, :] = local_mat[..., j, :, :]
        else:
            global_mat[..., j, :, :] = global_mat[..., parent, :, :] @ local_mat[..., j, :, :]

    # Convert back to row-major rot6d
    return _matrix_to_rot6d_row_np(global_mat)  # (..., 22, 6)


def global_to_local_rot6d(global_rot6d_rowmajor: np.ndarray) -> np.ndarray:
    """Convert global rotation 6D (row-major) to local rotation 6D (row-major).

    Args:
        global_rot6d_rowmajor: (T, 22, 6) row-major global rotations.

    Returns:
        (T, 22, 6) row-major local rotations.
    """
    assert global_rot6d_rowmajor.shape[-2:] == (NUM_JOINTS, 6), (
        f"Expected (..., 22, 6), got {global_rot6d_rowmajor.shape}"
    )
    # Convert to rotation matrices
    global_mat = _rot6d_row_to_matrix_np(global_rot6d_rowmajor)  # (..., 22, 3, 3)

    # Inverse FK: local[j] = inv(global[parent]) @ global[j]
    local_mat = np.zeros_like(global_mat)
    for j, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            local_mat[..., j, :, :] = global_mat[..., j, :, :]
        else:
            # R_local = R_parent^T @ R_global (for pure rotations, inv = transpose)
            parent_inv = np.swapaxes(global_mat[..., parent, :, :], -2, -1)
            local_mat[..., j, :, :] = parent_inv @ global_mat[..., j, :, :]

    return _matrix_to_rot6d_row_np(local_mat)  # (..., 22, 6)


# =====================================================================
# Torch versions (for inference decode)
# =====================================================================


def global_to_local_rot6d_torch(global_rot6d: Tensor) -> Tensor:
    """Convert global rotation 6D (row-major) to local rotation 6D (row-major).

    Uses geometry.py functions which are natively row-major.

    Args:
        global_rot6d: (*, 22, 6) row-major global rotations.

    Returns:
        (*, 22, 6) row-major local rotations.
    """
    from motius.models.motioncanvas.network.geometry import (
        rot6d_to_rotation_matrix,
        rotation_matrix_to_rot6d,
    )

    assert global_rot6d.shape[-2:] == (NUM_JOINTS, 6), (
        f"Expected (..., 22, 6), got {global_rot6d.shape}"
    )

    # Convert to rotation matrices: (..., 22, 3, 3)
    global_mat = rot6d_to_rotation_matrix(global_rot6d)

    # Inverse FK: local[j] = parent_global^T @ global[j]
    local_mat = torch.zeros_like(global_mat)
    for j, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            local_mat[..., j, :, :] = global_mat[..., j, :, :]
        else:
            parent_inv = global_mat[..., parent, :, :].transpose(-2, -1)
            local_mat[..., j, :, :] = parent_inv @ global_mat[..., j, :, :]

    return rotation_matrix_to_rot6d(local_mat)  # (..., 22, 6)


def local_to_global_rot6d_torch(local_rot6d: Tensor) -> Tensor:
    """Convert local rotation 6D (row-major) to global rotation 6D (row-major).

    Uses geometry.py functions which are natively row-major.

    Args:
        local_rot6d: (*, 22, 6) row-major local rotations.

    Returns:
        (*, 22, 6) row-major global rotations.
    """
    from motius.models.motioncanvas.network.geometry import (
        rot6d_to_rotation_matrix,
        rotation_matrix_to_rot6d,
    )

    assert local_rot6d.shape[-2:] == (NUM_JOINTS, 6), (
        f"Expected (..., 22, 6), got {local_rot6d.shape}"
    )

    local_mat = rot6d_to_rotation_matrix(local_rot6d)  # (..., 22, 3, 3)

    global_mat = torch.zeros_like(local_mat)
    for j, parent in enumerate(SMPL22_PARENTS):
        if parent < 0:
            global_mat[..., j, :, :] = local_mat[..., j, :, :]
        else:
            global_mat[..., j, :, :] = global_mat[..., parent, :, :] @ local_mat[..., j, :, :]

    return rotation_matrix_to_rot6d(global_mat)  # (..., 22, 6)
