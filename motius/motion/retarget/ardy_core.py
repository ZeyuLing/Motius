"""ARDY Core-27 to SMPL-22 joint retargeting helpers.

The released ARDY Core checkpoints use NVIDIA's ``cskel27`` animation
skeleton. It is not an SMPL-family skeleton. The mapping below is therefore a
joint-position bridge only: it is suitable for visualization and evaluator
inputs that consume SMPL-22 joint positions, but it does not recover SMPL
twist, shape, or a valid ``motion135`` rotation sequence.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


ARDY_CORE27_NAMES = [
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
]

ARDY_CORE27_INDEX: Mapping[str, int] = {
    name: index for index, name in enumerate(ARDY_CORE27_NAMES)
}

# SMPL-22 order from motius.motion.skeleton.names.SMPL22_NAMES.
SMPL22_FROM_ARDY_CORE27 = [
    ("Hips",),          # Pelvis
    ("LeftUpLeg",),     # L_Hip
    ("RightUpLeg",),    # R_Hip
    ("Spine",),         # Spine1
    ("LeftLeg",),       # L_Knee
    ("RightLeg",),      # R_Knee
    ("Spine2",),        # Spine2
    ("LeftFoot",),      # L_Ankle
    ("RightFoot",),     # R_Ankle
    ("Spine3",),        # Spine3
    ("LeftToeBase",),   # L_Foot
    ("RightToeBase",),  # R_Foot
    ("Neck",),          # Neck
    ("LeftShoulder",),  # L_Collar
    ("RightShoulder",), # R_Collar
    ("Head",),          # Head
    ("LeftArm",),       # L_Shoulder
    ("RightArm",),      # R_Shoulder
    ("LeftForeArm",),   # L_Elbow
    ("RightForeArm",),  # R_Elbow
    ("LeftHand",),      # L_Wrist
    ("RightHand",),     # R_Wrist
]


def ardy_core27_to_smpl22_joints(joints, *, recenter_root: bool = False) -> np.ndarray:
    """Map ARDY Core-27 joint positions to SMPL-22 joint order.

    Parameters
    ----------
    joints:
        Array ending in ``(27, 3)`` in ARDY Core coordinates.
    recenter_root:
        If true, subtract the first-frame pelvis X/Z translation while keeping
        vertical height. This is useful for qualitative viewers; evaluators
        should normally preserve the generated root trajectory.
    """

    value = np.asarray(joints, dtype=np.float32)
    if value.shape[-2:] != (27, 3):
        raise ValueError(f"ARDY Core joints must end in (27, 3), got {value.shape}")

    mapped = np.stack(
        [
            value[..., ARDY_CORE27_INDEX[source_names[0]], :]
            for source_names in SMPL22_FROM_ARDY_CORE27
        ],
        axis=-2,
    ).astype(np.float32)
    if recenter_root:
        origin = mapped[..., :1, :].copy()
        origin[..., 1] = 0.0
        mapped = mapped - origin
    return mapped


__all__ = [
    "ARDY_CORE27_INDEX",
    "ARDY_CORE27_NAMES",
    "SMPL22_FROM_ARDY_CORE27",
    "ardy_core27_to_smpl22_joints",
]
