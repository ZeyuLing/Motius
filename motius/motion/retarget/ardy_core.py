"""Core-27 to SMPL-22 joint retargeting helpers.

The released ARDY Core checkpoints use NVIDIA's ``cskel27`` animation
skeleton. It is not an SMPL-family skeleton. The mapping below is therefore a
joint-position bridge only: it is suitable for visualization and joint-space
evaluator inputs, but it does not recover SMPL twist, shape, or a valid
``motion135`` rotation sequence.
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

_SMPL22_TO_ARDY_CORE27_BY_NAME = {
    "Hips": 0,
    "LeftUpLeg": 1,
    "RightUpLeg": 2,
    "Spine": 3,
    "LeftLeg": 4,
    "RightLeg": 5,
    "Spine2": 6,
    "LeftFoot": 7,
    "RightFoot": 8,
    "Spine3": 9,
    "LeftToeBase": 10,
    "RightToeBase": 11,
    "Neck": 12,
    "LeftShoulder": 13,
    "RightShoulder": 14,
    "Head": 15,
    "LeftArm": 16,
    "RightArm": 17,
    "LeftForeArm": 18,
    "RightForeArm": 19,
    "LeftHand": 20,
    "RightHand": 21,
}

_INTERPOLATED_CORE_JOINTS = {
    "Spine1": ("Spine", "Spine2"),
    "RightHandEnd": ("RightHand", "RightForeArm"),
    "RightHandThumb1": ("RightHand", "RightForeArm"),
    "LeftHandEnd": ("LeftHand", "LeftForeArm"),
    "LeftHandThumb1": ("LeftHand", "LeftForeArm"),
}


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


def smpl22_joints_to_ardy_core27_joints(joints, *, recenter_root: bool = False) -> np.ndarray:
    """Map SMPL-22 joint positions to Core-27 joint order.

    Core has hand end/thumb helper joints that do not exist in SMPL-22. Those
    helper joints are placed by extending the wrist away from the forearm so
    the output remains stable for skeleton visualization and joint evaluators.
    This is not a rotation retargeter and does not produce ARDY's 330D feature
    tensor.
    """

    value = np.asarray(joints, dtype=np.float32)
    if value.shape[-2:] != (22, 3):
        raise ValueError(f"SMPL joints must end in (22, 3), got {value.shape}")

    mapped = np.empty(value.shape[:-2] + (27, 3), dtype=np.float32)
    for core_name in ARDY_CORE27_NAMES:
        core_index = ARDY_CORE27_INDEX[core_name]
        if core_name in _SMPL22_TO_ARDY_CORE27_BY_NAME:
            mapped[..., core_index, :] = value[
                ..., _SMPL22_TO_ARDY_CORE27_BY_NAME[core_name], :
            ]
            continue
        first_name, second_name = _INTERPOLATED_CORE_JOINTS[core_name]
        first = value[..., _SMPL22_TO_ARDY_CORE27_BY_NAME[first_name], :]
        second = value[..., _SMPL22_TO_ARDY_CORE27_BY_NAME[second_name], :]
        if core_name == "Spine1":
            mapped[..., core_index, :] = 0.5 * (first + second)
            continue
        direction = first - second
        scale = 0.22 if "Thumb" in core_name else 0.38
        mapped[..., core_index, :] = first + direction * scale

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
    "smpl22_joints_to_ardy_core27_joints",
]
