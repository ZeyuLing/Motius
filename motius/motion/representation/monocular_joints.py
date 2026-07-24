"""Name-audited joints used for cross-body-model HMR comparisons."""

from __future__ import annotations

from typing import Sequence

import numpy as np


SMPL24_NAMES = (
    "Pelvis",
    "L_Hip",
    "R_Hip",
    "Spine1",
    "L_Knee",
    "R_Knee",
    "Spine2",
    "L_Ankle",
    "R_Ankle",
    "Spine3",
    "L_Foot",
    "R_Foot",
    "Neck",
    "L_Collar",
    "R_Collar",
    "Head",
    "L_Shoulder",
    "R_Shoulder",
    "L_Elbow",
    "R_Elbow",
    "L_Wrist",
    "R_Wrist",
    "L_Hand",
    "R_Hand",
)

# Public 77 articulated joints from NVlabs/SOMA-X
# assets/SOMA_procedural_transforms.json (published v0026). The asset-level
# ``Root`` scene node is excluded from the 77 pose channels exposed by SOMALayer.
SOMA77_NAMES = (
    "Hips",
    "Spine1",
    "Spine2",
    "Chest",
    "Neck1",
    "Neck2",
    "Head",
    "HeadEnd",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandThumb1",
    "LeftHandThumb2",
    "LeftHandThumb3",
    "LeftHandThumbEnd",
    "LeftHandIndex1",
    "LeftHandIndex2",
    "LeftHandIndex3",
    "LeftHandIndex4",
    "LeftHandIndexEnd",
    "LeftHandMiddle1",
    "LeftHandMiddle2",
    "LeftHandMiddle3",
    "LeftHandMiddle4",
    "LeftHandMiddleEnd",
    "LeftHandRing1",
    "LeftHandRing2",
    "LeftHandRing3",
    "LeftHandRing4",
    "LeftHandRingEnd",
    "LeftHandPinky1",
    "LeftHandPinky2",
    "LeftHandPinky3",
    "LeftHandPinky4",
    "LeftHandPinkyEnd",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandThumb1",
    "RightHandThumb2",
    "RightHandThumb3",
    "RightHandThumbEnd",
    "RightHandIndex1",
    "RightHandIndex2",
    "RightHandIndex3",
    "RightHandIndex4",
    "RightHandIndexEnd",
    "RightHandMiddle1",
    "RightHandMiddle2",
    "RightHandMiddle3",
    "RightHandMiddle4",
    "RightHandMiddleEnd",
    "RightHandRing1",
    "RightHandRing2",
    "RightHandRing3",
    "RightHandRing4",
    "RightHandRingEnd",
    "RightHandPinky1",
    "RightHandPinky2",
    "RightHandPinky3",
    "RightHandPinky4",
    "RightHandPinkyEnd",
    "LeftLeg",
    "LeftShin",
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightLeg",
    "RightShin",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)

COMMON_HMR15_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "neck",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)

COMMON_HMR15_FROM_SMPL24 = {
    "pelvis": "Pelvis",
    "left_hip": "L_Hip",
    "right_hip": "R_Hip",
    "left_knee": "L_Knee",
    "right_knee": "R_Knee",
    "left_ankle": "L_Ankle",
    "right_ankle": "R_Ankle",
    "neck": "Neck",
    "head": "Head",
    "left_shoulder": "L_Shoulder",
    "right_shoulder": "R_Shoulder",
    "left_elbow": "L_Elbow",
    "right_elbow": "R_Elbow",
    "left_wrist": "L_Wrist",
    "right_wrist": "R_Wrist",
}

COMMON_HMR15_FROM_SOMA77 = {
    "pelvis": "Hips",
    "left_hip": "LeftLeg",
    "right_hip": "RightLeg",
    "left_knee": "LeftShin",
    "right_knee": "RightShin",
    "left_ankle": "LeftFoot",
    "right_ankle": "RightFoot",
    "neck": "Neck1",
    "head": "Head",
    "left_shoulder": "LeftArm",
    "right_shoulder": "RightArm",
    "left_elbow": "LeftForeArm",
    "right_elbow": "RightForeArm",
    "left_wrist": "LeftHand",
    "right_wrist": "RightHand",
}


def select_common_hmr15(
    joints: np.ndarray,
    joint_names: Sequence[str],
    *,
    body_model: str,
) -> np.ndarray:
    """Select the common 15 joints strictly by their published names."""

    array = np.asarray(joints)
    if array.ndim < 2 or array.shape[-1] != 3:
        raise ValueError("joints must end in (joint, xyz).")
    if array.shape[-2] != len(joint_names):
        raise ValueError("joint_names do not match the joint array.")
    normalized_model = (
        body_model.lower().replace("-", "").replace("_", "").replace(" ", "")
    )
    if normalized_model in {
        "smpl",
        "smpl24",
        "smplx",
        "smplxneutral",
        "smplxfemale",
        "smplxmale",
        "smplh",
        "smplhneutral",
        "smplhfemale",
        "smplhmale",
    }:
        mapping = COMMON_HMR15_FROM_SMPL24
    elif normalized_model in {"soma", "soma77", "gemx"}:
        mapping = COMMON_HMR15_FROM_SOMA77
    else:
        raise ValueError(f"No audited common-joint mapping for {body_model!r}.")
    index_by_name = {name: index for index, name in enumerate(joint_names)}
    missing = [mapping[name] for name in COMMON_HMR15_NAMES if mapping[name] not in index_by_name]
    if missing:
        raise ValueError(f"Missing required named joints: {missing}.")
    indices = [index_by_name[mapping[name]] for name in COMMON_HMR15_NAMES]
    return array[..., indices, :]


__all__ = [
    "COMMON_HMR15_FROM_SMPL24",
    "COMMON_HMR15_FROM_SOMA77",
    "COMMON_HMR15_NAMES",
    "SMPL24_NAMES",
    "SOMA77_NAMES",
    "select_common_hmr15",
]
