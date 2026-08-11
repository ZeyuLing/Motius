"""SMPL-22 skeletal data (bone offsets, parents, joint names)."""

import numpy as np

# Bone offsets (relative to parent) for SMPL-22 in T-pose
# These are standard SMPL-22 bone lengths used for Forward Kinematics
SMPL22_BONE_OFFSETS = np.array([
    [-0.00, 0.00, 0.00],      # 0: Pelvis (root)
    [-0.09, -0.22, 0.02],     # 1: L_Hip
    [0.09, -0.22, 0.02],      # 2: R_Hip
    [0.00, 0.22, 0.00],       # 3: Spine1
    [0.00, -0.43, 0.00],      # 4: L_Knee
    [0.00, -0.43, 0.00],      # 5: R_Knee
    [0.00, 0.22, 0.00],       # 6: Spine2
    [0.05, -0.42, 0.00],      # 7: L_Ankle
    [-0.05, -0.42, 0.00],     # 8: R_Ankle
    [0.00, 0.22, 0.00],       # 9: Spine3
    [0.05, -0.10, 0.05],      # 10: L_Foot
    [-0.05, -0.10, 0.05],     # 11: R_Foot
    [0.00, 0.15, 0.00],       # 12: Neck
    [-0.18, 0.08, 0.00],      # 13: L_Collar
    [0.18, 0.08, 0.00],       # 14: R_Collar
    [0.00, 0.20, 0.00],       # 15: Head
    [-0.28, 0.00, 0.00],      # 16: L_Shoulder
    [0.28, 0.00, 0.00],       # 17: R_Shoulder
    [-0.25, 0.00, 0.00],      # 18: L_Elbow
    [0.25, 0.00, 0.00],       # 19: R_Elbow
    [-0.12, 0.00, 0.00],      # 20: L_Wrist
    [0.12, 0.00, 0.00],       # 21: R_Wrist
], dtype=np.float32)

# Kinematic tree: parent index for each joint (-1 = root)
SMPL22_PARENTS = [
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

# Joint names
SMPL22_JOINT_NAMES = [
    "Pelvis",      # 0
    "L_Hip",       # 1
    "R_Hip",       # 2
    "Spine1",      # 3
    "L_Knee",      # 4
    "R_Knee",      # 5
    "Spine2",      # 6
    "L_Ankle",     # 7
    "R_Ankle",     # 8
    "Spine3",      # 9
    "L_Foot",      # 10
    "R_Foot",      # 11
    "Neck",        # 12
    "L_Collar",    # 13
    "R_Collar",    # 14
    "Head",        # 15
    "L_Shoulder",  # 16
    "R_Shoulder",  # 17
    "L_Elbow",     # 18
    "R_Elbow",     # 19
    "L_Wrist",     # 20
    "R_Wrist",     # 21
]

__all__ = [
    'SMPL22_BONE_OFFSETS',
    'SMPL22_PARENTS',
    'SMPL22_JOINT_NAMES',
]
