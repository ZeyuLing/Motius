"""Skeleton joint names and parent arrays.

The SMPL-22 body tree is the canonical skeleton used by ``motion_135`` /
``motion_138`` / ``motion_198`` / HML263-recovered joints / MS272. ``parents[j]``
is the index of joint ``j``'s parent; the root uses ``-1``.

SOMA-30/77 and Unitree G1 skeletons are defined in the retarget modules
(``motius.motion.retarget.smpl_soma`` / ``smpl_g1``) and re-exported here for
discoverability when those modules are importable.
"""

from __future__ import annotations

from typing import List

# SMPL-22 (SMPL body subset used by HumanML3D / HyMotion). Index order matches
# the motion_135 layout documented in motius/models/CLAUDE.md.
SMPL22_NAMES: List[str] = [
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

SMPL22_PARENTS: List[int] = [
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
]

assert len(SMPL22_NAMES) == len(SMPL22_PARENTS) == 22

# Convenient joint-group indices on SMPL-22 (used by quality metrics / masks).
SMPL22_FOOT_JOINTS: List[int] = [7, 8, 10, 11]      # ankles + feet
SMPL22_LEG_JOINTS: List[int] = [1, 2, 4, 5, 7, 8, 10, 11]
SMPL22_END_EFFECTORS: List[int] = [10, 11, 20, 21]  # feet + wrists


__all__ = [
    "SMPL22_NAMES",
    "SMPL22_PARENTS",
    "SMPL22_FOOT_JOINTS",
    "SMPL22_LEG_JOINTS",
    "SMPL22_END_EFFECTORS",
]
