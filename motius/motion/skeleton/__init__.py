"""Skeleton definitions, forward kinematics and body-model loaders.

Public API:

- ``names``: SMPL-22 (and re-exported SOMA/G1) joint names + parent arrays.
- ``fk``: a single forward-kinematics implementation + ``motion_135`` <-> FK
  helpers, built on the unified rotation module (row-major rot6d).
- ``body_models``: light SMPL / SMPL-X / SMPL-H loaders and rest-skeleton
  resolution.
"""

from motius.motion.skeleton.names import (  # noqa: F401
    SMPL22_NAMES,
    SMPL22_PARENTS,
)
from motius.motion.skeleton.fk import (  # noqa: F401
    forward_kinematics,
    differentiable_fk,
    motion135_to_fk,
    fk_to_motion135,
    local_to_global_rot6d,
    global_to_local_rot6d,
)
from motius.motion.skeleton.body_models import (  # noqa: F401
    SMPLSkeletonModel,
    load_smpl_skeleton_model,
    resolve_smpl_model_path,
    smpl_to_joints,
)
from motius.motion.skeleton.canonical import canonicalize_smpl22_joints

__all__ = [
    "SMPL22_NAMES",
    "SMPL22_PARENTS",
    "forward_kinematics",
    "differentiable_fk",
    "motion135_to_fk",
    "fk_to_motion135",
    "local_to_global_rot6d",
    "global_to_local_rot6d",
    "SMPLSkeletonModel",
    "load_smpl_skeleton_model",
    "resolve_smpl_model_path",
    "smpl_to_joints",
    "canonicalize_smpl22_joints",
]
