"""Compatibility shim — FK moved to the public motion library.

The canonical forward-kinematics implementation now lives at
``motius.motion.skeleton.fk``. This module re-exports it so legacy imports
such as::

    from motius.motion.pipeline_utils.differentiable_fk import (
        differentiable_fk, motion135_to_fk, fk_to_motion135,
    )

keep working unchanged. New code should import from
``motius.motion.skeleton.fk``. The new local-path math is verified
numerically identical to the legacy code (see
``scripts/debug/verify_motion_lib_migration.py``).
"""

from motius.motion.skeleton.fk import (  # noqa: F401
    NUM_JOINTS,
    differentiable_fk,
    forward_kinematics,
    rot6d_to_rotmat_row_major,
    rotmat_to_rot6d_row_major,
    motion135_to_fk,
    fk_to_motion135,
    local_to_global_rot6d,
    global_to_local_rot6d,
)
from motius.motion.skeleton.names import SMPL22_PARENTS  # noqa: F401
