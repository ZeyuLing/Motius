"""Compatibility imports for the public position-IK implementation.

The old soft Adam projection and target-shrinking CCD have been replaced by
analytic limb IK and residual-first, simultaneous damped least squares.
"""

from motius.motion.pipeline_utils.position_ik import (  # noqa: F401
    NUM_JOINTS, TWO_BONE_CHAINS, ONE_BONE_TARGETS, GRADIENT_IK_ANCESTORS,
    get_ik_strategy, solve_root_ik, solve_two_bone_ik, solve_one_bone_ik,
    solve_gradient_ik, solve_single_constraint, solve_joint_positions,
    _rotation_between_vectors,
)
