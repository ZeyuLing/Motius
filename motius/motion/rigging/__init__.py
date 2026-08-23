"""Automatic skeleton fitting and skin binding for static characters."""

from .api import (
    AUTO_RIG_METHODS,
    SUPPORTED_CHARACTER_INPUTS,
    SUPPORTED_RIG_OUTPUTS,
    CharacterRiggingError,
    CharacterRiggingResult,
    auto_rig_character,
)
from .mia import DEFAULT_MIA_SPACE, MIA_REST_POSES, MIXAMO_TO_SMPL22
from .template import (
    SMPL22_RIG_NAMES,
    SMPL22_RIG_PARENTS,
    FittedHumanoidSkeleton,
    SkinWeightResult,
    TemplateRiggingConfig,
    compute_skin_weights,
    fit_humanoid_skeleton,
)

__all__ = [
    "AUTO_RIG_METHODS",
    "DEFAULT_MIA_SPACE",
    "MIA_REST_POSES",
    "MIXAMO_TO_SMPL22",
    "SMPL22_RIG_NAMES",
    "SMPL22_RIG_PARENTS",
    "SUPPORTED_CHARACTER_INPUTS",
    "SUPPORTED_RIG_OUTPUTS",
    "CharacterRiggingError",
    "CharacterRiggingResult",
    "FittedHumanoidSkeleton",
    "SkinWeightResult",
    "TemplateRiggingConfig",
    "auto_rig_character",
    "compute_skin_weights",
    "fit_humanoid_skeleton",
]
