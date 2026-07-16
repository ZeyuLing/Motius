"""FBX export and rigged-character retargeting APIs."""

from .api import (
    FBXExportError,
    FBXExportResult,
    SMPLAnimation,
    SMPL_TO_BLENDER,
    export_smpl_fbx,
    resolve_blender_executable,
    retarget_smpl_to_fbx,
)
from .bridge import (
    MotionBridgeResult,
    export_motion_to_fbx,
    export_motion_to_mixamo_fbx,
    g1_joints_to_smpl22_joints,
    motion_to_smpl_animation,
)
from .characters import (
    BUILTIN_MIXAMO_CHARACTERS,
    list_mixamo_characters,
    resolve_mixamo_character,
)

__all__ = [
    "FBXExportError",
    "FBXExportResult",
    "SMPLAnimation",
    "SMPL_TO_BLENDER",
    "export_smpl_fbx",
    "resolve_blender_executable",
    "retarget_smpl_to_fbx",
    "BUILTIN_MIXAMO_CHARACTERS",
    "MotionBridgeResult",
    "export_motion_to_fbx",
    "export_motion_to_mixamo_fbx",
    "g1_joints_to_smpl22_joints",
    "list_mixamo_characters",
    "motion_to_smpl_animation",
    "resolve_mixamo_character",
]
