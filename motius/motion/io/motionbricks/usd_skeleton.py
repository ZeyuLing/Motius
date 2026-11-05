# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Skeleton-only USD loading; no skinned mesh or renderer dependency."""

import logging

import numpy as np
import warp as wp
from pxr import Tf, Usd, UsdGeom, UsdSkel

from motius.motion.runtime.motionbricks.skeleton import Skeleton

logger = logging.getLogger(__name__)
_EPSILON = 1e-6


def _resolve_time_code(stage):
    """Return the time code for evaluating transforms and attributes.

    A stage with a time range uses its start time. Exporters often animate
    xformOps without a meaningful default value, and bind pose loading expects
    the first frame.
    """
    start = stage.GetStartTimeCode()
    if start < stage.GetEndTimeCode():
        return Usd.TimeCode(start)
    return Usd.TimeCode.Default()


def _open_stage(usd_file_path):
    """Open a USD stage. Every failure raises ValueError."""
    try:
        stage = Usd.Stage.Open(str(usd_file_path))
    except Tf.ErrorException as exc:
        raise ValueError(f"Could not open USD stage: {usd_file_path}.") from exc
    if stage is None:
        raise ValueError(f"Could not open USD stage: {usd_file_path}.")
    return stage


def get_stage_meters_per_unit(stage):
    """Return the stage's meters per unit, such as 0.01 for centimeters.

    Args:
        stage: Open USD stage.
    """
    return UsdGeom.GetStageMetersPerUnit(stage)


def get_stage_up_axis(stage):
    """Return the stage's up axis, 'Y' or 'Z'.

    Args:
        stage: Open USD stage.
    """
    up_axis = UsdGeom.GetStageUpAxis(stage)
    return "Y" if up_axis == UsdGeom.Tokens.y else "Z"


def decompose_matrix(mat, translation_scale=1.0):
    """Split a row major 4x4 matrix into a position, a rotation, and a scale.

    Returns (wp.vec3, wp.quat, np.ndarray).

    Args:
        mat: Row major 4x4 matrix, as USD stores it.
        translation_scale: Factor applied to the position, such as 0.01 to convert
            centimeters to meters.
    """
    pos = mat[3, :3] * translation_scale
    rot_matrix = mat[:3, :3].copy()
    scale = np.array(
        [
            np.linalg.norm(rot_matrix[:, 0]),
            np.linalg.norm(rot_matrix[:, 1]),
            np.linalg.norm(rot_matrix[:, 2]),
        ]
    )
    for i in range(3):
        if scale[i] > _EPSILON:
            rot_matrix[:, i] /= scale[i]
    rot = wp.quat_from_matrix(wp.mat33(rot_matrix.T.flatten()))
    return (wp.vec3(pos[0], pos[1], pos[2]), rot, scale)


def get_prim_world_scale(prim, time_code):
    """Return a prim's world scale [sx, sy, sz] from its world transform.

    Args:
        prim: USD prim.
        time_code: Time at which to evaluate the transform.
    """
    xform_cache = UsdGeom.XformCache(time_code)
    mat = np.array(xform_cache.GetLocalToWorldTransform(prim), dtype=np.float32)
    scale = np.array(
        [
            np.linalg.norm(mat[:3, 0]),
            np.linalg.norm(mat[:3, 1]),
            np.linalg.norm(mat[:3, 2]),
        ]
    )
    return scale


def get_prim_meters_scale(stage, prim):
    """Return the factor that converts a prim's data to meters.

    It combines the stage's meters per unit with the prim's world scale. The
    scale is assumed uniform, so only its X component is used.

    Args:
        stage: Stage that holds the prim.
        prim: Prim whose data to convert.
    """
    meters_per_unit = get_stage_meters_per_unit(stage)
    prim_scale = get_prim_world_scale(prim, _resolve_time_code(stage))
    combined_scale = meters_per_unit * prim_scale[0]
    return combined_scale


def _load_skeleton(stage, skeleton_prim_path):
    skeleton_prim = stage.GetPrimAtPath(skeleton_prim_path)
    if not skeleton_prim.IsValid():
        raise ValueError(f"Skeleton prim not found at path {skeleton_prim_path}.")
    usd_skeleton = UsdSkel.Skeleton(skeleton_prim)
    joint_paths = usd_skeleton.GetJointsAttr().Get()
    rest_transforms = usd_skeleton.GetRestTransformsAttr().Get()
    if not joint_paths or not rest_transforms:
        raise ValueError(
            f"Skeleton {skeleton_prim_path} has no joints or rest transforms."
        )
    if len(joint_paths) != len(rest_transforms):
        raise ValueError(
            f"Skeleton {skeleton_prim_path} has {len(joint_paths)} joints but {len(rest_transforms)} rest transforms."
        )
    translation_scale = get_prim_meters_scale(stage, skeleton_prim)
    joint_path_strings = [str(path) for path in joint_paths]
    path_to_index = {path: index for index, path in enumerate(joint_path_strings)}
    joint_names = [path.rsplit("/", 1)[-1] for path in joint_path_strings]
    if len(set(joint_names)) != len(joint_names):
        raise ValueError(f"Skeleton {skeleton_prim_path} has duplicate joint names.")
    parent_indices = []
    for path in joint_path_strings:
        parent_path = path.rsplit("/", 1)[0] if "/" in path else None
        parent_indices.append(-1 if parent_path is None else path_to_index[parent_path])
    local_transforms = []
    for matrix in rest_transforms:
        position, rotation, _ = decompose_matrix(
            np.asarray(matrix, dtype=np.float32), translation_scale=translation_scale
        )
        local_transforms.append(wp.transform(position, rotation))
    skeleton = Skeleton(joint_names, parent_indices, local_transforms)
    if get_stage_up_axis(stage) == "Y":
        skeleton.up_axis = wp.vec3(0.0, 1.0, 0.0)
        skeleton.forward_axis = wp.vec3(0.0, 0.0, 1.0)
    return skeleton


def load_skeleton_from_usd(usd_file_path, skeleton_prim_path):
    """Load a Skeleton with its reference pose from USD.

    A stage that fails to open, or a missing skeleton, raises ValueError.

    Args:
        usd_file_path: USD file to open.
        skeleton_prim_path: Path of the UsdSkel.Skeleton prim.
    """
    stage = _open_stage(usd_file_path)
    return _load_skeleton(stage, skeleton_prim_path)
