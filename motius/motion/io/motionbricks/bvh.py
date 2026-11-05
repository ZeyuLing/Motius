# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Load skeletons and animation clips from BVH files."""

import logging
import time
from pathlib import Path

import numpy as np
import warp as wp

from motius.motion.runtime.motionbricks.animation_buffer import (
    AnimationBuffer,
    create_animation_buffer_for_skeleton,
)
from motius.motion.runtime.motionbricks.skeleton import Skeleton

logger = logging.getLogger(__name__)


@wp.func
def _euler_to_quaternion(
    euler_angles: wp.array(dtype=wp.float32), rotation_order: wp.array(dtype=wp.int32)
):
    """Convert angles in degrees to a quaternion using the file's channel order.

    Each entry in rotation_order identifies an axis: 0 for X, 1 for Y, 2 for Z.
    The result is a unit XYZW quaternion.
    """
    quaternion = wp.quat_identity()
    for i in range(rotation_order.shape[0]):
        axis = wp.vec3(0.0)
        axis[rotation_order[i]] = 1.0
        rotation = wp.quat_from_axis_angle(axis, wp.radians(euler_angles[i]))
        quaternion *= wp.normalize(rotation)
    return wp.normalize(quaternion)


@wp.kernel
def _convert_frame_animation_kernel(
    reference_local_transforms: wp.array(dtype=wp.transform),
    has_positions: wp.array(dtype=wp.bool),
    has_rotations: wp.array(dtype=wp.bool),
    animation_data_positions: wp.array3d(dtype=wp.float32),
    animation_data_rotations: wp.array3d(dtype=wp.float32),
    rotate_order: wp.array2d(dtype=wp.int32),
    frame_data: wp.array2d(dtype=wp.transform),
):
    """Build a local transform for each joint in each frame.

    Convert positions from centimeters to meters and angles to quaternions.
    Use the reference position or rotation when the joint has no such channels.
    """
    frame, joint_index = wp.tid()
    if has_positions[joint_index]:
        positions = (
            wp.vec3(
                animation_data_positions[frame, joint_index, 0],
                animation_data_positions[frame, joint_index, 1],
                animation_data_positions[frame, joint_index, 2],
            )
            * 0.01
        )
    else:
        positions = reference_local_transforms[joint_index].p
    if has_rotations[joint_index]:
        rotation = _euler_to_quaternion(
            animation_data_rotations[frame, joint_index], rotate_order[joint_index]
        )
    else:
        rotation = reference_local_transforms[joint_index].q
    frame_data[frame, joint_index] = wp.transform(positions, rotation)


def _open_bvh(bvh_file):
    path = Path(bvh_file)
    if not path.exists():
        raise ValueError(f"Could not find BVH file: {bvh_file}.")
    if path.suffix.lower() != ".bvh":
        raise ValueError(f"Expected a BVH file: {bvh_file}.")
    return path.open(encoding="utf-8")


def _read_bvh_hierarchy(source, remove_namespace):
    """Read joints and channel labels, stopping at the MOTION section."""
    names, parents, offsets, joint_channels = ([], [], [], [])
    joint_stack = []
    in_end_site = False
    for line in source:
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "MOTION":
            break
        if tokens[0] in ("ROOT", "JOINT"):
            name = tokens[1]
            names.append(name.rsplit(":", 1)[-1] if remove_namespace else name)
            parents.append(joint_stack[-1] if joint_stack else -1)
            joint_stack.append(len(names) - 1)
            offsets.append([0.0, 0.0, 0.0])
            joint_channels.append([])
        elif tokens[:2] == ["End", "Site"]:
            in_end_site = True
        elif tokens[0] == "OFFSET" and (not in_end_site):
            offsets[joint_stack[-1]] = [float(value) for value in tokens[1:4]]
        elif tokens[0] == "CHANNELS":
            channels = tokens[2:]
            channel_count = int(tokens[1])
            if channel_count != len(channels):
                raise ValueError(
                    f"BVH joint {names[joint_stack[-1]]!r} declares {channel_count} channels but lists {len(channels)}."
                )
            joint_channels[joint_stack[-1]] = channels
        elif tokens[0] == "}":
            if in_end_site:
                in_end_site = False
            else:
                joint_stack.pop()
    reference_pose = np.zeros((len(names), 7), dtype=np.float32)
    reference_pose[:, :3] = np.asarray(offsets, dtype=np.float32) * 0.01
    reference_pose[:, 6] = 1.0
    skeleton = Skeleton(names, parents, reference_pose)
    return (skeleton, joint_channels)


def _read_bvh(path, remove_namespace):
    """Read the skeleton, then load all motion values into a float32 array."""
    frame_count = 0
    with _open_bvh(path) as source:
        skeleton, joint_channels = _read_bvh_hierarchy(source, remove_namespace)
        for line in source:
            tokens = line.split()
            if not tokens:
                continue
            if tokens[0] == "Frames:":
                frame_count = int(tokens[1])
            elif tokens[:2] == ["Frame", "Time:"]:
                sample_rate = round(1.0 / float(tokens[2]), 2)
                break
        else:
            raise ValueError(f"BVH frame time is missing: {path}.")
        samples = np.loadtxt(source, dtype=np.float32, ndmin=2)
    channel_count = sum(len(channels) for channels in joint_channels)
    if samples.shape != (frame_count, channel_count):
        raise ValueError(
            f"BVH motion has shape {samples.shape}. Expected {frame_count} frames and {channel_count} channels."
        )
    return (skeleton, samples, joint_channels, sample_rate)


def _create_animation(skeleton, samples, joint_channels, sample_rate):
    """Copy each channel's values for all frames at once, then build poses in Warp.

    Position channels go into X, Y, Z slots. Rotation channels keep the order
    listed in the file, since changing that order changes the resulting pose.
    """
    frame_count, joint_count = (samples.shape[0], skeleton.num_joints)
    positions = np.zeros((frame_count, joint_count, 3), dtype=np.float32)
    rotations = np.zeros_like(positions)
    has_positions = np.zeros(joint_count, dtype=bool)
    has_rotations = np.zeros(joint_count, dtype=bool)
    rotation_orders = np.zeros((joint_count, 3), dtype=np.int32)
    column = 0
    for joint, channels in enumerate(joint_channels):
        rotation_index = 0
        for channel in channels:
            channel = channel.lower()
            if channel.endswith("position"):
                axis = "xyz".index(channel[0])
                positions[:, joint, axis] = samples[:, column]
                has_positions[joint] = True
            elif channel.endswith("rotation"):
                rotations[:, joint, rotation_index] = samples[:, column]
                rotation_orders[joint, rotation_index] = "xyz".index(channel[0])
                has_rotations[joint] = True
                rotation_index += 1
            column += 1
    local_transforms = wp.empty((frame_count, joint_count), dtype=wp.transform)
    wp.launch(
        _convert_frame_animation_kernel,
        dim=(frame_count, joint_count),
        inputs=[
            skeleton.reference_local_transforms,
            wp.array(has_positions, dtype=wp.bool),
            wp.array(has_rotations, dtype=wp.bool),
            wp.array(positions, dtype=wp.float32),
            wp.array(rotations, dtype=wp.float32),
            wp.array(rotation_orders, dtype=wp.int32),
        ],
        outputs=[local_transforms],
    )
    return AnimationBuffer(skeleton, sample_rate, local_transforms)


def load_skeleton_from_bvh(file_path, *, remove_namespace=True):
    """Read a BVH skeleton without loading its motion data.

    Args:
        file_path: Path to an existing .bvh file.
        remove_namespace: If True (the default), shorten names such as
            "actor:rig:Hips" to "Hips". Set False to keep the full names.
    """
    with _open_bvh(file_path) as source:
        skeleton, _ = _read_bvh_hierarchy(source, remove_namespace)
    return skeleton


def load_bvh(file_path, skeleton=None, *, remove_namespace=True):
    """Load an animation clip from a BVH file.

    Positions are converted from centimeters to meters. Rotation angles are
    read in degrees and applied in the order listed for each joint.

    Every motion row must have one value per listed channel, and the number of
    frames must match Frames: in the header.

    Args:
        file_path: Path to an existing .bvh file.
        skeleton: Skeleton to match by exact joint name. Its joints missing from the
            file keep their reference transforms. None keeps the file's skeleton.
        remove_namespace: If True (the default), shorten names such as
            "actor:rig:Hips" to "Hips". Set False to keep the full names.

    Returns:
        The clip. Its skeleton is the given skeleton, or the file's skeleton when
        skeleton is None.
    """
    start = time.perf_counter()
    file_skeleton, samples, joint_channels, sample_rate = _read_bvh(
        file_path, remove_namespace
    )
    animation = _create_animation(file_skeleton, samples, joint_channels, sample_rate)
    if skeleton is not None:
        animation = create_animation_buffer_for_skeleton(animation, skeleton)
    logger.debug(
        "Loaded %s: %d frames at a sample rate of %g Hz in %.2f s.",
        file_path,
        animation.num_frames,
        animation.sample_rate,
        time.perf_counter() - start,
    )
    return animation
