# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
# ruff: noqa: F811 -- Warp registers overloads by signature.
"""Read and write root transforms in arrays of joint coordinates."""

import warp as wp

ROOT_COORDINATE_COUNT = 7


@wp.func
def read_root_transform(joint_q: wp.array(dtype=wp.float32)):
    """Read the root transform of one pose of joint coordinates, with a normalized rotation."""
    return wp.transform(
        wp.vec3(joint_q[0], joint_q[1], joint_q[2]),
        wp.normalize(wp.quat(joint_q[3], joint_q[4], joint_q[5], joint_q[6])),
    )


@wp.func
def read_root_transform(joint_q: wp.array2d(dtype=wp.float32), frame: wp.int32):
    """Read a frame's root transform from joint coordinates, with a normalized rotation."""
    return wp.transform(
        wp.vec3(joint_q[frame, 0], joint_q[frame, 1], joint_q[frame, 2]),
        wp.normalize(
            wp.quat(
                joint_q[frame, 3],
                joint_q[frame, 4],
                joint_q[frame, 5],
                joint_q[frame, 6],
            )
        ),
    )


@wp.func
def read_root_transform(
    joint_q: wp.array3d(dtype=wp.float32), target: wp.int32, frame: wp.int32
):
    """Read the root transform of one blend target's frame from stacked joint coordinates."""
    return wp.transform(
        wp.vec3(
            joint_q[target, frame, 0],
            joint_q[target, frame, 1],
            joint_q[target, frame, 2],
        ),
        wp.normalize(
            wp.quat(
                joint_q[target, frame, 3],
                joint_q[target, frame, 4],
                joint_q[target, frame, 5],
                joint_q[target, frame, 6],
            )
        ),
    )


@wp.func
def write_root_transform(
    root: wp.transform, output: wp.array2d(dtype=wp.float32), frame: wp.int32
):
    """Write a root transform, normalized, into a frame of joint coordinates."""
    for coordinate in range(3):
        output[frame, coordinate] = root.p[coordinate]
    rotation = wp.normalize(root.q)
    for coordinate in range(4):
        output[frame, coordinate + 3] = rotation[coordinate]


@wp.func
def write_root_transform(root: wp.transform, output: wp.array(dtype=wp.float32)):
    """Write a root transform, normalized, into the joint coordinates of one pose."""
    for coordinate in range(3):
        output[coordinate] = root.p[coordinate]
    rotation = wp.normalize(root.q)
    for coordinate in range(4):
        output[coordinate + 3] = rotation[coordinate]
