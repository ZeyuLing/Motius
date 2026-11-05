# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Convert poses through the skeleton hierarchy, blend transforms, and apply pose deltas."""

from typing import Any

import numpy as np
import warp as wp

from motius.motion.runtime.motionbricks.time_utils import frame_index_from_time


@wp.kernel
def _fill_poses_kernel(pose: wp.array(dtype=Any), poses: wp.array2d(dtype=Any)):
    frame, joint = wp.tid()
    poses[frame, joint] = pose[joint]


def fill_poses(
    pose: wp.array, poses: wp.array, stream: wp.Stream | None = None
) -> None:
    """Copy one pose into every row of poses.

    Args:
        pose: One pose, as local joint transforms or joint coordinates.
        poses: Array of [frame, joint] or [frame, coordinate] with the same dtype.
        stream: Stream for the work. None uses the current stream.
    """
    wp.launch(
        _fill_poses_kernel,
        dim=poses.shape,
        inputs=[pose],
        outputs=[poses],
        device=poses.device,
        stream=stream,
    )


def sample_root_position(
    positions: np.ndarray,
    time: float,
    sample_rate: float,
    root_offset: wp.transform | None = None,
) -> np.ndarray:
    """Interpolate cached root positions and optionally place them in world space.

    Times outside the positions are clamped.

    Args:
        positions: Root position of each frame, as a host array of [frame, 3].
        time: Sample time in seconds.
        sample_rate: Sample rate in Hz.
        root_offset: World transform applied to the result. None returns the position
            as stored.
    """
    frame, next_frame, blend = frame_index_from_time(time, sample_rate, len(positions))
    position = positions[frame] * (1.0 - blend) + positions[next_frame] * blend
    if root_offset is not None:
        position = np.asarray(
            wp.transform_point(root_offset, wp.vec3(*position)), dtype=np.float32
        )
    return position


def _as_2d_view(poses: wp.array) -> wp.array:
    """Return a 1D pose as a 2D view with one row, or a 2D array unchanged."""
    if poses.dtype != wp.transform or poses.ndim not in (1, 2):
        raise ValueError("Poses must be a 1D or 2D Warp transform array.")
    if poses.ndim == 1:
        return poses.reshape((1, poses.shape[0]))
    return poses


@wp.func
def compute_world_pose_frame(
    num_joints: wp.int32,
    root_tx: wp.transform,
    input_frame: wp.int32,
    output_frame: wp.int32,
    parent_indices: wp.array(dtype=wp.int32),
    local_poses: wp.array2d(dtype=wp.transform),
    world_poses: wp.array2d(dtype=wp.transform),
):
    """Compose one frame of local joint transforms through the hierarchy.

    root_tx places the root and sets the space of the result. Each parent comes before
    its children.
    """
    world_poses[output_frame, 0] = wp.mul(root_tx, local_poses[input_frame, 0])
    for joint in range(1, num_joints):
        parent = parent_indices[joint]
        world_poses[output_frame, joint] = wp.mul(
            world_poses[output_frame, parent], local_poses[input_frame, joint]
        )


@wp.kernel
def _compute_world_poses_kernel(
    num_joints: wp.int32,
    root_tx: wp.transform,
    parent_indices: wp.array(dtype=wp.int32),
    local_poses: wp.array2d(dtype=wp.transform),
    world_poses: wp.array2d(dtype=wp.transform),
):
    frame = wp.tid()
    compute_world_pose_frame(
        num_joints, root_tx, frame, frame, parent_indices, local_poses, world_poses
    )


def compute_world_poses(
    local_poses: wp.array,
    parent_indices: wp.array,
    world_poses: wp.array,
    root_tx=wp.transform_identity(),
    num_poses: int | None = None,
    stream: wp.Stream | None = None,
) -> None:
    """Compute one or more world poses into existing storage.

    Args:
        local_poses: Local joint transforms of one pose [joint] or several [pose,
            joint].
        parent_indices: Parent index of each joint. Each parent comes before its
            children.
        world_poses: Array that receives the world transforms, in the same layout.
        root_tx: Transform applied to each root. It sets the space of the result.
        num_poses: Number of poses to compute. None computes them all.
        stream: Stream for the work. None uses the current stream.
    """
    local_batch = _as_2d_view(local_poses)
    world_batch = _as_2d_view(world_poses)
    count = local_batch.shape[0] if num_poses is None else num_poses
    wp.launch(
        _compute_world_poses_kernel,
        dim=count,
        inputs=[local_batch.shape[1], root_tx, parent_indices, local_batch],
        outputs=[world_batch],
        stream=stream,
    )


@wp.kernel
def _compute_local_poses_kernel(
    root_tx: wp.transform,
    parent_indices: wp.array(dtype=wp.int32),
    world_poses: wp.array2d(dtype=wp.transform),
    local_poses: wp.array2d(dtype=wp.transform),
):
    frame, joint = wp.tid()
    parent = parent_indices[joint]
    if parent < 0:
        parent_tx = root_tx
    else:
        parent_tx = world_poses[frame, parent]
    local_poses[frame, joint] = wp.mul(
        wp.transform_inverse(parent_tx), world_poses[frame, joint]
    )


def compute_local_poses(
    world_poses: wp.array,
    parent_indices: wp.array,
    local_poses: wp.array,
    root_tx=wp.transform_identity(),
    num_poses: int | None = None,
    stream: wp.Stream | None = None,
) -> None:
    """Compute the local joint transforms of one or more poses into existing storage.

    Args:
        world_poses: World transforms of one pose [joint] or several [pose, joint].
        parent_indices: Parent index of each joint, or -1 for the root.
        local_poses: Array that receives the local joint transforms, in the same
            layout.
        root_tx: Transform that each root is measured from.
        num_poses: Number of poses to compute. None computes them all.
        stream: Stream for the work. None uses the current stream.
    """
    world_batch = _as_2d_view(world_poses)
    local_batch = _as_2d_view(local_poses)
    count = world_batch.shape[0] if num_poses is None else num_poses
    wp.launch(
        _compute_local_poses_kernel,
        dim=(count, world_batch.shape[1]),
        inputs=[root_tx, parent_indices, world_batch],
        outputs=[local_batch],
        stream=stream,
    )


@wp.func
def blend_transform(
    transform0: wp.transform, transform1: wp.transform, blend: wp.float32
) -> wp.transform:
    """Interpolate positions linearly and rotations spherically between two transforms."""
    p = wp.lerp(transform0.p, transform1.p, blend)
    q = wp.quat_slerp(transform0.q, transform1.q, blend)
    return wp.transform(p, q)


@wp.kernel
def blend_pose_joints_with_root_offset_kernel(
    local_poses: wp.array2d(dtype=wp.transform),
    frame0: wp.int32,
    frame1: wp.int32,
    blend: wp.float32,
    root_offset: wp.transform,
    output_pose: wp.array(dtype=wp.transform),
):
    """Blend two frames into output_pose, and apply root_offset to the root."""
    joint = wp.tid()
    transform = blend_transform(
        local_poses[frame0, joint], local_poses[frame1, joint], blend
    )
    if joint == 0:
        transform = wp.mul(root_offset, transform)
    output_pose[joint] = transform


@wp.func
def compute_delta_transform(
    src_transform: wp.transform, dst_transform: wp.transform
) -> wp.transform:
    """Return the delta from ``src_transform`` to ``dst_transform``."""
    return wp.mul(wp.transform_inverse(src_transform), dst_transform)


@wp.func
def apply_delta_transform(
    transform: wp.transform, delta_transform: wp.transform, weight: wp.float32
) -> wp.transform:
    """Apply a weighted transform delta."""
    weighted_delta = blend_transform(wp.transform_identity(), delta_transform, weight)
    return wp.mul(transform, weighted_delta)


@wp.func
def compute_root_endpoint_delta(
    generated_root: wp.transform, guide_root: wp.transform
) -> wp.transform:
    """Return the endpoint correction that moves a generated root onto a context or target root.

    The position delta is a world space offset. The rotation delta applies after the
    generated rotation.
    """
    rotation = wp.mul(
        wp.quat_inverse(wp.normalize(generated_root.q)), wp.normalize(guide_root.q)
    )
    return wp.transform(guide_root.p - generated_root.p, wp.normalize(rotation))


@wp.func
def apply_root_endpoint_correction(
    root: wp.transform,
    start_delta: wp.transform,
    end_delta: wp.transform,
    end_weight: wp.float32,
) -> wp.transform:
    """Apply the start and end root corrections at one frame of the generated motion.

    end_weight runs from 0 at the first frame to 1 at the last frame. The start
    correction fades out as the end correction fades in. Positions blend linearly, so
    the root does not swing along an arc.
    """
    start_weight = 1.0 - end_weight
    rotation = wp.mul(
        wp.mul(root.q, wp.quat_slerp(wp.quat_identity(), start_delta.q, start_weight)),
        wp.quat_slerp(wp.quat_identity(), end_delta.q, end_weight),
    )
    return wp.transform(
        root.p + start_delta.p * start_weight + end_delta.p * end_weight,
        wp.normalize(rotation),
    )


@wp.kernel
def _compute_delta_pose_kernel(
    src_pose: wp.array(dtype=wp.transform),
    dst_pose: wp.array(dtype=wp.transform),
    delta_pose: wp.array(dtype=wp.transform),
):
    joint = wp.tid()
    delta_pose[joint] = compute_delta_transform(src_pose[joint], dst_pose[joint])


@wp.kernel
def _apply_delta_pose_kernel(
    delta_pose: wp.array(dtype=wp.transform),
    weight: wp.float32,
    pose: wp.array(dtype=wp.transform),
):
    joint = wp.tid()
    pose[joint] = apply_delta_transform(pose[joint], delta_pose[joint], weight)


def _validate_matching_poses(*poses: wp.array) -> None:
    if any(pose.dtype != wp.transform or pose.ndim != 1 for pose in poses):
        raise ValueError("Poses must be 1D Warp transform arrays.")
    if any(pose.shape != poses[0].shape for pose in poses[1:]):
        raise ValueError("Pose storage must have matching shapes.")


def compute_delta_pose(
    src_pose: wp.array,
    dst_pose: wp.array,
    output_delta_pose: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Compute the transform delta of each joint from src_pose to dst_pose.

    Args:
        src_pose: Start pose, one transform per joint.
        dst_pose: End pose, with the same shape.
        output_delta_pose: Array that receives one delta per joint. Applying it to
            src_pose gives dst_pose.
        stream: Stream for the work. None uses the current stream.
    """
    _validate_matching_poses(src_pose, dst_pose, output_delta_pose)
    wp.launch(
        _compute_delta_pose_kernel,
        dim=src_pose.shape[0],
        inputs=[src_pose, dst_pose],
        outputs=[output_delta_pose],
        device=output_delta_pose.device,
        stream=stream,
    )


def apply_delta_pose(
    pose: wp.array, delta_pose: wp.array, weight: float, stream: wp.Stream | None = None
) -> None:
    """Apply one weighted delta pose in place.

    Args:
        pose: Pose to change, one transform per joint.
        delta_pose: Deltas from compute_delta_pose(), with the same shape.
        weight: Share of each delta to apply, clamped to 0 to 1.
        stream: Stream for the work. None uses the current stream.
    """
    _validate_matching_poses(pose, delta_pose)
    wp.launch(
        _apply_delta_pose_kernel,
        dim=pose.shape[0],
        inputs=[delta_pose, wp.clamp(weight, 0.0, 1.0)],
        outputs=[pose],
        device=pose.device,
        stream=stream,
    )


@wp.func
def blend_additive_accumulate_transform(
    accumulated: wp.transform, pose: wp.transform, weight: wp.float32
) -> wp.transform:
    """Add a weighted transform to a running sum.

    The rotation is flipped onto the sum's side of the quaternion sphere first.
    """
    rotation = pose.q
    rotation = wp.where(wp.dot(accumulated.q, rotation) < 0.0, -rotation, rotation)
    return accumulated + wp.transform(pose.p, rotation) * weight


@wp.func
def blend_additive_end_transform(
    accumulated: wp.transform, default_transform: wp.transform, weight_sum: wp.float32
) -> wp.transform:
    """Finish an additive blend and normalize its rotation.

    Weight missing below one comes from default_transform.
    """
    if weight_sum < 1.0:
        accumulated = blend_additive_accumulate_transform(
            accumulated, default_transform, 1.0 - weight_sum
        )
    return wp.transform(accumulated.p, wp.normalize(accumulated.q))
