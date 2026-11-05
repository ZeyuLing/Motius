# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Root alignment, decoding, and endpoint correction of G1 joint coordinates."""

import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_BOUNDARY_FRAMES,
    MODEL_POSES_PER_TOKEN,
)
from motius.motion.runtime.motionbricks.dof_utils import (
    ROOT_COORDINATE_COUNT,
    read_root_transform,
    write_root_transform,
)
from motius.motion.runtime.motionbricks.foot_contacts import FOOT_CONTACT_COUNT
from motius.motion.runtime.motionbricks.math_utils import quadratic_fade_weight
from motius.motion.runtime.motionbricks.pose_utils import (
    apply_root_endpoint_correction,
    compute_root_endpoint_delta,
)

ROOT_POSITION_FILTER = (-0.0465, 0.0639, 0.2827, 0.3998, 0.2827, 0.0639, -0.0465)
JOINT_ANGLE_FADE_SECONDS = 0.3


@wp.kernel
def compute_guide_alignment_kernel(
    desired_heading: wp.vec2,
    runtime_to_model: wp.transform,
    context_joint_q: wp.array2d(dtype=wp.float32),
    alignment: wp.array(dtype=wp.transform),
):
    """Compute the heading alignment of the last context pose to desired_heading.

    runtime_to_model brings the root into G1 model space first. The result maps
    aligned poses back to G1 model space and keeps the root height.
    """
    root = wp.mul(
        runtime_to_model,
        read_root_transform(context_joint_q, MODEL_POSES_PER_TOKEN - 1),
    )
    forward = wp.transform_vector(root, wp.vec3(1.0, 0.0, 0.0))
    current_angle = wp.atan2(forward[1], forward[0])
    desired_angle = wp.atan2(desired_heading[1], desired_heading[0])
    alignment[0] = wp.transform(
        wp.vec3(root.p[0], root.p[1], 0.0),
        wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), current_angle - desired_angle),
    )


@wp.kernel
def encode_motionbricks_inputs_kernel(
    runtime_to_model: wp.transform,
    joint_q: wp.array2d(dtype=wp.float32),
    alignment: wp.array(dtype=wp.transform),
    qpos: wp.array2d(dtype=wp.float32),
):
    """Encode a token's joint coordinates relative to the alignment into context_qpos or target_qpos.

    runtime_to_model brings each root into G1 model space first. The root height and
    the joint angles stay unchanged.
    """
    frame = wp.tid()
    align = alignment[0]
    root = wp.mul(runtime_to_model, read_root_transform(joint_q, frame))
    position = wp.quat_rotate_inv(align.q, root.p - align.p)
    rotation = wp.mul(wp.quat_inverse(align.q), root.q)
    write_root_transform(
        wp.transform(wp.vec3(position[0], position[1], root.p[2]), rotation),
        qpos,
        frame,
    )
    for index in range(ROOT_COORDINATE_COUNT, joint_q.shape[1]):
        qpos[frame, index] = joint_q[frame, index]


@wp.kernel
def decode_motionbricks_output_kernel(
    alignment: wp.array(dtype=wp.transform),
    model_to_runtime: wp.transform,
    pred_qpos: wp.array2d(dtype=wp.float32),
    pred_foot_contacts: wp.array2d(dtype=wp.float32),
    pred_num_tokens: wp.array(dtype=wp.int32),
    root_position_filter: wp.array(dtype=wp.float32),
    output_joint_q: wp.array2d(dtype=wp.float32),
    output_foot_contacts: wp.array2d(dtype=wp.float32),
    output_frame_count: wp.array(dtype=wp.int32),
):
    """Decode the predicted joint coordinates and foot contacts into world space.

    It drops the boundary frames and writes the valid frame count. The root position is
    filtered over neighboring frames. Foot contacts are clamped to 0 to 1. The alignment
    and model_to_runtime place the root in world space.
    """
    frame = wp.tid()
    raw_count = wp.clamp(
        pred_num_tokens[0] * MODEL_POSES_PER_TOKEN, 0, pred_qpos.shape[0]
    )
    frame_count = wp.max(0, raw_count - 2 * MODEL_BOUNDARY_FRAMES)
    if frame == 0:
        output_frame_count[0] = frame_count
    if frame >= frame_count:
        return
    raw_frame = frame + MODEL_BOUNDARY_FRAMES
    for contact in range(FOOT_CONTACT_COUNT):
        output_foot_contacts[frame, contact] = wp.clamp(
            pred_foot_contacts[raw_frame, contact], 0.0, 1.0
        )
    radius = root_position_filter.shape[0] // 2
    position = wp.vec3(0.0, 0.0, 0.0)
    for tap in range(root_position_filter.shape[0]):
        source = wp.clamp(raw_frame + tap - radius, 0, raw_count - 1)
        position += root_position_filter[tap] * read_root_transform(pred_qpos, source).p
    root = wp.mul(
        alignment[0],
        wp.transform(position, read_root_transform(pred_qpos, raw_frame).q),
    )
    root = wp.mul(
        model_to_runtime,
        wp.transform(wp.vec3(root.p[0], root.p[1], position[2]), root.q),
    )
    write_root_transform(root, output_joint_q, frame)
    for index in range(ROOT_COORDINATE_COUNT, output_joint_q.shape[1]):
        output_joint_q[frame, index] = pred_qpos[raw_frame, index]


@wp.kernel
def compute_endpoint_deltas_kernel(
    output_frame_count: wp.array(dtype=wp.int32),
    context_joint_q: wp.array2d(dtype=wp.float32),
    target_joint_q: wp.array2d(dtype=wp.float32),
    output_joint_q: wp.array2d(dtype=wp.float32),
    deltas: wp.array2d(dtype=wp.float32),
):
    """Compute the endpoint correction for the first and last generated frames.

    Row 0 corrects toward the last context pose. Row 1 corrects toward the first target
    pose. Each row holds the root delta in the root coordinates, then one offset per
    joint angle.
    """
    endpoint = wp.tid()
    frame_count = output_frame_count[0]
    if frame_count < 2:
        return
    frame = wp.where(endpoint == 0, 0, frame_count - 1)
    last_context_pose = context_joint_q.shape[0] - 1
    guide_root = read_root_transform(target_joint_q, 0)
    if endpoint == 0:
        guide_root = read_root_transform(context_joint_q, last_context_pose)
    root_delta = compute_root_endpoint_delta(
        read_root_transform(output_joint_q, frame), guide_root
    )
    write_root_transform(root_delta, deltas, endpoint)
    for index in range(ROOT_COORDINATE_COUNT, output_joint_q.shape[1]):
        guide = target_joint_q[0, index]
        if endpoint == 0:
            guide = context_joint_q[last_context_pose, index]
        deltas[endpoint, index] = guide - output_joint_q[frame, index]


@wp.kernel
def apply_endpoint_deltas_kernel(
    output_frame_count: wp.array(dtype=wp.int32),
    deltas: wp.array2d(dtype=wp.float32),
    joint_angle_fade_frames: wp.float32,
    output_joint_q: wp.array2d(dtype=wp.float32),
):
    """Apply the endpoint correction across the generated motion in place.

    Root corrections span the motion. Joint angle corrections fade over
    joint_angle_fade_frames at each end.
    """
    frame = wp.tid()
    frame_count = output_frame_count[0]
    if frame_count < 2 or frame >= frame_count:
        return
    end_weight = wp.float32(frame) / wp.float32(frame_count - 1)
    root = apply_root_endpoint_correction(
        read_root_transform(output_joint_q, frame),
        read_root_transform(deltas, 0),
        read_root_transform(deltas, 1),
        end_weight,
    )
    write_root_transform(root, output_joint_q, frame)
    fade_frames = wp.min(joint_angle_fade_frames, 0.5 * wp.float32(frame_count - 1))
    angle_start_weight = quadratic_fade_weight(wp.float32(frame) / fade_frames)
    angle_end_weight = quadratic_fade_weight(
        wp.float32(frame_count - 1 - frame) / fade_frames
    )
    for index in range(ROOT_COORDINATE_COUNT, output_joint_q.shape[1]):
        output_joint_q[frame, index] += (
            deltas[0, index] * angle_start_weight + deltas[1, index] * angle_end_weight
        )
