# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Warp kernels for the SOMA MotionBricks pose contract."""

import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_BOUNDARY_FRAMES,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
)
from motius.motion.runtime.motionbricks import pose_utils
from motius.motion.runtime.motionbricks.foot_contacts import FOOT_CONTACT_COUNT


@wp.func
def _heading_from_hip_positions(right_hip: wp.vec3, left_hip: wp.vec3) -> wp.float32:
    across = wp.vec3(right_hip[0] - left_hip[0], 0.0, right_hip[2] - left_hip[2])
    length = wp.length(across)
    if length <= 1e-06:
        return 0.0
    across = across / length
    forward = wp.cross(wp.vec3(0.0, 1.0, 0.0), across)
    return wp.atan2(forward[0], forward[2])


@wp.func
def _rotation_from_6d(x_raw: wp.vec3, y_raw: wp.vec3) -> wp.quat:
    x_axis = wp.normalize(x_raw)
    z_axis = wp.normalize(wp.cross(x_axis, y_raw))
    y_axis = wp.cross(z_axis, x_axis)
    return wp.quat_from_matrix(wp.matrix_from_cols(x_axis, y_axis, z_axis))


@wp.func
def _read_rotation(
    features: wp.array3d(dtype=wp.float32), frame: wp.int32, offset: wp.int32
) -> wp.quat:
    x_axis = wp.vec3(
        features[0, frame, offset],
        features[0, frame, offset + 1],
        features[0, frame, offset + 2],
    )
    y_axis = wp.vec3(
        features[0, frame, offset + 3],
        features[0, frame, offset + 4],
        features[0, frame, offset + 5],
    )
    return _rotation_from_6d(x_axis, y_axis)


@wp.kernel
def compute_guide_alignment_kernel(
    right_leg_index: wp.int32,
    left_leg_index: wp.int32,
    desired_heading: wp.vec2,
    world_poses: wp.array2d(dtype=wp.transform),
    alignment: wp.array(dtype=wp.transform),
):
    """Compute the heading alignment of the last context pose to desired_heading.

    The heading comes from the positions of the leg joints. The result maps aligned
    poses back to SOMA model space and keeps the root height.
    """
    last_context_frame = MODEL_POSES_PER_TOKEN - 1
    hips_position = world_poses[last_context_frame, 0].p
    current_angle = _heading_from_hip_positions(
        world_poses[last_context_frame, right_leg_index].p,
        world_poses[last_context_frame, left_leg_index].p,
    )
    desired_angle = wp.atan2(desired_heading[0], desired_heading[1])
    alignment[0] = wp.transform(
        wp.vec3(hips_position[0], 0.0, hips_position[2]),
        wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), current_angle - desired_angle),
    )


@wp.func
def _aligned_position(alignment: wp.transform, position: wp.vec3) -> wp.vec3:
    return wp.quat_rotate_inv(alignment.q, position - alignment.p)


@wp.kernel
def encode_motionbricks_inputs_kernel(
    num_model_joints: wp.int32,
    right_leg_index: wp.int32,
    left_leg_index: wp.int32,
    model_to_pose: wp.array(dtype=wp.int32),
    world_poses: wp.array2d(dtype=wp.transform),
    alignment: wp.array(dtype=wp.transform),
    global_root_values: wp.array3d(dtype=wp.float32),
    local_root_values: wp.array3d(dtype=wp.float32),
    local_poses: wp.array3d(dtype=wp.float32),
):
    """Encode each context and target pose into the model's root and pose input tensors.

    Positions and headings are measured after the alignment. Root velocities use the
    next pose, so they are zero at the last context pose and the last target pose.
    """
    frame = wp.tid()
    final_frame = 2 * MODEL_POSES_PER_TOKEN - 1
    next_frame = wp.where(
        frame == MODEL_POSES_PER_TOKEN - 1, frame, wp.min(frame + 1, final_frame)
    )
    align = alignment[0]
    current_hips = _aligned_position(align, world_poses[frame, 0].p)
    next_hips = _aligned_position(align, world_poses[next_frame, 0].p)
    current_right = _aligned_position(align, world_poses[frame, right_leg_index].p)
    current_left = _aligned_position(align, world_poses[frame, left_leg_index].p)
    next_right = _aligned_position(align, world_poses[next_frame, right_leg_index].p)
    next_left = _aligned_position(align, world_poses[next_frame, left_leg_index].p)
    current_heading = _heading_from_hip_positions(current_right, current_left)
    next_heading = _heading_from_hip_positions(next_right, next_left)
    heading_delta = wp.atan2(
        wp.sin(next_heading - current_heading), wp.cos(next_heading - current_heading)
    )
    global_root_values[0, frame, 0] = current_hips[0]
    global_root_values[0, frame, 1] = current_hips[1]
    global_root_values[0, frame, 2] = current_hips[2]
    global_root_values[0, frame, 3] = wp.cos(current_heading)
    global_root_values[0, frame, 4] = wp.sin(current_heading)
    local_root_values[0, frame, 0] = heading_delta * MODEL_SAMPLE_RATE
    local_root_values[0, frame, 1] = (
        next_hips[0] - current_hips[0]
    ) * MODEL_SAMPLE_RATE
    local_root_values[0, frame, 2] = (
        next_hips[2] - current_hips[2]
    ) * MODEL_SAMPLE_RATE
    local_root_values[0, frame, 3] = current_hips[1]
    root_position = wp.vec3(current_hips[0], 0.0, current_hips[2])
    for model_joint in range(num_model_joints):
        pose_joint = model_to_pose[model_joint]
        joint_position = _aligned_position(align, world_poses[frame, pose_joint].p)
        joint_rotation = wp.mul(
            wp.quat_inverse(align.q), world_poses[frame, pose_joint].q
        )
        if model_joint > 0:
            position_offset = (model_joint - 1) * 3
            centered_position = joint_position - root_position
            local_poses[0, frame, position_offset] = centered_position[0]
            local_poses[0, frame, position_offset + 1] = centered_position[1]
            local_poses[0, frame, position_offset + 2] = centered_position[2]
        rotation_offset = (num_model_joints - 1) * 3 + model_joint * 6
        x_axis = wp.quat_rotate(joint_rotation, wp.vec3(1.0, 0.0, 0.0))
        y_axis = wp.quat_rotate(joint_rotation, wp.vec3(0.0, 1.0, 0.0))
        local_poses[0, frame, rotation_offset] = x_axis[0]
        local_poses[0, frame, rotation_offset + 1] = x_axis[1]
        local_poses[0, frame, rotation_offset + 2] = x_axis[2]
        local_poses[0, frame, rotation_offset + 3] = y_axis[0]
        local_poses[0, frame, rotation_offset + 4] = y_axis[1]
        local_poses[0, frame, rotation_offset + 5] = y_axis[2]


@wp.kernel
def decode_motionbricks_output_kernel(
    num_model_joints: wp.int32,
    num_pose_joints: wp.int32,
    parent_indices: wp.array(dtype=wp.int32),
    pose_to_model: wp.array(dtype=wp.int32),
    reference_local_poses: wp.array(dtype=wp.transform),
    alignment: wp.array(dtype=wp.transform),
    model_to_runtime: wp.transform,
    pred_global_motions: wp.array3d(dtype=wp.float32),
    pred_num_tokens: wp.array(dtype=wp.int32),
    output_local_poses: wp.array2d(dtype=wp.transform),
    output_foot_contacts: wp.array2d(dtype=wp.float32),
    output_frame_count: wp.array(dtype=wp.int32),
):
    """Decode the predicted motion into local joint transforms and foot contacts.

    It drops the boundary frames and writes the valid frame count. Foot contacts are
    clamped to 0 to 1. The alignment and model_to_runtime place the root in world space.
    """
    frame = wp.tid()
    raw_count = wp.clamp(
        pred_num_tokens[0] * MODEL_POSES_PER_TOKEN, 0, pred_global_motions.shape[1]
    )
    frame_count = wp.max(0, raw_count - 2 * MODEL_BOUNDARY_FRAMES)
    if frame == 0:
        output_frame_count[0] = frame_count
    if frame >= frame_count:
        return
    raw_frame = MODEL_BOUNDARY_FRAMES + frame
    first_contact = pred_global_motions.shape[2] - FOOT_CONTACT_COUNT
    for contact in range(FOOT_CONTACT_COUNT):
        output_foot_contacts[frame, contact] = wp.clamp(
            pred_global_motions[0, raw_frame, first_contact + contact], 0.0, 1.0
        )
    position_offset = 5
    rotation_offset = position_offset + (num_model_joints - 1) * 3
    hips_position = wp.vec3(
        pred_global_motions[0, raw_frame, 0],
        pred_global_motions[0, raw_frame, 1],
        pred_global_motions[0, raw_frame, 2],
    )
    hips_rotation = _read_rotation(pred_global_motions, raw_frame, rotation_offset)
    hips_transform = wp.mul(alignment[0], wp.transform(hips_position, hips_rotation))
    output_local_poses[frame, 0] = wp.mul(model_to_runtime, hips_transform)
    for joint in range(1, num_pose_joints):
        model_joint = pose_to_model[joint]
        if model_joint < 0:
            output_local_poses[frame, joint] = reference_local_poses[joint]
            continue
        predicted_rotation = _read_rotation(
            pred_global_motions, raw_frame, rotation_offset + model_joint * 6
        )
        parent_model_joint = pose_to_model[parent_indices[joint]]
        parent_rotation = _read_rotation(
            pred_global_motions, raw_frame, rotation_offset + parent_model_joint * 6
        )
        local_rotation = wp.mul(wp.quat_inverse(parent_rotation), predicted_rotation)
        output_local_poses[frame, joint] = wp.transform(
            reference_local_poses[joint].p, wp.normalize(local_rotation)
        )


@wp.kernel
def compute_endpoint_deltas_kernel(
    output_frame_count: wp.array(dtype=wp.int32),
    context_poses: wp.array2d(dtype=wp.transform),
    target_poses: wp.array2d(dtype=wp.transform),
    output_local_poses: wp.array2d(dtype=wp.transform),
    start_deltas: wp.array(dtype=wp.transform),
    end_deltas: wp.array(dtype=wp.transform),
):
    """Compute the endpoint correction of each joint.

    Start deltas move the first generated pose to the last context pose. End deltas move
    the last generated pose to the first target pose. The root moves in world space, so
    its deltas come from pose_utils.compute_root_endpoint_delta().
    """
    joint = wp.tid()
    frame_count = output_frame_count[0]
    if frame_count < 2:
        return
    last_frame = frame_count - 1
    last_context_pose = context_poses.shape[0] - 1
    if joint == 0:
        start_deltas[0] = pose_utils.compute_root_endpoint_delta(
            output_local_poses[0, 0], context_poses[last_context_pose, 0]
        )
        end_deltas[0] = pose_utils.compute_root_endpoint_delta(
            output_local_poses[last_frame, 0], target_poses[0, 0]
        )
        return
    start_deltas[joint] = pose_utils.compute_delta_transform(
        output_local_poses[0, joint], context_poses[last_context_pose, joint]
    )
    end_deltas[joint] = pose_utils.compute_delta_transform(
        output_local_poses[last_frame, joint], target_poses[0, joint]
    )


@wp.kernel
def apply_endpoint_deltas_kernel(
    output_frame_count: wp.array(dtype=wp.int32),
    start_deltas: wp.array(dtype=wp.transform),
    end_deltas: wp.array(dtype=wp.transform),
    output_local_poses: wp.array2d(dtype=wp.transform),
):
    """Apply the endpoint correction across the generated motion in place.

    The start correction fades out over the motion as the end correction fades in.
    """
    frame, joint = wp.tid()
    frame_count = output_frame_count[0]
    if frame_count < 2 or frame >= frame_count:
        return
    end_weight = wp.float32(frame) / wp.float32(frame_count - 1)
    start_weight = 1.0 - end_weight
    if joint == 0:
        output_local_poses[frame, 0] = pose_utils.apply_root_endpoint_correction(
            output_local_poses[frame, 0], start_deltas[0], end_deltas[0], end_weight
        )
        return
    corrected_pose = pose_utils.apply_delta_transform(
        output_local_poses[frame, joint], start_deltas[joint], start_weight
    )
    output_local_poses[frame, joint] = pose_utils.apply_delta_transform(
        corrected_pose, end_deltas[joint], end_weight
    )
