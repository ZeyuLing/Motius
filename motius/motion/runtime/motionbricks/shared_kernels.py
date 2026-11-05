# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Conditioning masks and token padding that the SOMA and Unitree G1 engines share."""

from typing import Any

import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_CONTEXT_AND_TARGET_POSES,
    MODEL_POSES_PER_TOKEN,
)


@wp.kernel
def pad_token_poses_kernel(
    poses: wp.array2d(dtype=Any),
    num_poses: wp.int32,
    token_poses: wp.array2d(dtype=Any),
):
    """Copy the supplied poses into a token, repeating the last one in the unused rows."""
    pose, value = wp.tid()
    token_poses[pose, value] = poses[wp.min(pose, num_poses - 1), value]


@wp.kernel
def write_motionbricks_conditions_kernel(
    num_target_poses: wp.int32,
    allowed_token_bits: wp.int32,
    has_global_root_values: wp.array2d(dtype=wp.int32),
    has_local_root_values: wp.array2d(dtype=wp.int32),
    has_local_poses: wp.array2d(dtype=wp.int32),
    allowed_pred_num_tokens: wp.array2d(dtype=wp.int32),
):
    """Mark the conditioning slots in use, and write the allowed token counts as a mask.

    One thread runs per token count, from 6 to 16. The first eight threads also mark the
    four context pose slots and the four target pose slots. The context poses and the
    first num_target_poses target poses count as given. Local root values stay unmarked
    at the last context pose and the last target pose, because their derivatives are
    unknown there.
    """
    index = wp.tid()
    if index < MODEL_CONTEXT_AND_TARGET_POSES:
        value = wp.where(
            index < MODEL_POSES_PER_TOKEN + num_target_poses, wp.int32(1), wp.int32(0)
        )
        has_global_root_values[0, index] = value
        has_local_poses[0, index] = value
        has_local_root_values[0, index] = wp.where(
            index == MODEL_POSES_PER_TOKEN - 1
            or index == MODEL_POSES_PER_TOKEN + num_target_poses - 1,
            wp.int32(0),
            value,
        )
    allowed_pred_num_tokens[0, index] = allowed_token_bits >> index & 1
