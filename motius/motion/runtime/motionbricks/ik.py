# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Solve two-bone IK chains in Warp kernels."""

import warp as wp

_EPSILON = wp.constant(0.001)


@wp.struct
class TwoBoneIKChain:
    """IK chain with two bones.

    pin is the IKPin flag the chain serves. The joint indices name the chain start, the
    middle joint, and the tip. hint_offset_local is the point in the middle joint's
    local space that the chain bends toward.
    """

    pin: wp.int32
    root_index: wp.int32
    mid_index: wp.int32
    tip_index: wp.int32
    hint_offset_local: wp.vec3


@wp.func
def solve_two_bone_ik(
    in_weight: wp.float32,
    in_a_parent_world_tx: wp.transform,
    in_a_world_tx: wp.transform,
    in_b_world_tx: wp.transform,
    in_c_world_tx: wp.transform,
    in_t_world_tx: wp.transform,
    in_use_hint: wp.bool,
    in_hint_world: wp.vec3,
) -> tuple[wp.transform, wp.transform, wp.transform]:
    """Solve a two-bone IK chain so its tip reaches the pinned transform in_t_world_tx.

    It returns the solved world transforms of the chain start, the middle joint, and the
    tip. in_weight blends from the input pose at 0 to the solved pose at 1. With
    in_use_hint, the chain bends toward in_hint_world.
    """
    in_weight = wp.clamp(in_weight, 0.0, 1.0)
    a = in_a_world_tx.p
    b = in_b_world_tx.p
    c = in_c_world_tx.p
    t = wp.lerp(c, in_t_world_tx.p, in_weight)
    l_ab = wp.length(b - a)
    l_cb = wp.length(b - c)
    l_at = wp.clamp(wp.length(t - a), _EPSILON, l_ab + l_cb - _EPSILON)
    rot_axis = wp.normalize(
        wp.where(
            in_use_hint, wp.cross(c - a, in_hint_world - a), wp.cross(c - a, b - a)
        )
    )
    ac_ab_0 = wp.atan2(wp.dot(rot_axis, wp.cross(c - a, b - a)), wp.dot(c - a, b - a))
    ba_bc_0 = wp.atan2(wp.dot(rot_axis, wp.cross(a - b, c - b)), wp.dot(a - b, c - b))
    ac_ab_1 = wp.acos(
        wp.clamp(
            (l_ab * l_ab + l_at * l_at - l_cb * l_cb) / (2.0 * l_ab * l_at), -1.0, 1.0
        )
    )
    ba_bc_1 = wp.acos(
        wp.clamp(
            (l_ab * l_ab + l_cb * l_cb - l_at * l_at) / (2.0 * l_ab * l_cb), -1.0, 1.0
        )
    )
    rot_a_local = wp.quat_from_axis_angle(
        wp.quat_rotate_inv(in_a_world_tx.q, rot_axis), ac_ab_1 - ac_ab_0
    )
    rot_b_local = wp.quat_from_axis_angle(
        wp.quat_rotate_inv(in_b_world_tx.q, rot_axis), ba_bc_1 - ba_bc_0
    )
    a_local_tx = wp.mul(
        wp.mul(wp.transform_inverse(in_a_parent_world_tx), in_a_world_tx),
        wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_a_local),
    )
    a_world_tx = wp.mul(in_a_parent_world_tx, a_local_tx)
    ac_local = wp.quat_rotate_inv(a_world_tx.q, c - a)
    at_local = wp.quat_rotate_inv(a_world_tx.q, t - a)
    a_local_tx = wp.mul(
        a_local_tx,
        wp.transform(
            wp.vec3(0.0, 0.0, 0.0), wp.quat_between_vectors(ac_local, at_local)
        ),
    )
    a_world_tx = wp.mul(in_a_parent_world_tx, a_local_tx)
    b_local_tx = wp.mul(
        wp.mul(wp.transform_inverse(in_a_world_tx), in_b_world_tx),
        wp.transform(wp.vec3(0.0, 0.0, 0.0), rot_b_local),
    )
    b_world_tx = wp.mul(a_world_tx, b_local_tx)
    c_world_tx = wp.mul(
        b_world_tx, wp.mul(wp.transform_inverse(in_b_world_tx), in_c_world_tx)
    )
    return (
        wp.transform(a_world_tx.p, wp.normalize(a_world_tx.q)),
        wp.transform(b_world_tx.p, wp.normalize(b_world_tx.q)),
        wp.transform(
            c_world_tx.p,
            wp.normalize(wp.quat_slerp(c_world_tx.q, in_t_world_tx.q, in_weight)),
        ),
    )


@wp.kernel
def apply_two_bone_ik_chains_kernel(
    weight: wp.float32,
    pin_mask: wp.int32,
    chains: wp.array(dtype=TwoBoneIKChain),
    pinned_transforms: wp.array(dtype=wp.transform),
    parent_indices: wp.array(dtype=wp.int32),
    input_world_transforms: wp.array(dtype=wp.transform),
    output_local_transforms: wp.array(dtype=wp.transform),
):
    """Apply weighted IK corrections to one local pose.

    One thread runs per chain. Chains whose pin is not in pin_mask stay as they are.
    """
    chain_index = wp.tid()
    chain = chains[chain_index]
    if pin_mask & chain.pin == 0:
        return
    root_index = chain.root_index
    mid_index = chain.mid_index
    tip_index = chain.tip_index
    root_parent_index = parent_indices[root_index]
    root_parent_world = wp.transform_identity()
    if root_parent_index >= 0:
        root_parent_world = input_world_transforms[root_parent_index]
    root_world = input_world_transforms[root_index]
    mid_world = input_world_transforms[mid_index]
    tip_world = input_world_transforms[tip_index]
    hint_world = wp.transform_point(mid_world, chain.hint_offset_local)
    solved_root, solved_mid, solved_tip = solve_two_bone_ik(
        weight,
        root_parent_world,
        root_world,
        mid_world,
        tip_world,
        pinned_transforms[chain_index],
        True,
        hint_world,
    )
    output_local_transforms[root_index] = wp.mul(
        wp.transform_inverse(root_parent_world), solved_root
    )
    output_local_transforms[mid_index] = wp.mul(
        wp.transform_inverse(solved_root), solved_mid
    )
    output_local_transforms[tip_index] = wp.mul(
        wp.transform_inverse(solved_mid), solved_tip
    )
