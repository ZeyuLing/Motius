# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Target pose generation settings, target pose IK, and locomotion target placement."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntFlag
from typing import TYPE_CHECKING

import warp as wp

from motius.configuration.motionbricks_runtime_contract import MODEL_SAMPLE_RATE
from motius.motion.runtime.motionbricks.dof_utils import (
    ROOT_COORDINATE_COUNT,
    read_root_transform,
    write_root_transform,
)
from motius.motion.runtime.motionbricks.ik import (
    TwoBoneIKChain,
    apply_two_bone_ik_chains_kernel,
)

if TYPE_CHECKING:
    from motius.motion.runtime.motionbricks.skeleton import Skeleton
MAX_TRANSITION_EARLY_BY_FRAMES = 20


class IKPin(IntFlag):
    """Hands and feet that a target pose can pin with IK."""

    NONE = 0
    LEFT_HAND = 1 << 0
    RIGHT_HAND = 1 << 1
    LEFT_FOOT = 1 << 2
    RIGHT_FOOT = 1 << 3
    ALL = LEFT_HAND | RIGHT_HAND | LEFT_FOOT | RIGHT_FOOT


IK_PINS = (IKPin.LEFT_HAND, IKPin.RIGHT_HAND, IKPin.LEFT_FOOT, IKPin.RIGHT_FOOT)


@dataclass(frozen=True, slots=True)
class TargetPoseInferenceParams:
    """Generation settings for reaching one target.

    Attributes:
        transition_early_by_frames: How many frames before the end of the generated
            motion the transition point sits.
        num_poses_to_use: How many of the target's four poses inform the model, from 1
            to 4.
        ik_pin_mask: IKPin flags for the hands and feet that target pose IK pins.
        allowed_token_counts: Token counts the model may choose from. They set the
            generated length, whatever the number of target poses. None keeps the
            counts of the generation config.
    """

    transition_early_by_frames: int
    num_poses_to_use: int
    ik_pin_mask: IKPin = IKPin.NONE
    allowed_token_counts: tuple[int, ...] | None = None


@dataclass(frozen=True, slots=True)
class TwoBoneIKChainConfig:
    """Resolved joint indices and hint offset for an IK chain with two bones.

    A pin other than one of IK_PINS raises ValueError.

    Attributes:
        pin: The one IK pin the chain serves.
        root_index: Joint index of the chain start.
        mid_index: Joint index of the middle joint, a child of the chain start.
        tip_index: Joint index of the tip, a child of the middle joint. The pin holds
            the tip in place.
        hint_offset_local: Point in the middle joint's local space that the chain bends
            toward, in meters.
    """

    pin: IKPin
    root_index: int
    mid_index: int
    tip_index: int
    hint_offset_local: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.pin not in IK_PINS:
            raise ValueError("pin must identify exactly one supported IK pin.")

    @classmethod
    def from_joint_names(
        cls,
        pin: IKPin,
        skeleton: Skeleton,
        root_joint: str,
        mid_joint: str,
        tip_joint: str,
        hint_offset_local: tuple[float, float, float],
    ) -> TwoBoneIKChainConfig:
        """Build a config from joint names, storing only their indices.

        A name missing from the skeleton raises ValueError.

        Args:
            pin: The one IK pin the chain serves.
            skeleton: Skeleton that holds the joints.
            root_joint: Name of the chain start.
            mid_joint: Name of the middle joint.
            tip_joint: Name of the tip.
            hint_offset_local: Point in the middle joint's local space that the chain
                bends toward, in meters.
        """
        joint_names = (root_joint, mid_joint, tip_joint)
        joint_indices = tuple(skeleton.joint_index(name) for name in joint_names)
        missing_names = [
            name for name, index in zip(joint_names, joint_indices) if index < 0
        ]
        if missing_names:
            names = ", ".join(repr(name) for name in missing_names)
            raise ValueError(
                f"{pin.name} IK chain joints are missing from the skeleton: {names}."
            )
        return cls(pin, *joint_indices, hint_offset_local)


_SOMA_IK_CHAINS = (
    (IKPin.LEFT_HAND, "LeftArm", "LeftForeArm", "LeftHand", (0.0, -0.25, 0.0)),
    (IKPin.RIGHT_HAND, "RightArm", "RightForeArm", "RightHand", (0.0, 0.25, 0.0)),
    (IKPin.LEFT_FOOT, "LeftLeg", "LeftShin", "LeftFoot", (0.0, -0.25, 0.0)),
    (IKPin.RIGHT_FOOT, "RightLeg", "RightShin", "RightFoot", (0.0, 0.25, 0.0)),
)


def create_soma_target_pose_ik_rig(
    skeleton: Skeleton,
) -> tuple[TwoBoneIKChainConfig, ...]:
    """Resolve the SOMA target pose IK chains and local hint offsets.

    Args:
        skeleton: SOMA skeleton that holds the arm and leg joints.
    """
    return tuple(
        (
            TwoBoneIKChainConfig.from_joint_names(pin, skeleton, root, mid, tip, hint)
            for pin, root, mid, tip, hint in _SOMA_IK_CHAINS
        )
    )


@wp.kernel
def _capture_pinned_transforms_kernel(
    chains: wp.array(dtype=TwoBoneIKChain),
    world_transforms: wp.array(dtype=wp.transform),
    pinned_transforms: wp.array(dtype=wp.transform),
):
    """Copy the world transform of each chain's tip as its pinned transform."""
    chain_index = wp.tid()
    pinned_transforms[chain_index] = world_transforms[chains[chain_index].tip_index]


class TargetPoseIKCorrector:
    """Apply IK pins to motion generated toward a target pose."""

    def __init__(
        self, skeleton: Skeleton, chains: Sequence[TwoBoneIKChainConfig]
    ) -> None:
        """Check the chains and allocate the solver arrays on the skeleton's device.

        No chains, or a chain whose joints do not form a parent chain, raise ValueError.

        Args:
            skeleton: Skeleton of the corrected poses.
            chains: One chain per pin, as create_soma_target_pose_ik_rig() returns.
                Chains cannot share joints.
        """
        if not chains:
            raise ValueError("Target pose IK chains cannot be empty.")
        self._skeleton = skeleton
        self._device = skeleton.parent_indices.device
        self._pin_mask = IKPin.NONE
        warp_chains = []
        for chain in chains:
            if (
                skeleton.joint_parent(chain.mid_index) != chain.root_index
                or skeleton.joint_parent(chain.tip_index) != chain.mid_index
            ):
                raise ValueError(
                    "An IK chain's mid joint must be a child of its chain start, and its tip must be a child of its mid joint."
                )
            warp_chain = TwoBoneIKChain()
            warp_chain.pin = int(chain.pin)
            warp_chain.root_index = chain.root_index
            warp_chain.mid_index = chain.mid_index
            warp_chain.tip_index = chain.tip_index
            warp_chain.hint_offset_local = wp.vec3(*chain.hint_offset_local)
            warp_chains.append(warp_chain)
        self._chains = wp.array(warp_chains, dtype=TwoBoneIKChain, device=self._device)
        self._pinned_transforms = wp.empty(
            len(self._chains), dtype=wp.transform, device=self._device
        )
        self._world_transforms = wp.empty(
            skeleton.num_joints, dtype=wp.transform, device=self._device
        )

    @property
    def pin_mask(self) -> IKPin:
        """Pins captured from the current target pose."""
        return self._pin_mask

    def capture_pins(
        self, target_pose: wp.array, pin_mask: IKPin, stream: wp.Stream | None = None
    ) -> None:
        """Capture the pinned hand and foot transforms from a target's local pose.

        A mask of IKPin.NONE clears the pins.

        Args:
            target_pose: Local joint transforms of the target pose, on the corrector's
                device.
            pin_mask: IKPin flags to pin. A pin without a chain has no effect.
            stream: Stream for the work. None uses the current stream.
        """
        if pin_mask == IKPin.NONE:
            self.clear_pins()
            return
        self._validate_pose(target_pose, "target_pose")
        self._skeleton.compute_world_transforms(
            target_pose, self._world_transforms, stream=stream
        )
        wp.launch(
            _capture_pinned_transforms_kernel,
            dim=len(self._chains),
            inputs=[self._chains, self._world_transforms],
            outputs=[self._pinned_transforms],
            device=self._device,
            stream=stream,
        )
        self._pin_mask = pin_mask

    def apply(
        self, current_pose: wp.array, weight: float, stream: wp.Stream | None = None
    ) -> None:
        """Correct the pinned chains of a sampled local pose in place.

        Nothing changes without captured pins or with a weight of zero or less.

        Args:
            current_pose: Local joint transforms to correct, on the corrector's device.
            weight: Blend from the sampled pose at 0 to the pinned pose at 1. Values
                above 1 count as 1.
            stream: Stream for the work. None uses the current stream.
        """
        if self._pin_mask == IKPin.NONE or weight <= 0.0:
            return
        self._validate_pose(current_pose, "current_pose")
        self._skeleton.compute_world_transforms(
            current_pose, self._world_transforms, stream=stream
        )
        wp.launch(
            apply_two_bone_ik_chains_kernel,
            dim=len(self._chains),
            inputs=[
                weight,
                int(self._pin_mask),
                self._chains,
                self._pinned_transforms,
                self._skeleton.parent_indices,
                self._world_transforms,
            ],
            outputs=[current_pose],
            device=self._device,
            stream=stream,
        )

    def clear_pins(self) -> None:
        """Disable all IK pins without reallocating device storage."""
        self._pin_mask = IKPin.NONE

    def _validate_pose(self, pose: wp.array, name: str) -> None:
        if (
            pose.dtype != wp.transform
            or pose.ndim != 1
            or pose.shape[0] != self._skeleton.num_joints
        ):
            raise ValueError(
                f"{name} must contain one wp.transform per skeleton joint."
            )
        if pose.device != self._device:
            raise ValueError(f"{name} must be stored on the corrector device.")


@wp.func
def _ground_speed_scale(
    start: wp.vec3, end: wp.vec3, pose_count: wp.int32, desired_speed: wp.float32
) -> wp.float32:
    """Return the scale that brings the mean ground speed from start to end to desired_speed.

    The poses are 30 Hz apart. Targets that barely move keep a scale of 1.
    """
    velocity = (end - start) * (MODEL_SAMPLE_RATE / wp.float32(pose_count - 1))
    source_speed = wp.length(wp.vec2(velocity[0], velocity[1]))
    if source_speed <= 1e-06:
        return 1.0
    return desired_speed / source_speed


@wp.func
def _scaled_ground_offset(
    position: wp.vec3, start: wp.vec3, scale: wp.float32
) -> wp.vec3:
    """Scale position's ground offset from start, and keep its height."""
    return wp.vec3(
        start[0] + (position[0] - start[0]) * scale,
        start[1] + (position[1] - start[1]) * scale,
        position[2],
    )


@wp.func
def _ground_placement(
    ground_position: wp.vec3, facing_direction: wp.vec3
) -> wp.transform:
    """Return the transform that moves a target to ground_position, turned to face facing_direction."""
    yaw = wp.atan2(facing_direction[1], facing_direction[0])
    return wp.transform(
        ground_position, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    )


@wp.kernel
def _adjust_speed_kernel(
    desired_speed: wp.float32, target_poses: wp.array2d(dtype=wp.transform)
):
    pose_count = target_poses.shape[0]
    start = target_poses[0, 0].p
    scale = _ground_speed_scale(
        start, target_poses[pose_count - 1, 0].p, pose_count, desired_speed
    )
    for frame in range(1, pose_count):
        root = target_poses[frame, 0]
        target_poses[frame, 0] = wp.transform(
            _scaled_ground_offset(root.p, start, scale), root.q
        )


def adjust_target_pose_speed(
    target_poses: wp.array, desired_speed: float, stream: wp.Stream | None = None
) -> None:
    """Scale the root's ground motion to a desired mean speed, with poses sampled at 30 Hz.

    One scale applies to each pose's ground offset from the first pose. The mean
    speed comes from the offset between the first and last poses. Height,
    rotations, and the other joints are kept. Targets that barely move stay
    unchanged.

    Args:
        target_poses: Local joint transforms of the target poses, laid out as [pose,
            joint]. It needs at least two poses. The poses are edited in place.
        desired_speed: Mean ground speed to reach, in m/s. It must be finite and
            non-negative.
        stream: Stream for the work. None uses the current stream.
    """
    if (
        target_poses.dtype != wp.transform
        or target_poses.ndim != 2
        or target_poses.shape[0] < 2
        or (target_poses.shape[1] == 0)
    ):
        raise ValueError("target_poses must contain at least two poses.")
    if not math.isfinite(desired_speed) or desired_speed < 0.0:
        raise ValueError("desired_speed must be finite and non-negative.")
    wp.launch(
        _adjust_speed_kernel,
        dim=1,
        inputs=[desired_speed],
        outputs=[target_poses],
        stream=stream,
    )


@wp.kernel
def _adjust_dof_speed_kernel(
    desired_speed: wp.float32, target_joint_q: wp.array2d(dtype=wp.float32)
):
    pose_count = target_joint_q.shape[0]
    start = read_root_transform(target_joint_q, 0).p
    end = read_root_transform(target_joint_q, pose_count - 1).p
    scale = _ground_speed_scale(start, end, pose_count, desired_speed)
    for frame in range(1, pose_count):
        position = _scaled_ground_offset(
            read_root_transform(target_joint_q, frame).p, start, scale
        )
        target_joint_q[frame, 0] = position[0]
        target_joint_q[frame, 1] = position[1]


def adjust_target_dof_speed(
    target_joint_q: wp.array, desired_speed: float, stream: wp.Stream | None = None
) -> None:
    """Scale the root's ground motion to a desired mean speed, with poses sampled at 30 Hz.

    One scale applies to each pose's ground offset from the first pose. The mean speed
    comes from the offset between the first and last poses. Root height, rotation,
    and joint angles are kept. Targets that barely move stay unchanged.

    Args:
        target_joint_q: Joint coordinates of the target poses, laid out as [pose,
            coordinate]. Each pose starts with the root position and XYZW quaternion.
            It needs at least two poses. The poses are edited in place.
        desired_speed: Mean ground speed to reach, in m/s. It must be finite and
            non-negative.
        stream: Stream for the work. None uses the current stream.
    """
    if (
        target_joint_q.dtype != wp.float32
        or target_joint_q.ndim != 2
        or target_joint_q.shape[0] < 2
        or (target_joint_q.shape[1] < ROOT_COORDINATE_COUNT)
    ):
        raise ValueError(
            "target_joint_q must contain at least two poses with root XYZ and XYZW values."
        )
    if not math.isfinite(desired_speed) or desired_speed < 0.0:
        raise ValueError("desired_speed must be finite and non-negative.")
    wp.launch(
        _adjust_dof_speed_kernel,
        dim=1,
        inputs=[desired_speed],
        outputs=[target_joint_q],
        device=target_joint_q.device,
        stream=stream,
    )


@wp.kernel
def _place_target_poses_kernel(
    ground_position: wp.vec3,
    facing_direction: wp.vec3,
    blended_poses: wp.array2d(dtype=wp.transform),
    target_poses: wp.array2d(dtype=wp.transform),
):
    frame, joint = wp.tid()
    pose = blended_poses[frame, joint]
    if joint == 0:
        pose = wp.mul(_ground_placement(ground_position, facing_direction), pose)
    target_poses[frame, joint] = pose


def place_target_poses(
    ground_position: wp.vec3,
    facing_direction: wp.vec3,
    blended_poses: wp.array,
    target_poses: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Place blended target poses at a ground position, turned to a facing direction.

    Only the root moves. The other joints keep their local transforms.

    Args:
        ground_position: Where the targets' origin goes, in world space.
        facing_direction: Ground direction the targets face.
        blended_poses: Local joint transforms of the blended targets, laid out as [pose,
            joint].
        target_poses: Array of the same shape that receives the placed poses.
        stream: Stream for the work. None uses the current stream.
    """
    wp.launch(
        _place_target_poses_kernel,
        dim=target_poses.shape,
        inputs=[ground_position, facing_direction, blended_poses],
        outputs=[target_poses],
        device=target_poses.device,
        stream=stream,
    )


@wp.kernel
def _place_target_dof_kernel(
    ground_position: wp.vec3,
    facing_direction: wp.vec3,
    blended_joint_q: wp.array2d(dtype=wp.float32),
    target_joint_q: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    if coordinate == 0:
        placement = _ground_placement(ground_position, facing_direction)
        write_root_transform(
            wp.mul(placement, read_root_transform(blended_joint_q, frame)),
            target_joint_q,
            frame,
        )
    elif coordinate >= ROOT_COORDINATE_COUNT:
        target_joint_q[frame, coordinate] = blended_joint_q[frame, coordinate]


def place_target_dof(
    ground_position: wp.vec3,
    facing_direction: wp.vec3,
    blended_joint_q: wp.array,
    target_joint_q: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Place blended target poses of joint coordinates at a ground position, turned to a facing direction.

    Only the root coordinates move. The joint angles are kept.

    Args:
        ground_position: Where the targets' origin goes, in world space.
        facing_direction: Ground direction the targets face.
        blended_joint_q: Joint coordinates of the blended targets, laid out as [pose,
            coordinate].
        target_joint_q: Array of the same shape that receives the placed poses.
        stream: Stream for the work. None uses the current stream.
    """
    wp.launch(
        _place_target_dof_kernel,
        dim=target_joint_q.shape,
        inputs=[ground_position, facing_direction, blended_joint_q],
        outputs=[target_joint_q],
        device=target_joint_q.device,
        stream=stream,
    )


__all__ = [
    "IK_PINS",
    "MAX_TRANSITION_EARLY_BY_FRAMES",
    "IKPin",
    "TargetPoseIKCorrector",
    "TargetPoseInferenceParams",
    "TwoBoneIKChainConfig",
    "adjust_target_dof_speed",
    "adjust_target_pose_speed",
    "create_soma_target_pose_ik_rig",
    "place_target_dof",
    "place_target_poses",
]
