# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Store animations as local joint transforms, and loop or remap them."""

from __future__ import annotations

import numpy as np
import warp as wp

from motius.motion.runtime.motionbricks import pose_utils, time_utils
from motius.motion.runtime.motionbricks.foot_contacts import (
    sample_foot_contacts,
    validate_foot_contacts,
)


@wp.kernel
def _remap_animation_kernel(
    source_local_poses: wp.array2d(dtype=wp.transform),
    joint_mapping: wp.array(dtype=wp.int32),
    destination_local_poses: wp.array2d(dtype=wp.transform),
):
    frame, source_joint = wp.tid()
    destination_joint = joint_mapping[source_joint]
    if destination_joint >= 0:
        destination_local_poses[frame, destination_joint] = source_local_poses[
            frame, source_joint
        ]


@wp.kernel
def _loop_delta_kernel(
    local_poses: wp.array2d(dtype=wp.transform),
    last: wp.int32,
    deltas: wp.array(dtype=wp.transform),
):
    joint = wp.tid()
    if joint == 0:
        deltas[0] = pose_utils.compute_root_endpoint_delta(
            local_poses[last, 0], local_poses[0, 0]
        )
    else:
        deltas[joint] = pose_utils.compute_delta_transform(
            local_poses[last, joint], local_poses[0, joint]
        )


@wp.kernel
def _make_loopable_kernel(
    deltas: wp.array(dtype=wp.transform),
    last: wp.int32,
    local_poses: wp.array2d(dtype=wp.transform),
):
    frame, joint = wp.tid()
    weight = wp.float32(frame) / wp.float32(last)
    if joint == 0:
        local_poses[frame, 0] = pose_utils.apply_root_endpoint_correction(
            local_poses[frame, 0], wp.transform_identity(), deltas[0], weight
        )
    else:
        local_poses[frame, joint] = pose_utils.apply_delta_transform(
            local_poses[frame, joint], deltas[joint], weight
        )


class AnimationBuffer:
    """Store animation frames as local joint transforms at a fixed sample rate.

    num_frames counts the frames in use, which can be fewer than the storage holds.
    foot_contacts is a [capacity, 4] array of predicted foot contacts for generated
    motion. It is None for clips.
    """

    def __init__(
        self,
        skeleton,
        sample_rate: float,
        poses: wp.array,
        *,
        num_frames: int | None = None,
        foot_contacts: wp.array | None = None,
    ) -> None:
        """Create an animation whose frames live in poses.

        Args:
            skeleton: Skeleton of the animation.
            sample_rate: Sample rate in Hz. It must be positive.
            poses: Warp array of [capacity, num_joints] local joint transforms. It is
                used as is.
            num_frames: Number of frames in use. None uses the whole capacity.
            foot_contacts: Predicted foot contacts as a float32 [capacity, 4] array on
                the same device, or None for clips.
        """
        self.skeleton = skeleton
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        self.poses = poses
        if (
            self.poses.dtype != wp.transform
            or self.poses.ndim != 2
            or self.poses.shape[1] != self.skeleton.num_joints
        ):
            raise ValueError("poses must have shape [capacity, num_joints].")
        validate_foot_contacts(foot_contacts, self.capacity, self.device)
        self.foot_contacts = foot_contacts
        self.num_frames = self.capacity if num_frames is None else num_frames

    @property
    def device(self):
        """Device of the frame storage."""
        return self.poses.device

    @property
    def capacity(self) -> int:
        """Number of frames the storage holds."""
        return self.poses.shape[0]

    @property
    def num_frames(self) -> int:
        """Number of frames in use, from 0 to capacity."""
        return self._num_frames

    @num_frames.setter
    def num_frames(self, frame_count: int) -> None:
        if not 0 <= frame_count <= self.capacity:
            raise ValueError(f"num_frames must be in [0, {self.capacity}].")
        self._num_frames = int(frame_count)

    @property
    def duration(self) -> float:
        """Time from the first to the last frame in use, in seconds."""
        return max(0, self.num_frames - 1) / self.sample_rate

    def root_transforms(self) -> np.ndarray:
        """Return the root of each frame in use on the host, shaped [num_frames, 7].

        Each row holds the position (XYZ), then the rotation (XYZW).
        """
        return self.poses[: self.num_frames, 0].numpy().copy()

    def sample(
        self,
        time: float,
        output_pose: wp.array,
        *,
        root_offset: wp.transform = wp.transform_identity(),
        stream: wp.Stream | None = None,
    ) -> None:
        """Write the pose sampled at time into output_pose.

        The two nearest frames are blended. Times outside the animation are clamped.

        Args:
            time: Sample time in seconds.
            output_pose: Array that receives one local joint transform per joint.
            root_offset: World transform applied to the root. The default leaves the
                root unchanged.
            stream: Stream for the work. None uses the current stream.
        """
        if self.num_frames == 0:
            raise ValueError("Cannot sample an empty animation buffer.")
        if (
            output_pose.dtype != wp.transform
            or output_pose.ndim != 1
            or output_pose.shape[0] != self.skeleton.num_joints
        ):
            raise ValueError("output_pose must contain one transform per joint.")
        frame, next_frame, blend = time_utils.frame_index_from_time(
            time, self.sample_rate, self.num_frames
        )
        wp.launch(
            pose_utils.blend_pose_joints_with_root_offset_kernel,
            dim=output_pose.shape[0],
            inputs=[self.poses, frame, next_frame, blend, root_offset],
            outputs=[output_pose],
            device=self.device,
            stream=stream,
        )

    def sample_foot_contacts(
        self, time: float, output: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Write the four foot contacts sampled at time into output.

        The two nearest frames are blended.

        Args:
            time: Sample time in seconds.
            output: Float32 array that receives the four foot contacts.
            stream: Stream for the work. None uses the current stream.
        """
        sample_foot_contacts(self, time, output, stream)

    def copy(self) -> AnimationBuffer:
        """Return a new animation with copies of the frames in use and their foot contacts."""
        foot_contacts = (
            None
            if self.foot_contacts is None
            else wp.clone(self.foot_contacts[: self.num_frames])
        )
        return AnimationBuffer(
            self.skeleton,
            self.sample_rate,
            wp.clone(self.poses[: self.num_frames]),
            foot_contacts=foot_contacts,
        )

    def make_loopable(self, stream: wp.Stream | None = None) -> None:
        """Adjust the frames in use so the first and last poses match.

        Each correction grows from zero at the first frame to the full difference at the
        last frame in use. The root position correction blends linearly, as in endpoint
        correction.

        Args:
            stream: Stream for the work. None uses the current stream.
        """
        if self.num_frames < 2:
            return
        last = self.num_frames - 1
        deltas = wp.empty(
            self.skeleton.num_joints, dtype=wp.transform, device=self.device
        )
        wp.launch(
            _loop_delta_kernel,
            dim=self.skeleton.num_joints,
            inputs=[self.poses, last],
            outputs=[deltas],
            device=self.device,
            stream=stream,
        )
        wp.launch(
            _make_loopable_kernel,
            dim=(self.num_frames, self.skeleton.num_joints),
            inputs=[deltas, last],
            outputs=[self.poses],
            device=self.device,
            stream=stream,
        )


def create_animation_buffer_for_skeleton(animation_buffer, new_skeleton):
    """Remap an animation to a new skeleton by joint name.

    Joints missing from the source keep the new skeleton's reference pose. Foot
    contacts are not copied. The same buffer comes back when it already uses
    new_skeleton.

    Args:
        animation_buffer: Animation to remap.
        new_skeleton: Skeleton of the new buffer.
    """
    if animation_buffer.skeleton is new_skeleton:
        return animation_buffer
    source_skeleton = animation_buffer.skeleton
    joint_mapping = wp.array(
        [new_skeleton.joint_index(name) for name in source_skeleton.joint_names],
        dtype=wp.int32,
    )
    output = AnimationBuffer(
        new_skeleton,
        animation_buffer.sample_rate,
        wp.empty(
            (animation_buffer.num_frames, new_skeleton.num_joints), dtype=wp.transform
        ),
    )
    if animation_buffer.num_frames == 0:
        return output
    pose_utils.fill_poses(new_skeleton.reference_local_transforms, output.poses)
    wp.launch(
        _remap_animation_kernel,
        dim=(animation_buffer.num_frames, source_skeleton.num_joints),
        inputs=[animation_buffer.poses, joint_mapping],
        outputs=[output.poses],
    )
    return output
