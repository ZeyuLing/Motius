# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Store animation frames as joint coordinates."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import warp as wp

from motius.motion.runtime.motionbricks.dof_utils import (
    ROOT_COORDINATE_COUNT,
    read_root_transform,
    write_root_transform,
)
from motius.motion.runtime.motionbricks.foot_contacts import (
    sample_foot_contacts,
    validate_foot_contacts,
)
from motius.motion.runtime.motionbricks.pose_utils import (
    apply_root_endpoint_correction,
    blend_transform,
    compute_root_endpoint_delta,
)
from motius.motion.runtime.motionbricks.time_utils import frame_index_from_time


@wp.kernel
def _loop_delta_kernel(
    joint_q: wp.array2d(dtype=wp.float32),
    last: wp.int32,
    root_delta: wp.array(dtype=wp.transform),
    angle_deltas: wp.array(dtype=wp.float32),
):
    coordinate = wp.tid()
    if coordinate == 0:
        root_delta[0] = compute_root_endpoint_delta(
            read_root_transform(joint_q, last), read_root_transform(joint_q, 0)
        )
    elif coordinate >= ROOT_COORDINATE_COUNT:
        angle_deltas[coordinate] = joint_q[0, coordinate] - joint_q[last, coordinate]


@wp.kernel
def _make_loopable_kernel(
    root_delta: wp.array(dtype=wp.transform),
    angle_deltas: wp.array(dtype=wp.float32),
    last: wp.int32,
    joint_q: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    weight = float(frame) / float(last)
    if coordinate == 0:
        root = apply_root_endpoint_correction(
            read_root_transform(joint_q, frame),
            wp.transform_identity(),
            root_delta[0],
            weight,
        )
        write_root_transform(root, joint_q, frame)
    elif coordinate >= ROOT_COORDINATE_COUNT:
        joint_q[frame, coordinate] += angle_deltas[coordinate] * weight


@wp.kernel
def _sample_dof_pose_kernel(
    joint_q: wp.array2d(dtype=wp.float32),
    frame: wp.int32,
    next_frame: wp.int32,
    blend: wp.float32,
    root_offset: wp.transform,
    output: wp.array(dtype=wp.float32),
):
    coordinate = wp.tid()
    if coordinate == 0:
        root = blend_transform(
            read_root_transform(joint_q, frame),
            read_root_transform(joint_q, next_frame),
            blend,
        )
        write_root_transform(root_offset * root, output)
    elif coordinate >= ROOT_COORDINATE_COUNT:
        output[coordinate] = wp.lerp(
            joint_q[frame, coordinate], joint_q[next_frame, coordinate], blend
        )


class DOFAnimationBuffer:
    """Store animation frames as joint coordinates at a fixed sample rate.

    Each frame stores root position (XYZ, meters), rotation (XYZW quaternion),
    and joint angles (radians), in joint_names order. Sampling blends positions
    and joint angles linearly. It uses spherical interpolation for the root
    rotation.
    """

    def __init__(
        self,
        joint_names: Sequence[str],
        sample_rate: float,
        poses: wp.array,
        *,
        num_frames: int | None = None,
        foot_contacts: wp.array | None = None,
    ) -> None:
        """Create an animation whose frames live in poses.

        Args:
            joint_names: Names of the joint angles, in storage order.
            sample_rate: Sample rate in Hz. It must be positive.
            poses: Float32 Warp array of [capacity, 7 + len(joint_names)] joint
                coordinates. It is used as is.
            num_frames: Number of frames in use. None uses the whole capacity.
            foot_contacts: Predicted foot contacts as a float32 [capacity, 4] array on
                the same device, or None for clips.
        """
        self.joint_names = tuple(joint_names)
        self.num_dofs = len(self.joint_names)
        self.num_coordinates = ROOT_COORDINATE_COUNT + self.num_dofs
        self.sample_rate = float(sample_rate)
        if self.sample_rate <= 0.0:
            raise ValueError("sample_rate must be positive.")
        self.poses = poses
        if (
            self.poses.dtype != wp.float32
            or self.poses.ndim != 2
            or self.poses.shape[1] != self.num_coordinates
        ):
            raise ValueError(
                f"poses must have shape [capacity, {self.num_coordinates}] and dtype=float32."
            )
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
        return self.poses[: self.num_frames, :ROOT_COORDINATE_COUNT].numpy().copy()

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
            output_pose: Float32 array that receives the joint coordinates of one pose.
                It can be a view into one robot's joint_q.
            root_offset: World transform applied to the root. The default leaves the
                root unchanged.
            stream: Stream for the work. None uses the current stream.
        """
        if self.num_frames == 0:
            raise ValueError("Cannot sample an empty animation buffer.")
        if output_pose.dtype != wp.float32 or output_pose.shape != (
            self.num_coordinates,
        ):
            raise ValueError(
                f"output_pose must be a float32 vector of {self.num_coordinates} coordinates."
            )
        frame, next_frame, blend = frame_index_from_time(
            time, self.sample_rate, self.num_frames
        )
        wp.launch(
            _sample_dof_pose_kernel,
            dim=self.num_coordinates,
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

    def copy(self) -> DOFAnimationBuffer:
        """Return a new animation with copies of the frames in use and their foot contacts."""
        foot_contacts = (
            None
            if self.foot_contacts is None
            else wp.clone(self.foot_contacts[: self.num_frames])
        )
        return DOFAnimationBuffer(
            self.joint_names,
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
        root_delta = wp.empty(1, dtype=wp.transform, device=self.device)
        angle_deltas = wp.empty(
            self.num_coordinates, dtype=wp.float32, device=self.device
        )
        wp.launch(
            _loop_delta_kernel,
            dim=self.num_coordinates,
            inputs=[self.poses, last],
            outputs=[root_delta, angle_deltas],
            device=self.device,
            stream=stream,
        )
        wp.launch(
            _make_loopable_kernel,
            dim=(self.num_frames, self.num_coordinates),
            inputs=[root_delta, angle_deltas, last],
            outputs=[self.poses],
            device=self.device,
            stream=stream,
        )
