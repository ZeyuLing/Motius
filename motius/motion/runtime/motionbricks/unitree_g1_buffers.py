# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Context and target tokens for the Unitree G1 model."""

import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
)
from motius.motion.runtime.motionbricks.dof_utils import (
    ROOT_COORDINATE_COUNT,
    read_root_transform,
    write_root_transform,
)
from motius.motion.runtime.motionbricks.pose_utils import blend_transform
from motius.motion.runtime.motionbricks.time_utils import history_sample_indices
from motius.pipelines.motionbricks.runtime_generation import write_token_poses
from motius.pipelines.motionbricks.scene_motion_recorder import MotionRecorder


@wp.kernel
def _sample_dof_recorder_kernel(
    joint_q: wp.array2d(dtype=wp.float32),
    lower_frames: wp.vec4i,
    upper_frames: wp.vec4i,
    blends: wp.vec4,
    output: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    lower, upper, blend = (lower_frames[frame], upper_frames[frame], blends[frame])
    if coordinate == 0:
        root = blend_transform(
            read_root_transform(joint_q, lower),
            read_root_transform(joint_q, upper),
            blend,
        )
        write_root_transform(root, output, frame)
    elif coordinate >= ROOT_COORDINATE_COUNT:
        output[frame, coordinate] = wp.lerp(
            joint_q[lower, coordinate], joint_q[upper, coordinate], blend
        )


class DOFToken:
    """Hold up to four poses for model context or a target.

    poses holds the joint coordinates of all four poses, laid out as [pose, coordinate].
    Each pose holds the root position (XYZ, meters), its rotation (XYZW quaternion), and
    the joint angles (radians). num_poses counts the supplied poses. Padding repeats the
    last pose in the unused space.
    """

    def __init__(self, num_dofs: int, *, device=None) -> None:
        """Allocate room for four poses.

        Args:
            num_dofs: Number of joint angles per pose, 29 for G1. It must be positive.
            device: Device for new arrays. None uses the current device.
        """
        if num_dofs <= 0:
            raise ValueError("num_dofs must be positive.")
        self.num_dofs = num_dofs
        self.num_coordinates = ROOT_COORDINATE_COUNT + self.num_dofs
        self.poses = wp.empty(
            (MODEL_POSES_PER_TOKEN, self.num_coordinates),
            dtype=wp.float32,
            device=device,
        )
        self._num_poses = 0

    @property
    def num_poses(self) -> int:
        """Number of supplied poses, from 1 to 4, or 0 before the first update."""
        return self._num_poses

    def update_from_poses(
        self, joint_q: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Copy one to four poses and pad the unused space with the last pose.

        The poses are in time order at the model's sample rate of 30 Hz. A context
        token ends at the current pose. A target token starts at the pose to reach.

        Args:
            joint_q: Joint coordinates of one to four poses, as a float32 array of
                [pose, coordinate] or [coordinate].
            stream: Stream for the work. None uses the current stream.
        """
        self._num_poses = write_token_poses(joint_q, self.poses, stream)

    def update_from_recorder(
        self, motion_history: MotionRecorder, stream: wp.Stream | None = None
    ) -> None:
        """Sample four poses at 30 Hz, ending at the latest recorded pose.

        Args:
            motion_history: Recorder of played poses, with the token's coordinate
                count.
            stream: Stream for the work. None uses the current stream.
        """
        if motion_history.poses.shape[1] != self.num_coordinates:
            raise ValueError("History and token coordinate counts do not match.")
        lower, upper, blends = history_sample_indices(
            motion_history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE
        )
        wp.launch(
            _sample_dof_recorder_kernel,
            dim=self.poses.shape,
            inputs=[
                motion_history.poses,
                wp.vec4i(*lower),
                wp.vec4i(*upper),
                wp.vec4(*blends),
            ],
            outputs=[self.poses],
            device=self.poses.device,
            stream=stream,
        )
        self._num_poses = MODEL_POSES_PER_TOKEN
