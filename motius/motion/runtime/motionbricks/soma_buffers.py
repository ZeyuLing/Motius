# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Context and target tokens for the SOMA model."""

import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
)
from motius.motion.runtime.motionbricks import pose_utils
from motius.motion.runtime.motionbricks.time_utils import history_sample_indices
from motius.pipelines.motionbricks.runtime_generation import write_token_poses
from motius.pipelines.motionbricks.scene_motion_recorder import MotionRecorder


@wp.kernel
def _sample_recorder_kernel(
    recorded_poses: wp.array2d(dtype=wp.transform),
    lower_frames: wp.vec4i,
    upper_frames: wp.vec4i,
    blends: wp.vec4,
    token_poses: wp.array2d(dtype=wp.transform),
):
    pose, joint = wp.tid()
    token_poses[pose, joint] = pose_utils.blend_transform(
        recorded_poses[lower_frames[pose], joint],
        recorded_poses[upper_frames[pose], joint],
        blends[pose],
    )


class PoseToken:
    """Hold up to four poses for model context or a target.

    poses holds the local joint transforms of all four poses, laid out as [pose, joint].
    num_poses counts the supplied poses. Padding repeats the last pose in the unused
    space.
    """

    def __init__(self, num_joints: int, *, device=None) -> None:
        """Allocate room for four poses.

        Args:
            num_joints: Joints per pose. It must be positive.
            device: Device for new arrays. None uses the current device.
        """
        if num_joints <= 0:
            raise ValueError("num_joints must be positive.")
        self.num_joints = num_joints
        self.poses = wp.empty(
            (MODEL_POSES_PER_TOKEN, self.num_joints), dtype=wp.transform, device=device
        )
        self._num_poses = 0

    @property
    def num_poses(self) -> int:
        """Number of supplied poses, from 1 to 4, or 0 before the first update."""
        return self._num_poses

    def update_from_poses(
        self, local_poses: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Copy one to four poses and pad the unused space with the last pose.

        The poses are in time order at the model's sample rate of 30 Hz. A context
        token ends at the current pose. A target token starts at the pose to reach.

        Args:
            local_poses: Local joint transforms of one to four poses, as a transform
                array of [pose, joint] or [joint].
            stream: Stream for the work. None uses the current stream.
        """
        self._num_poses = write_token_poses(local_poses, self.poses, stream)

    def update_from_recorder(
        self, motion_history: MotionRecorder, stream: wp.Stream | None = None
    ) -> None:
        """Sample four poses at 30 Hz, ending at the latest recorded pose.

        Args:
            motion_history: Recorder of played poses, with the token's joint count.
            stream: Stream for the work. None uses the current stream.
        """
        if motion_history.poses.shape[1] != self.num_joints:
            raise ValueError("History and token joint counts do not match.")
        lower_frames, upper_frames, blends = history_sample_indices(
            motion_history.delta_times, MODEL_POSES_PER_TOKEN, MODEL_SAMPLE_RATE
        )
        wp.launch(
            _sample_recorder_kernel,
            dim=self.poses.shape,
            inputs=[
                motion_history.poses,
                wp.vec4i(*lower_frames),
                wp.vec4i(*upper_frames),
                wp.vec4(*blends),
            ],
            outputs=[self.poses],
            device=self.poses.device,
            stream=stream,
        )
        self._num_poses = MODEL_POSES_PER_TOKEN
