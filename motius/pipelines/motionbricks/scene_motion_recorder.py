# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Record playback poses for use as model context."""

from typing import Any

import numpy as np
import warp as wp

from motius.motion.runtime.motionbricks.pose_utils import fill_poses


@wp.kernel
def _shift_and_store_kernel(pose: wp.array(dtype=Any), history: wp.array2d(dtype=Any)):
    joint = wp.tid()
    for frame in range(history.shape[0] - 1):
        history[frame, joint] = history[frame + 1, joint]
    history[history.shape[0] - 1, joint] = pose[joint]


class MotionRecorder:
    """Store a fixed number of recent poses and their frame durations.

    poses holds the recorded poses, oldest first, one row per history slot. They keep
    the reference pose's layout: local joint transforms for SOMA, or joint coordinates
    for G1. delta_times holds the time from each pose to the next, in seconds. The
    latest pose's value is always zero. Unrecorded slots hold the pose from the last
    fill() with zero duration. Recording shifts out the oldest pose and adds the newest
    at the end.
    """

    def __init__(
        self,
        reference_pose: wp.array,
        capacity: int = 32,
        stream: wp.Stream | None = None,
    ) -> None:
        """Fill the history with the reference pose.

        Args:
            reference_pose: Pose that sets the layout and fills the history, such as a
                skeleton's reference local transforms or a robot's reference joint
                coordinates.
            capacity: Number of recent poses kept. It must be positive.
            stream: Stream for the work. None uses the current stream.
        """
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = capacity
        self.poses = wp.empty(
            (self.capacity, reference_pose.shape[0]),
            dtype=reference_pose.dtype,
            device=reference_pose.device,
        )
        self.delta_times = np.zeros(self.capacity, dtype=np.float32)
        self.fill(reference_pose, stream)

    def fill(self, pose: wp.array, stream: wp.Stream | None = None) -> None:
        """Fill every slot with one pose and zero durations, as if the character held it.

        Args:
            pose: Pose to store, with the reference pose's shape and dtype.
            stream: Stream for the work. None uses the current stream.
        """
        self._check_pose(pose)
        fill_poses(pose, self.poses, stream)
        self.delta_times.fill(0.0)

    def record(
        self, delta_time: float, pose: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Store a pose sampled delta_time seconds after the previous pose.

        Args:
            delta_time: Time since the previous pose, in seconds. It must be finite and
                positive.
            pose: Pose to store, as local joint transforms or joint coordinates, with the
                reference pose's shape and dtype.
            stream: Stream for the work. None uses the current stream.
        """
        dt = np.float32(delta_time)
        self._check_pose(pose)
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("delta_time must be finite and positive.")
        self.delta_times[-1] = dt
        self.delta_times[:-1] = self.delta_times[1:]
        self.delta_times[-1] = 0.0
        wp.launch(
            _shift_and_store_kernel,
            dim=self.poses.shape[1],
            inputs=[pose],
            outputs=[self.poses],
            stream=stream,
        )

    def _check_pose(self, pose: wp.array) -> None:
        if pose.dtype != self.poses.dtype or pose.shape != (self.poses.shape[1],):
            raise ValueError("pose must match the reference pose's shape and dtype.")
