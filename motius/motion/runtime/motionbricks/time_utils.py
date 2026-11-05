# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Map times to animation frames and to samples of recorded history."""

import numpy as np
import warp as wp


def history_sample_indices(
    delta_times: np.ndarray, num_samples: int, sample_rate: float
):
    """Find samples at a fixed rate, ending at the latest recorded pose.

    Returns the lower frame, upper frame, and blend of each sample. Samples
    before the available history clamp to its oldest pose.

    Args:
        delta_times: Time from each recorded pose to the next, in seconds. The last
            value is zero.
        num_samples: Number of samples to find.
        sample_rate: Sample rate in Hz.
    """
    pose_times = np.zeros_like(delta_times)
    for frame in range(len(pose_times) - 2, -1, -1):
        pose_times[frame] = pose_times[frame + 1] - delta_times[frame]
    lower_frames = np.empty(num_samples, dtype=np.int32)
    upper_frames = np.empty(num_samples, dtype=np.int32)
    blends = np.empty(num_samples, dtype=np.float32)
    first_sample = -(num_samples - 1) / sample_rate
    for sample in range(num_samples):
        sample_time = first_sample + sample / sample_rate
        upper = min(
            int(np.searchsorted(pose_times, sample_time, side="right")),
            len(pose_times) - 1,
        )
        lower = max(upper - 1, 0)
        frame_time = float(pose_times[upper] - pose_times[lower])
        if frame_time > 0.0:
            blend = (sample_time - float(pose_times[lower])) / frame_time
        else:
            lower = upper
            blend = 0.0
        lower_frames[sample] = lower
        upper_frames[sample] = upper
        blends[sample] = np.clip(blend, 0.0, 1.0)
    return (lower_frames, upper_frames, blends)


def frame_index_from_time(time, sample_rate, num_frames):
    """Return the frame at a time, the next frame, and the blend toward the next frame.

    Both frames are clamped to [0, num_frames - 1], so the last frame is its own next
    frame. The blend is in [0, 1]. A blend of 0 uses only the frame, and 1 uses only
    the next one.

    Args:
        time: Time in seconds.
        sample_rate: Sample rate in Hz.
        num_frames: Number of frames.
    """
    last_frame = wp.max(0, num_frames - 1)
    fractional = time * sample_rate
    frame = wp.clamp(int(wp.floor(fractional)), 0, last_frame)
    next_frame_blend = 0.0
    if frame < last_frame:
        next_frame_blend = wp.clamp(fractional - frame, 0.0, 1.0)
    return (frame, min(frame + 1, last_frame), next_frame_blend)
