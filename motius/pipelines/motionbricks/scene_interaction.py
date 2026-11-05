# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Place the probe box ahead of a character to find Smart Primitive interaction boxes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from motius.pipelines.motionbricks.scene_smart_primitive import InteractionBox


@wp.kernel
def _write_interaction_box_kernel(
    segment_start: wp.vec3,
    segment_end: wp.vec3,
    width: wp.float32,
    height: wp.float32,
    output_box: wp.array(dtype=InteractionBox),
):
    direction = segment_end - segment_start
    depth = wp.length(direction)
    center = 0.5 * (segment_start + segment_end)
    center.z = 0.5 * height
    yaw = wp.atan2(direction.y, direction.x) - 0.5 * wp.pi
    box = InteractionBox()
    box.transform = wp.transform(
        center, wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), yaw)
    )
    box.half_extents = 0.5 * wp.vec3(width, depth, height)
    output_box[0] = box


@dataclass(frozen=True, slots=True)
class InteractionProbe:
    """How far ahead a character looks for Smart Primitive interaction boxes.

    Attributes:
        box_width: Width of the probe box in meters.
        box_min_depth: Smallest depth of the box in meters.
        box_height: Height of the box in meters while standing.
        near_distance: Distance from the character to the near end of the box, in meters.
        lookahead_time: Seconds of travel at the desired speed that the box reaches.
        extra_lookahead_min_speed: Speed in m/s above which the box reaches further.
        extra_lookahead_time: Extra seconds of travel for the speed above
            extra_lookahead_min_speed.
    """

    box_width: float
    box_min_depth: float
    box_height: float
    near_distance: float
    lookahead_time: float
    extra_lookahead_min_speed: float
    extra_lookahead_time: float

    def segment(
        self, ground_position: np.ndarray, direction: np.ndarray, speed: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the near and far ends of the probe on the ground."""
        reach = speed * self.lookahead_time
        reach += (
            max(0.0, speed - self.extra_lookahead_min_speed) * self.extra_lookahead_time
        )
        far = max(self.near_distance + self.box_min_depth, reach)
        return (
            ground_position + direction * self.near_distance,
            ground_position + direction * far,
        )

    def write_box(
        self,
        segment: tuple[np.ndarray, np.ndarray],
        height_scale: float,
        output_box: wp.array,
        stream: wp.Stream | None = None,
    ) -> None:
        """Write the box that covers the segment, standing on the ground."""
        wp.launch(
            _write_interaction_box_kernel,
            dim=1,
            inputs=[
                wp.vec3(*segment[0]),
                wp.vec3(*segment[1]),
                self.box_width,
                self.box_height * height_scale,
            ],
            outputs=[output_box],
            stream=stream,
        )


__all__ = ["InteractionProbe"]
