# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Common scalar and transform math helpers."""

from __future__ import annotations

import math

import numpy as np
import warp as wp


@wp.func
def quadratic_fade_weight(value: wp.float32) -> wp.float32:
    """Fade from 1 to 0 with a quadratic ease in and out as value goes from 0 to 1."""
    t = wp.clamp(value, 0.0, 1.0)
    remaining = 1.0 - t
    return wp.where(t < 0.5, 1.0 - 2.0 * t * t, 2.0 * remaining * remaining)


@wp.func
def project_to_plane(transform: wp.transform, plane_normal: wp.vec3) -> wp.transform:
    """Project a transform onto the plane through the origin with this normal."""
    normal = wp.normalize(plane_normal)
    position = transform.p - normal * wp.dot(transform.p, normal)
    return wp.transform(position, wp.quat_twist(normal, transform.q))


def critical_spring_damper_exact(
    current: float, current_rate: float, goal: float, halflife: float, delta_time: float
) -> tuple[float, float]:
    """Advance a critically damped spring, and return its new value and rate.

    A spring within 1e-4 of the goal and nearly still snaps to the goal.

    Args:
        current: Current value.
        current_rate: Current rate of change, per second.
        goal: Value the spring moves toward.
        halflife: Half-life of the spring's response, in seconds. Smaller values reach
            the goal faster.
        delta_time: Time to advance, in seconds.
    """
    damping = 4.0 * math.log(2.0) / (halflife + 1e-05)
    decay_rate = damping / 2.0
    position_offset = current - goal
    velocity_offset = current_rate + position_offset * decay_rate
    decay = math.exp(-decay_rate * delta_time)
    next_current = decay * (position_offset + velocity_offset * delta_time) + goal
    next_rate = decay * (current_rate - velocity_offset * decay_rate * delta_time)
    if abs(next_current - goal) <= 0.0001 and abs(next_rate) <= 0.0001:
        return (goal, 0.0)
    return (next_current, next_rate)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to [-pi, pi).

    Args:
        angle: Angle in radians.
    """
    return (angle + math.pi) % math.tau - math.pi


def critical_spring_angle(
    current_angle: float,
    current_rate: float,
    desired_direction: np.ndarray,
    halflife: float,
    delta_time: float,
) -> tuple[float, float]:
    """Advance an angle spring toward a 2D direction by the shortest arc.

    Returns the new angle, wrapped to [-pi, pi), and its rate.

    Args:
        current_angle: Current angle in radians.
        current_rate: Current rate of change, in radians per second.
        desired_direction: 2D direction (x, y) to turn toward. Its length does not
            matter.
        halflife: Half-life of the spring's response, in seconds. Smaller values reach
            the goal faster.
        delta_time: Time to advance, in seconds.
    """
    desired_angle = math.atan2(desired_direction[1], desired_direction[0])
    goal_angle = current_angle + wrap_angle(desired_angle - current_angle)
    angle, rate = critical_spring_damper_exact(
        current_angle, current_rate, goal_angle, halflife, delta_time
    )
    return (wrap_angle(angle), rate)
