# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Move and turn a Smart Primitive pivot to match the character's approach."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import warp as wp

from motius.motion.runtime.motionbricks.math_utils import wrap_angle

if TYPE_CHECKING:
    from motius.pipelines.motionbricks.scene_smart_primitive import CharacterProperties


class Axis(Enum):
    """Local axis of a pivot rule. NONE turns the rule off."""

    NONE = "none"
    X = "x"
    Y = "y"
    Z = "z"


class ConstraintType(Enum):
    """Limit of a pivot rule.

    Translation allows LINE_PROJECTION. Rotation allows CONE.
    """

    UNCONSTRAINED = "unconstrained"
    CONE = "cone"
    LINE_PROJECTION = "line_projection"


class ConeSide(Enum):
    """Where the rotation cone points, relative to the motion direction."""

    FORWARD = "forward"
    MIRRORED = "mirrored"
    BOTH = "both"


class OperationOrder(Enum):
    """Order in which a pivot applies its translation and rotation."""

    TRANSLATION_THEN_ROTATION = "translation_then_rotation"
    ROTATION_THEN_TRANSLATION = "rotation_then_translation"


_AXIS_VECTORS = {
    Axis.X: wp.vec3(1.0, 0.0, 0.0),
    Axis.Y: wp.vec3(0.0, 1.0, 0.0),
    Axis.Z: wp.vec3(0.0, 0.0, 1.0),
}


@dataclass
class Pivot:
    """Rules that move and turn a pivot node before its sequence plays.

    The rules measure from the pivot's authored world transform. Moving the pivot does
    not move its limits, so repeated alignments never drift. Axes are local to the
    authored transform, so tilting the node tilts its rules. Invalid settings raise
    ValueError.

    Attributes:
        motion_direction: Local direction that lines up with the approach. It is
            stored at unit length.
        operation_order: Whether translation or rotation comes first. Rotating first
            also turns the line the pivot slides along.
        translation_axis: Local axis the pivot slides along. NONE keeps its position.
        translation_constraint: UNCONSTRAINED or LINE_PROJECTION. LINE_PROJECTION keeps
            the pivot between the line offsets.
        line_min_offset: Start of the line, in meters from the authored position.
        line_max_offset: End of the line, in meters from the authored position. It must
            be at least line_min_offset.
        rotation_axis: Local axis the pivot turns around. NONE keeps its rotation.
        rotation_constraint: UNCONSTRAINED or CONE.
        cone_half_angle_degrees: Allowed turn to each side of the cone's center, from 0
            to 180 degrees.
        cone_side: FORWARD centers the cone on motion_direction. MIRRORED centers it on
            the opposite direction. BOTH allows either cone.
    """

    motion_direction: tuple[float, float, float] = (0.0, 1.0, 0.0)
    operation_order: OperationOrder = OperationOrder.TRANSLATION_THEN_ROTATION
    translation_axis: Axis = Axis.NONE
    translation_constraint: ConstraintType = ConstraintType.UNCONSTRAINED
    line_min_offset: float = -0.5
    line_max_offset: float = 0.5
    rotation_axis: Axis = Axis.NONE
    rotation_constraint: ConstraintType = ConstraintType.UNCONSTRAINED
    cone_half_angle_degrees: float = 45.0
    cone_side: ConeSide = ConeSide.FORWARD

    def __post_init__(self) -> None:
        self.set_motion_direction(self.motion_direction)
        self.validate()

    def validate(self) -> None:
        """Check the settings, and raise ValueError for any that are invalid."""
        if self.translation_constraint not in (
            ConstraintType.UNCONSTRAINED,
            ConstraintType.LINE_PROJECTION,
        ):
            raise ValueError(
                "Translation only supports Unconstrained or Line Projection."
            )
        if self.rotation_constraint not in (
            ConstraintType.UNCONSTRAINED,
            ConstraintType.CONE,
        ):
            raise ValueError("Rotation only supports Unconstrained or Cone.")
        if (
            not all(
                math.isfinite(offset)
                for offset in (self.line_min_offset, self.line_max_offset)
            )
            or self.line_min_offset > self.line_max_offset
        ):
            raise ValueError(
                "Line offsets must be finite, with Min Offset <= Max Offset."
            )
        if (
            not math.isfinite(self.cone_half_angle_degrees)
            or not 0.0 <= self.cone_half_angle_degrees <= 180.0
        ):
            raise ValueError("Cone Half Angle must be between 0 and 180 degrees.")

    def set_motion_direction(self, direction) -> None:
        """Store direction at unit length.

        Args:
            direction: Three finite numbers, not all zero. Invalid input raises
                ValueError and keeps the stored direction.
        """
        if len(direction) != 3:
            raise ValueError("Motion Direction must contain three numbers.")
        length = math.hypot(*direction)
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("Motion Direction must be finite and non-zero.")
        self.motion_direction = tuple(float(value / length) for value in direction)

    def line_endpoints(self, limits_transform: wp.transform) -> tuple[wp.vec3, wp.vec3]:
        """Return both ends of the line the pivot slides along, in world space.

        The pivot must have a translation axis.

        Args:
            limits_transform: Transform the limits measure from, normally the pivot's
                authored world transform.
        """
        direction = wp.transform_vector(
            limits_transform, _AXIS_VECTORS[self.translation_axis]
        )
        return (
            limits_transform.p + self.line_min_offset * direction,
            limits_transform.p + self.line_max_offset * direction,
        )

    def aligned_transform(
        self,
        authored_world_transform: wp.transform,
        current_world_transform: wp.transform,
        character: CharacterProperties,
    ) -> wp.transform:
        """Return the pivot's world transform aligned to the character's approach.

        An axis set to NONE keeps the current position or rotation.

        Args:
            authored_world_transform: The pivot's authored world transform. The limits
                measure from it.
            current_world_transform: The pivot's world transform before this alignment.
            character: The approaching character. Line projection uses its probe segment.
                The other rules use its position and desired velocity.
        """
        limits_transform = authored_world_transform
        position = current_world_transform.p
        rotation = current_world_transform.q
        if self.operation_order is OperationOrder.ROTATION_THEN_TRANSLATION:
            rotation = self._aligned_rotation(limits_transform, rotation, character)
            if self.rotation_axis is not Axis.NONE:
                limits_transform = wp.transform(limits_transform.p, rotation)
            position = self._aligned_position(limits_transform, position, character)
        else:
            position = self._aligned_position(limits_transform, position, character)
            rotation = self._aligned_rotation(limits_transform, rotation, character)
        return wp.transform(position, rotation)

    def _aligned_position(
        self,
        limits_transform: wp.transform,
        current_position: wp.vec3,
        character: CharacterProperties,
    ) -> wp.vec3:
        if self.translation_axis is Axis.NONE:
            return current_position
        if self.translation_constraint is ConstraintType.LINE_PROJECTION:
            line_start, line_end = self.line_endpoints(limits_transform)
            approach_start, approach_end = character.probe_segment
            return _closest_point_to_segment(
                line_start, line_end, wp.vec3(*approach_start), wp.vec3(*approach_end)
            )
        character_position = wp.vec3(*character.position)
        desired = wp.vec3(*character.desired_velocity)
        origin = limits_transform.p
        axis = wp.normalize(
            wp.transform_vector(limits_transform, _AXIS_VECTORS[self.translation_axis])
        )
        relative = character_position - origin
        nearest_offset = float(wp.dot(relative, axis))
        offset = nearest_offset
        speed = float(wp.length(desired))
        if math.isfinite(speed) and speed > 1e-06:
            ray = desired / speed
            cosine = float(wp.dot(axis, ray))
            denominator = 1.0 - cosine * cosine
            if denominator > 1e-06:
                offset = (
                    nearest_offset - cosine * float(wp.dot(relative, ray))
                ) / denominator
            if wp.dot(origin + offset * axis - character_position, ray) < 0.0:
                offset = nearest_offset
        return origin + offset * axis

    def _aligned_rotation(
        self,
        limits_transform: wp.transform,
        current_rotation: wp.quat,
        character: CharacterProperties,
    ) -> wp.quat:
        if self.rotation_axis is Axis.NONE:
            return current_rotation
        axis = wp.normalize(
            wp.transform_vector(limits_transform, _AXIS_VECTORS[self.rotation_axis])
        )
        motion = wp.transform_vector(limits_transform, wp.vec3(*self.motion_direction))
        motion = motion - wp.dot(motion, axis) * axis
        desired = wp.vec3(*character.desired_velocity)
        desired = desired - wp.dot(desired, axis) * axis
        for length in (wp.length(motion), wp.length(desired)):
            if not math.isfinite(length) or length <= 1e-06:
                return current_rotation
        angle = math.atan2(
            wp.dot(axis, wp.cross(motion, desired)), wp.dot(motion, desired)
        )
        if self.rotation_constraint is ConstraintType.CONE:
            half_angle = math.radians(self.cone_half_angle_degrees)
            centers = (
                (0.0, math.pi)
                if self.cone_side is ConeSide.BOTH
                else (math.pi if self.cone_side is ConeSide.MIRRORED else 0.0,)
            )
            candidates = []
            for center in centers:
                delta = wrap_angle(angle - center)
                clamped = max(-half_angle, min(half_angle, delta))
                candidates.append((abs(delta - clamped), center + clamped))
            angle = min(candidates, key=lambda candidate: candidate[0])[1]
        turn = wp.quat_from_axis_angle(axis, angle)
        return wp.mul(turn, limits_transform.q)


def _closest_point_on_segment(start: wp.vec3, end: wp.vec3, point: wp.vec3) -> wp.vec3:
    segment = end - start
    t = wp.dot(point - start, segment) / (wp.length_sq(segment) + 1e-06)
    return start + min(max(t, 0.0), 1.0) * segment


def _half_length_and_direction(start: wp.vec3, end: wp.vec3) -> tuple[float, wp.vec3]:
    vector = end - start
    length = wp.length(vector)
    return (0.5 * length, vector / length if length > 0.0 else vector)


def _closest_point_to_segment(
    start: wp.vec3, end: wp.vec3, other_start: wp.vec3, other_end: wp.vec3
) -> wp.vec3:
    """Return the point between start and end that is closest to the other segment."""
    half_length, direction = _half_length_and_direction(start, end)
    other_half_length, other_direction = _half_length_and_direction(
        other_start, other_end
    )
    center = start + half_length * direction
    other_center = other_start + other_half_length * other_direction
    offset = center - other_center
    cosine = wp.dot(direction, other_direction)
    along = wp.dot(direction, offset)
    other_along = wp.dot(other_direction, offset)
    t = (cosine * other_along - along) / (1.0 - cosine * cosine + 1e-06)
    other_t = other_along + cosine * t
    point = center + min(max(t, -half_length), half_length) * direction
    other_point = (
        other_center
        + min(max(other_t, -other_half_length), other_half_length) * other_direction
    )
    projected = _closest_point_on_segment(start, end, other_point)
    other_projected = _closest_point_on_segment(other_start, other_end, point)
    if wp.length_sq(other_point - projected) < wp.length_sq(point - other_projected):
        return projected
    return point


__all__ = ["Axis", "ConeSide", "ConstraintType", "OperationOrder", "Pivot"]
