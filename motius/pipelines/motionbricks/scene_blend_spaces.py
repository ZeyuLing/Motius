# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Blend target poses with a 1D blend space or a 2D directional blend space."""

from __future__ import annotations

import math
from bisect import bisect_left
from enum import Enum

import warp as wp

from motius.motion.runtime.motionbricks.dof_utils import (
    ROOT_COORDINATE_COUNT,
    read_root_transform,
    write_root_transform,
)
from motius.motion.runtime.motionbricks.pose_utils import (
    blend_additive_accumulate_transform,
    blend_additive_end_transform,
    blend_transform,
)

_EPSILON = 1e-06


class BlendSpaceType(Enum):
    """The type of blend space used to compute target poses."""

    LINEAR_1D = "linear_1d"
    DIRECTIONAL_2D = "directional_2d"


@wp.struct
class BlendSample:
    """Blend targets and blend weights that BlendSpace.evaluate() selects.

    indices tells which added targets were selected. data_indices tells where
    each target's pose data is stored.

    A 2D space without a target at (0, 0) uses the average of all targets as its
    center. uniform_weight is each target's share of the center's weight.
    """

    indices: wp.vec3i
    weights: wp.vec3
    data_indices: wp.array(dtype=wp.int32)
    uniform_weight: float


def _value_components(value) -> tuple[float, float]:
    components = (float(value[0]), float(value[1]))
    if not all(math.isfinite(component) for component in components):
        raise ValueError("Blend space values must contain two finite components.")
    return components


class BlendSpace:
    """Select targets and weights from a 1D value or a 2D direction.

    Add targets with add_target(), call compile() after setup, then use
    evaluate() to get the targets and weights for each blend.

    A 1D space blends between neighboring values. A 2D directional space blends
    between neighboring directions and an optional center target at (0, 0).
    """

    def __init__(self, blend_space_type: BlendSpaceType, device=None) -> None:
        """Create an empty blend space.

        Args:
            blend_space_type: LINEAR_1D or DIRECTIONAL_2D.
            device: Device for new arrays.
        """
        self._type = BlendSpaceType(blend_space_type)
        self._device = wp.get_device(device)
        self._target_values: list[tuple[float, float]] = []
        self._target_data_indices: list[int] = []
        self._data_indices = wp.empty(0, dtype=wp.int32, device=self._device)
        self._compiled = False

    def add_target(self, value: wp.vec2, data_index: int) -> None:
        """Add a blend target with its value and its index in the source pose array.

        Args:
            value: Blend value of the target. A 1D space uses only x. A 2D space reads
                it as a direction, and (0, 0) marks the center target.
            data_index: Index of the target's poses in the source pose array. It must be
                non-negative.
        """
        value = _value_components(value)
        if data_index < 0:
            raise ValueError("data_index must be non-negative.")
        self._target_values.append(value)
        self._target_data_indices.append(data_index)
        self._compiled = False

    def compile(self) -> None:
        """Prepare the target lookup used by evaluate().

        Call this after adding the targets, and again if you add more.
        """
        self._compiled = False
        values = self._target_values
        if not values:
            raise ValueError("Cannot compile a blend space without targets.")
        if self._type is BlendSpaceType.LINEAR_1D:
            order = sorted(range(len(values)), key=lambda index: values[index][0])
            self._keys = tuple(values[index][0] for index in order)
            if len(set(self._keys)) != len(values):
                raise ValueError(
                    "Blend target values in a 1D blend space must be unique."
                )
            self._intervals = tuple(
                (
                    (a, b, values[a][0], 1.0 / (values[b][0] - values[a][0]))
                    for a, b in zip(order, order[1:])
                )
            )
        else:
            centers = [
                index for index, value in enumerate(values) if value == (0.0, 0.0)
            ]
            if len(centers) > 1:
                raise ValueError(
                    "A directional blend space can have only one center target."
                )
            self._center = centers[0] if centers else -1
            order = sorted(
                (index for index in range(len(values)) if index != self._center),
                key=lambda index: math.atan2(values[index][1], values[index][0]),
            )
            self._angles = tuple(
                math.atan2(values[index][1], values[index][0]) for index in order
            )
            self._sectors = []
            for upper, b in enumerate(order):
                a = order[upper - 1]
                gap = (self._angles[upper] - self._angles[upper - 1]) % math.tau
                if len(order) > 1 and gap < _EPSILON:
                    raise ValueError(
                        "Directional targets must have distinct directions."
                    )
                ax, ay = values[a]
                bx, by = values[b]
                determinant = ax * by - ay * bx
                if gap >= math.pi or determinant <= 0.0:
                    self._sectors.append(None)
                    continue
                inverse = (
                    by / determinant,
                    -bx / determinant,
                    -ay / determinant,
                    ax / determinant,
                )
                slots = (a, b, max(self._center, 0))
                weight_order = tuple(sorted(range(3), key=lambda slot: slots[slot]))
                self._sectors.append(
                    (inverse, tuple(slots[slot] for slot in weight_order), weight_order)
                )
        self._data_indices = wp.array(
            self._target_data_indices, dtype=wp.int32, device=self._device
        )
        self._compiled = True

    @property
    def type(self) -> BlendSpaceType:
        """Kind of blend space, 1D or 2D directional."""
        return self._type

    @property
    def target_count(self) -> int:
        """Number of added targets."""
        return len(self._target_values)

    @property
    def data_indices(self) -> wp.array:
        """Source pose indices, in the order the targets were added."""
        return self._data_indices

    @property
    def device(self):
        """Device of the blend space's arrays."""
        return self._device

    def _sample(
        self, indices=(0, 0, 0), weights=(0.0, 0.0, 0.0), uniform_weight=0.0
    ) -> BlendSample:
        sample = BlendSample()
        sample.indices = wp.vec3i(*indices)
        sample.weights = wp.vec3(*weights)
        sample.data_indices = self._data_indices
        sample.uniform_weight = uniform_weight
        return sample

    def evaluate(self, value: wp.vec2) -> BlendSample:
        """Select targets and compute their blend weights for the given input.

        1D targets are ordered from the lower value to the higher one. 2D targets
        follow the order they were added. Pass the result to a blend function, such as
        linear_1d_pose_blend(), to compute the final poses.

        Args:
            value: Blend value. It must be finite. A 1D space clamps x to the target
                range. A 2D space reads (x, y) as a direction and a distance from the
                center. Near zero, it selects the center target. Without a center
                target, every target then gets the same weight.
        """
        if not self._compiled:
            raise RuntimeError(
                "Cannot evaluate before compile(). Call it after adding blend targets."
            )
        x, y = _value_components(value)
        if self.target_count == 1:
            return self._sample(weights=(1.0, 0.0, 0.0))
        if self._type is BlendSpaceType.LINEAR_1D:
            x = min(max(x, self._keys[0]), self._keys[-1])
            interval = max(0, bisect_left(self._keys, x) - 1)
            a, b, start, inverse_width = self._intervals[interval]
            blend = (x - start) * inverse_width
            return self._sample((a, b, 0), (1.0 - blend, blend, 0.0))
        if abs(x) > _EPSILON or abs(y) > _EPSILON:
            if self._angles:
                upper = bisect_left(self._angles, math.atan2(y, x)) % len(self._angles)
                sector = self._sectors[upper]
                if sector is not None:
                    inverse, indices, order = sector
                    a = max(0.0, inverse[0] * x + inverse[1] * y)
                    b = max(0.0, inverse[2] * x + inverse[3] * y)
                    total = max(1.0, a + b)
                    a, b = (a / total, b / total)
                    center = max(0.0, 1.0 - a - b)
                    weights = (a, b, center if self._center >= 0 else 0.0)
                    return self._sample(
                        indices,
                        tuple(weights[slot] for slot in order),
                        center / self.target_count if self._center < 0 else 0.0,
                    )
        if self._center >= 0:
            return self._sample((self._center, 0, 0), (1.0, 0.0, 0.0))
        return self._sample(uniform_weight=1.0 / self.target_count)


@wp.func
def _contribution_count(sample: BlendSample):
    return wp.where(sample.uniform_weight > 0.0, sample.data_indices.shape[0], 3)


@wp.func
def _contribution(sample: BlendSample, slot: wp.int32):
    target = slot
    weight = sample.uniform_weight
    if sample.uniform_weight > 0.0:
        for selected in range(3):
            if sample.indices[selected] == slot:
                weight += sample.weights[selected]
    else:
        target = sample.indices[slot]
        weight = sample.weights[slot]
    return (sample.data_indices[target], weight)


@wp.kernel
def _prepare_directional_poses_kernel(
    poses: wp.array3d(dtype=wp.transform), reference_pose: wp.array(dtype=wp.transform)
):
    target, frame, joint = wp.tid()
    pose = poses[target, frame, joint]
    if wp.dot(reference_pose[joint].q, pose.q) < 0.0:
        poses[target, frame, joint] = wp.transform(pose.p, -pose.q)


def prepare_directional_poses(
    poses: wp.array, reference_pose: wp.array, stream: wp.Stream | None = None
) -> None:
    """Give target quaternions consistent signs before 2D directional blending.

    Each quaternion takes the sign that matches the reference pose's rotation for its
    joint. The poses are edited in place.

    Args:
        poses: Local joint transforms of every target, laid out as [target, frame, joint].
        reference_pose: Reference pose with one transform per joint.
        stream: Stream for the work. None uses the current stream.
    """
    if poses.ndim != 3 or reference_pose.shape != (poses.shape[2],):
        raise ValueError(
            "Expected [target, frame, joint] poses and one reference transform per joint."
        )
    wp.launch(
        _prepare_directional_poses_kernel,
        dim=poses.shape,
        inputs=[poses, reference_pose],
        device=poses.device,
        stream=stream,
    )


def _validate_blend_arrays(
    data: wp.array, sample: BlendSample, output: wp.array
) -> None:
    """Check that the source and output arrays have compatible shapes."""
    if data.ndim != 3 or output.ndim != 2:
        raise ValueError(
            "Blend data must be [target, frame, component] and output [frame, component]."
        )
    if output.shape[0] < data.shape[1] or output.shape[1] != data.shape[2]:
        raise ValueError(
            "Blend output must have frame capacity and matching components."
        )
    if sample.data_indices.size == 0:
        raise ValueError("Blending requires registered data indices.")


@wp.kernel
def _linear_1d_pose_blend_kernel(
    poses: wp.array3d(dtype=wp.transform),
    sample: BlendSample,
    output: wp.array2d(dtype=wp.transform),
):
    frame, joint = wp.tid()
    first = sample.data_indices[sample.indices[0]]
    second = sample.data_indices[sample.indices[1]]
    blend = sample.weights[1]
    result = poses[first, frame, joint]
    if blend >= 1.0:
        result = poses[second, frame, joint]
    elif blend > 0.0 and first != second:
        result = blend_transform(result, poses[second, frame, joint], blend)
    output[frame, joint] = result


@wp.kernel
def _directional_2d_pose_blend_kernel(
    poses: wp.array3d(dtype=wp.transform),
    sample: BlendSample,
    reference_pose: wp.array(dtype=wp.transform),
    output: wp.array2d(dtype=wp.transform),
):
    frame, joint = wp.tid()
    accumulated = wp.transform(wp.vec3(0.0), wp.quat(0.0, 0.0, 0.0, 0.0))
    weight_sum = wp.float32(0.0)
    for slot in range(_contribution_count(sample)):
        target, weight = _contribution(sample, slot)
        if weight > 0.0:
            accumulated = accumulated + poses[target, frame, joint] * weight
            weight_sum += weight
    reference = reference_pose[joint]
    missing = wp.max(0.0, 1.0 - weight_sum)
    output[frame, joint] = wp.transform(
        accumulated.p + reference.p * missing,
        wp.normalize(accumulated.q + reference.q * missing),
    )


def linear_1d_pose_blend(
    poses: wp.array,
    sample: BlendSample,
    output: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Blend the selected 1D targets into one sequence of local joint transforms.

    Args:
        poses: Local joint transforms of every target, laid out as [target, frame, joint].
        sample: Targets and weights from BlendSpace.evaluate().
        output: Array that receives the blended frames, laid out as [frame, joint]. It
            needs room for every source frame.
        stream: Stream for the work. None uses the current stream.
    """
    _validate_blend_arrays(poses, sample, output)
    wp.launch(
        _linear_1d_pose_blend_kernel,
        dim=(poses.shape[1], poses.shape[2]),
        inputs=[poses, sample],
        outputs=[output],
        device=output.device,
        stream=stream,
    )


def directional_2d_pose_blend(
    poses: wp.array,
    sample: BlendSample,
    reference_pose: wp.array,
    output: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Blend the selected 2D targets into one sequence of local joint transforms.

    Call prepare_directional_poses() on poses once during setup. When the weights add
    up to less than one, the reference pose supplies the rest.

    Args:
        poses: Local joint transforms of every target, laid out as [target, frame, joint].
        sample: Targets and weights from BlendSpace.evaluate().
        reference_pose: Reference pose with one transform per joint.
        output: Array that receives the blended frames, laid out as [frame, joint]. It
            needs room for every source frame.
        stream: Stream for the work. None uses the current stream.
    """
    _validate_blend_arrays(poses, sample, output)
    if reference_pose.shape != (poses.shape[2],):
        raise ValueError("reference_pose must contain one transform per joint.")
    wp.launch(
        _directional_2d_pose_blend_kernel,
        dim=(poses.shape[1], poses.shape[2]),
        inputs=[poses, sample, reference_pose],
        outputs=[output],
        device=output.device,
        stream=stream,
    )


@wp.kernel
def _linear_1d_dof_blend_kernel(
    joint_q: wp.array3d(dtype=wp.float32),
    sample: BlendSample,
    output: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    if coordinate == 0 or coordinate >= ROOT_COORDINATE_COUNT:
        first = sample.data_indices[sample.indices[0]]
        second = sample.data_indices[sample.indices[1]]
        blend = sample.weights[1]
        if coordinate == 0:
            root = read_root_transform(joint_q, first, frame)
            if blend >= 1.0:
                root = read_root_transform(joint_q, second, frame)
            elif blend > 0.0 and first != second:
                root = blend_transform(
                    root, read_root_transform(joint_q, second, frame), blend
                )
            write_root_transform(root, output, frame)
        else:
            value = joint_q[first, frame, coordinate]
            if blend >= 1.0:
                value = joint_q[second, frame, coordinate]
            elif blend > 0.0:
                value = wp.lerp(value, joint_q[second, frame, coordinate], blend)
            output[frame, coordinate] = value


@wp.kernel
def _directional_2d_dof_blend_kernel(
    joint_q: wp.array3d(dtype=wp.float32),
    sample: BlendSample,
    reference_joint_q: wp.array(dtype=wp.float32),
    output: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    if coordinate == 0:
        accumulated = wp.transform(wp.vec3(0.0), wp.quat(0.0, 0.0, 0.0, 0.0))
        weight_sum = wp.float32(0.0)
        for slot in range(_contribution_count(sample)):
            target, weight = _contribution(sample, slot)
            if weight > 0.0:
                accumulated = blend_additive_accumulate_transform(
                    accumulated, read_root_transform(joint_q, target, frame), weight
                )
                weight_sum += weight
        reference_root = read_root_transform(reference_joint_q)
        write_root_transform(
            blend_additive_end_transform(accumulated, reference_root, weight_sum),
            output,
            frame,
        )
    elif coordinate >= ROOT_COORDINATE_COUNT:
        value = wp.float32(0.0)
        weight_sum = wp.float32(0.0)
        for slot in range(_contribution_count(sample)):
            target, weight = _contribution(sample, slot)
            if weight > 0.0:
                value += joint_q[target, frame, coordinate] * weight
                weight_sum += weight
        output[frame, coordinate] = value + reference_joint_q[coordinate] * wp.max(
            0.0, 1.0 - weight_sum
        )


def linear_1d_dof_blend(
    joint_q: wp.array,
    sample: BlendSample,
    output: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Blend the selected 1D targets into one sequence of joint coordinates.

    Args:
        joint_q: Joint coordinates of every target, laid out as [target, frame,
            coordinate]. Each frame starts with the root position and XYZW quaternion.
        sample: Targets and weights from BlendSpace.evaluate().
        output: Array that receives the blended frames, laid out as [frame, coordinate].
            It needs room for every source frame.
        stream: Stream for the work. None uses the current stream.
    """
    _validate_blend_arrays(joint_q, sample, output)
    if joint_q.shape[2] < ROOT_COORDINATE_COUNT:
        raise ValueError(
            "joint_q must contain the root position and XYZW quaternion before the joint angles."
        )
    wp.launch(
        _linear_1d_dof_blend_kernel,
        dim=(joint_q.shape[1], joint_q.shape[2]),
        inputs=[joint_q, sample],
        outputs=[output],
        device=output.device,
        stream=stream,
    )


def directional_2d_dof_blend(
    joint_q: wp.array,
    sample: BlendSample,
    reference_joint_q: wp.array,
    output: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Blend the selected 2D targets into one sequence of joint coordinates.

    When the weights add up to less than one, reference_joint_q supplies the rest.

    Args:
        joint_q: Joint coordinates of every target, laid out as [target, frame,
            coordinate]. Each frame starts with the root position and XYZW quaternion.
        sample: Targets and weights from BlendSpace.evaluate().
        reference_joint_q: Joint coordinates of the reference pose.
        output: Array that receives the blended frames, laid out as [frame, coordinate].
            It needs room for every source frame.
        stream: Stream for the work. None uses the current stream.
    """
    _validate_blend_arrays(joint_q, sample, output)
    if joint_q.shape[2] < ROOT_COORDINATE_COUNT or reference_joint_q.shape != (
        joint_q.shape[2],
    ):
        raise ValueError(
            "reference_joint_q must hold the root position, XYZW quaternion, and joint angles."
        )
    wp.launch(
        _directional_2d_dof_blend_kernel,
        dim=(joint_q.shape[1], joint_q.shape[2]),
        inputs=[joint_q, sample, reference_joint_q],
        outputs=[output],
        device=output.device,
        stream=stream,
    )
