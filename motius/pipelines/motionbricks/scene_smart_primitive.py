# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Smart Primitive assets, and the runtime instances that share them."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_POSES_PER_TOKEN,
    ModelRig,
)
from motius.motion.runtime.motionbricks import pose_utils as pose_utils
from motius.motion.runtime.motionbricks.dof_utils import (
    ROOT_COORDINATE_COUNT,
    read_root_transform,
    write_root_transform,
)
from motius.pipelines.motionbricks.scene_pivot import Pivot
from motius.pipelines.motionbricks.scene_target_pose import TargetPoseInferenceParams


@dataclass(frozen=True, slots=True)
class CharacterProperties:
    """Character values in world space, for Smart Primitive behaviors.

    Attributes:
        position: Character position on the ground, in meters.
        velocity: Actual velocity of the character, in m/s.
        facing: Direction the character faces on the ground, for custom behaviors.
        desired_velocity: Velocity the character aims for, in m/s.
        probe_segment: Near and far ends of the character's probe box on the
            ground.
    """

    position: np.ndarray
    velocity: np.ndarray
    facing: np.ndarray
    desired_velocity: np.ndarray
    probe_segment: tuple[np.ndarray, np.ndarray]


@dataclass(frozen=True, slots=True)
class SmartPrimitiveSequence:
    """Target nodes in playback order, with their approach settings.

    Attributes:
        name: Name of the sequence.
        node_indices: Target nodes in playback order.
        start_speed: Authored speed of the first target pose, in m/s.
        pivot_node_index: Node whose pivot aligns the sequence, or -1 without one.
    """

    name: str
    node_indices: tuple[int, ...]
    start_speed: float
    pivot_node_index: int = -1


@dataclass(frozen=True, slots=True)
class SmartPrimitiveData:
    """Sampled poses and metadata used to build a runtime asset.

    Node zero is the root node. Each parent comes before its children. Component and
    sequence references use these node indices.

    Attributes:
        rig: Rig of the sampled poses.
        node_local_transforms: Each node's transform relative to its parent, as 7
            values: position, then XYZW quaternion.
        node_parent_indices: Each node's parent index, or -1 for the root node.
        node_names: Display name of each node.
        target_node_indices: Node of each target.
        target_poses: Four poses per target, in target order. SOMA stores local joint
            transforms as [poses, joints, 7]. G1 stores joint coordinates as [poses,
            coordinates]. Both store the root transform first.
        target_inference_params: Generation settings of each target.
        box_node_indices: Node of each interaction box.
        box_half_extents: Half size of each interaction box along its node's axes, in
            meters.
        pivot_node_indices: Node of each pivot.
        pivots: Pivot rules, in the order of pivot_node_indices.
        sequences: Sequences in authored order. Their steps keep their authored order.
    """

    rig: ModelRig
    node_local_transforms: np.ndarray
    node_parent_indices: np.ndarray
    node_names: tuple[str, ...]
    target_node_indices: np.ndarray
    target_poses: np.ndarray
    target_inference_params: tuple[TargetPoseInferenceParams, ...]
    box_node_indices: np.ndarray
    box_half_extents: np.ndarray
    pivot_node_indices: np.ndarray
    pivots: tuple[Pivot, ...]
    sequences: tuple[SmartPrimitiveSequence, ...]


@wp.struct
class InteractionBox:
    """Oriented box for interaction boxes and the probe box.

    Attributes:
        transform: Transform of the box center.
        half_extents: Half size along each of the box's axes, in meters.
    """

    transform: wp.transform
    half_extents: wp.vec3


@wp.kernel
def _set_local_transform_kernel(
    node_index: wp.int32,
    transform: wp.transform,
    local_transforms: wp.array(dtype=wp.transform),
):
    local_transforms[node_index] = transform


@wp.kernel
def _set_world_transform_kernel(
    node_index: wp.int32,
    transform: wp.transform,
    parent_indices: wp.array(dtype=wp.int32),
    world_transforms: wp.array(dtype=wp.transform),
    local_transforms: wp.array(dtype=wp.transform),
):
    parent_index = parent_indices[node_index]
    if parent_index < 0:
        local_transforms[node_index] = transform
    else:
        local_transforms[node_index] = wp.mul(
            wp.transform_inverse(world_transforms[parent_index]), transform
        )


@wp.kernel
def _write_skeleton_target_poses_kernel(
    source_frame_offset: wp.int32,
    node_index: wp.int32,
    node_transforms: wp.array(dtype=wp.transform),
    target_poses: wp.array2d(dtype=wp.transform),
    output_poses: wp.array2d(dtype=wp.transform),
):
    frame, joint = wp.tid()
    transform = target_poses[source_frame_offset + frame, joint]
    if joint == 0:
        transform = wp.mul(node_transforms[node_index], transform)
    output_poses[frame, joint] = transform


@wp.kernel
def _write_dof_target_poses_kernel(
    source_frame_offset: wp.int32,
    node_index: wp.int32,
    node_transforms: wp.array(dtype=wp.transform),
    target_poses: wp.array2d(dtype=wp.float32),
    output_poses: wp.array2d(dtype=wp.float32),
):
    frame, coordinate = wp.tid()
    source_frame = source_frame_offset + frame
    if coordinate == 0:
        root = wp.mul(
            node_transforms[node_index], read_root_transform(target_poses, source_frame)
        )
        write_root_transform(root, output_poses, frame)
    elif coordinate >= ROOT_COORDINATE_COUNT:
        output_poses[frame, coordinate] = target_poses[source_frame, coordinate]


@wp.kernel
def _write_interaction_boxes_kernel(
    box_offset: wp.int32,
    box_node_indices: wp.array(dtype=wp.int32),
    node_transforms: wp.array(dtype=wp.transform),
    box_half_extents: wp.array(dtype=wp.vec3),
    output_boxes: wp.array(dtype=InteractionBox),
):
    output_index = wp.tid()
    box_index = box_offset + output_index
    box = InteractionBox()
    box.transform = node_transforms[box_node_indices[box_index]]
    box.half_extents = box_half_extents[box_index]
    output_boxes[output_index] = box


@wp.func
def _boxes_overlap(a: InteractionBox, b: InteractionBox) -> bool:
    """Test two oriented boxes on every separating axis. Touching boxes overlap."""
    rotation_a = wp.quat_to_matrix(a.transform.q)
    r = wp.mul(wp.transpose(rotation_a), wp.quat_to_matrix(b.transform.q))
    t = wp.mul(wp.transpose(rotation_a), b.transform.p - a.transform.p)
    ea = a.half_extents
    eb = b.half_extents
    abs_r = wp.mat33()
    for i in range(3):
        for j in range(3):
            abs_r[i, j] = wp.abs(r[i, j]) + 1e-06
    for i in range(3):
        if (
            wp.abs(t[i])
            > ea[i] + eb[0] * abs_r[i, 0] + eb[1] * abs_r[i, 1] + eb[2] * abs_r[i, 2]
        ):
            return False
    for j in range(3):
        distance = wp.abs(t[0] * r[0, j] + t[1] * r[1, j] + t[2] * r[2, j])
        if (
            distance
            > ea[0] * abs_r[0, j] + ea[1] * abs_r[1, j] + ea[2] * abs_r[2, j] + eb[j]
        ):
            return False
    for i in range(3):
        i1 = (i + 1) % 3
        i2 = (i + 2) % 3
        for j in range(3):
            j1 = (j + 1) % 3
            j2 = (j + 2) % 3
            radius_a = ea[i1] * abs_r[i2, j] + ea[i2] * abs_r[i1, j]
            radius_b = eb[j1] * abs_r[i, j2] + eb[j2] * abs_r[i, j1]
            if wp.abs(t[i2] * r[i1, j] - t[i1] * r[i2, j]) > radius_a + radius_b:
                return False
    return True


@wp.kernel
def _test_interaction_box_overlap_kernel(
    interaction_boxes: wp.array(dtype=InteractionBox),
    probe_box: wp.array(dtype=InteractionBox),
    overlap_results: wp.array(dtype=wp.int32),
):
    box_index = wp.tid()
    overlap_results[box_index] = wp.where(
        _boxes_overlap(interaction_boxes[box_index], probe_box[0]), 1, 0
    )


def compute_interaction_box_overlaps(
    interaction_boxes: wp.array,
    probe_box: wp.array,
    overlap_results: wp.array,
    stream: wp.Stream | None = None,
) -> None:
    """Test every interaction box against the character's probe box, without waiting.

    Touching boxes count as overlapping.

    Args:
        interaction_boxes: Interaction boxes in world space.
        probe_box: Array whose first element is the character's probe box.
        overlap_results: Array that receives 1 for each overlapping box and 0 for the
            others. It needs room for every interaction box.
        stream: Stream for the work. None uses the current stream.
    """
    if interaction_boxes.dtype != InteractionBox or interaction_boxes.ndim != 1:
        raise ValueError("interaction_boxes must be a 1D InteractionBox array.")
    if (
        probe_box.dtype != InteractionBox
        or probe_box.ndim != 1
        or probe_box.shape[0] < 1
    ):
        raise ValueError(
            "probe_box must be a 1D InteractionBox array with capacity for at least 1 box."
        )
    box_count = interaction_boxes.shape[0]
    if (
        overlap_results.dtype != wp.int32
        or overlap_results.ndim != 1
        or overlap_results.shape[0] < box_count
    ):
        raise ValueError(
            f"overlap_results must be a 1D wp.int32 array with capacity for at least {box_count} results."
        )
    if box_count == 0:
        return
    wp.launch(
        _test_interaction_box_overlap_kernel,
        dim=box_count,
        inputs=[interaction_boxes, probe_box],
        outputs=[overlap_results],
        device=interaction_boxes.device,
        stream=stream,
    )


class SmartPrimitiveAsset:
    """Immutable Smart Primitive data shared by runtime instances."""

    def __init__(self, data: SmartPrimitiveData, device=None) -> None:
        """Copy the sampled data into device arrays.

        Args:
            data: Sampled poses and metadata.
            device: Device for new arrays.
        """
        self.rig = data.rig
        self.device = wp.get_device(device)
        self.node_names = data.node_names
        self.node_indices_by_name = {
            name: index for index, name in enumerate(self.node_names)
        }
        self.node_local_transforms = wp.array(
            data.node_local_transforms, dtype=wp.transform, device=self.device
        )
        self.node_parent_indices = wp.array(
            data.node_parent_indices, dtype=wp.int32, device=self.device
        )
        self.target_poses = wp.array(
            data.target_poses,
            dtype=wp.transform if self.rig is ModelRig.SOMA else wp.float32,
            device=self.device,
        )
        self.target_node_indices = tuple(
            int(value) for value in data.target_node_indices
        )
        self.target_inference_params = data.target_inference_params
        self.target_index_by_node = {
            node_index: target_index
            for target_index, node_index in enumerate(self.target_node_indices)
        }
        self.pivot_node_indices = tuple(int(value) for value in data.pivot_node_indices)
        self.pivot_index_by_node = {
            node_index: index
            for index, node_index in enumerate(self.pivot_node_indices)
        }
        self.pivots = tuple(replace(pivot) for pivot in data.pivots)
        self.sequences = data.sequences
        self.sequence_indices_by_name = {
            sequence.name: index for index, sequence in enumerate(self.sequences)
        }
        self.max_sequence_length = max(
            (len(sequence.node_indices) for sequence in self.sequences), default=0
        )
        self.box_node_indices = tuple(int(value) for value in data.box_node_indices)
        self.box_node_indices_device = wp.array(
            data.box_node_indices, dtype=wp.int32, device=self.device
        )
        self.box_index_by_node = {
            node_index: box_index
            for box_index, node_index in enumerate(self.box_node_indices)
        }
        self.box_half_extents = wp.array(
            data.box_half_extents, dtype=wp.vec3, device=self.device
        )


class SmartPrimitive:
    """A runtime instance with its own hierarchy, backed by shared asset data.

    Node and sequence indices out of range raise IndexError.
    """

    def __init__(self, asset: SmartPrimitiveAsset) -> None:
        """Create an instance with its own copy of the node transforms.

        Args:
            asset: Data shared with the other instances of the same document.
        """
        self.asset = asset
        self._node_local_transforms = wp.clone(asset.node_local_transforms)
        self._node_world_transforms = wp.empty_like(self._node_local_transforms)
        self._authored_pivot_transforms: dict[int, wp.transform] | None = None
        self._world_transforms_dirty = True

    @property
    def device(self):
        """Device of the instance's arrays."""
        return self.asset.device

    @property
    def rig(self) -> ModelRig:
        """Rig of the asset's poses."""
        return self.asset.rig

    @property
    def node_count(self) -> int:
        """Number of nodes, including the root node."""
        return len(self.asset.node_names)

    @property
    def target_pose_count(self) -> int:
        """Number of nodes with a target pose."""
        return len(self.asset.target_node_indices)

    @property
    def sequence_count(self) -> int:
        """Number of authored sequences."""
        return len(self.asset.sequences)

    @property
    def max_sequence_length(self) -> int:
        """Most steps in any sequence, or 0 without sequences."""
        return self.asset.max_sequence_length

    @property
    def interaction_box_count(self) -> int:
        """Number of interaction boxes."""
        return len(self.asset.box_node_indices)

    @property
    def pivot_node_indices(self) -> tuple[int, ...]:
        """Nodes with a Pivot, independent of their display names."""
        return self.asset.pivot_node_indices

    def pivot(self, node_index: int) -> Pivot:
        """Return a copy of the node's pivot rules.

        Args:
            node_index: A node from pivot_node_indices.
        """
        self._validate_node_index(node_index)
        return replace(self.asset.pivots[self.asset.pivot_index_by_node[node_index]])

    @property
    def pose_size(self) -> int:
        """Values in one target pose: joints for SOMA, or joint coordinates for G1."""
        return self.asset.target_poses.shape[1]

    @property
    def node_local_transforms(self) -> wp.array:
        """Current local transform of each node. Change them with the setter methods."""
        return self._node_local_transforms

    @property
    def node_parent_indices(self) -> wp.array:
        """Each node's parent index, or -1 for the root node."""
        return self.asset.node_parent_indices

    def node_name(self, node_index: int) -> str:
        """Return the node's display name.

        Args:
            node_index: Index of the node.
        """
        self._validate_node_index(node_index)
        return self.asset.node_names[node_index]

    def node_index(self, name: str) -> int:
        """Return the node index for name, or -1 when it is absent.

        Args:
            name: Display name of the node.
        """
        return self.asset.node_indices_by_name.get(name, -1)

    def sequence_name(self, sequence_index: int) -> str:
        """Return the sequence's name.

        Args:
            sequence_index: Index of the sequence.
        """
        self._validate_sequence_index(sequence_index)
        return self.asset.sequences[sequence_index].name

    def sequence_index(self, name: str) -> int:
        """Return the sequence index for name, or -1 when it is absent.

        Args:
            name: Name of the sequence.
        """
        return self.asset.sequence_indices_by_name.get(name, -1)

    def sequence_length(self, sequence_index: int) -> int:
        """Return the number of steps in the sequence.

        Args:
            sequence_index: Index of the sequence.
        """
        self._validate_sequence_index(sequence_index)
        return len(self.asset.sequences[sequence_index].node_indices)

    def sequence_start_speed(self, sequence_index: int) -> float:
        """Return the authored speed of the sequence's first target pose, in m/s.

        Args:
            sequence_index: Index of the sequence.
        """
        self._validate_sequence_index(sequence_index)
        return self.asset.sequences[sequence_index].start_speed

    def sequence_pivot_node_index(self, sequence_index: int) -> int:
        """Return the sequence's pivot node, or -1 when it has none.

        Args:
            sequence_index: Index of the sequence.
        """
        self._validate_sequence_index(sequence_index)
        return self.asset.sequences[sequence_index].pivot_node_index

    def target_names_for_sequence(self, sequence_index: int) -> tuple[str, ...]:
        """Return the display names of the sequence's target nodes, in playback order.

        Args:
            sequence_index: Index of the sequence.
        """
        self._validate_sequence_index(sequence_index)
        return tuple(
            self.asset.node_names[node_index]
            for node_index in self.asset.sequences[sequence_index].node_indices
        )

    def has_target_pose(self, node_index: int) -> bool:
        """Return whether the node has a target pose.

        Args:
            node_index: Index of the node.
        """
        self._validate_node_index(node_index)
        return node_index in self.asset.target_index_by_node

    def has_interaction_box(self, node_index: int) -> bool:
        """Return whether the node has an interaction box.

        Args:
            node_index: Index of the node.
        """
        self._validate_node_index(node_index)
        return node_index in self.asset.box_index_by_node

    def set_node_local_transform(
        self, node_index: int, transform: wp.transform, stream: wp.Stream | None = None
    ) -> None:
        """Set the node's transform relative to its parent. Its children move with it.

        Args:
            node_index: Index of the node.
            transform: New transform relative to the parent.
            stream: Stream for the work. None uses the current stream.
        """
        self._validate_node_index(node_index)
        wp.launch(
            _set_local_transform_kernel,
            dim=1,
            inputs=[node_index, transform],
            outputs=[self._node_local_transforms],
            device=self.asset.device,
            stream=stream,
        )
        self._world_transforms_dirty = True

    def set_node_world_transform(
        self, node_index: int, transform: wp.transform, stream: wp.Stream | None = None
    ) -> None:
        """Place the node in world space. Its children move with it.

        Args:
            node_index: Index of the node.
            transform: New world transform.
            stream: Stream for the work. None uses the current stream.
        """
        self._validate_node_index(node_index)
        self._ensure_world_transforms(stream)
        wp.launch(
            _set_world_transform_kernel,
            dim=1,
            inputs=[
                node_index,
                transform,
                self.asset.node_parent_indices,
                self._node_world_transforms,
            ],
            outputs=[self._node_local_transforms],
            device=self.asset.device,
            stream=stream,
        )
        self._world_transforms_dirty = True

    def compute_world_transforms(
        self, output_transforms: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Compute every node's world transform into output_transforms.

        Args:
            output_transforms: Array that receives one world transform per node.
            stream: Stream for the work. None uses the current stream.
        """
        if (
            output_transforms.dtype != wp.transform
            or output_transforms.ndim != 1
            or output_transforms.shape[0] < self.node_count
        ):
            raise ValueError(
                f"output_transforms must be a 1D wp.transform array with room for {self.node_count} transforms."
            )
        self._ensure_world_transforms(stream)
        wp.copy(
            output_transforms,
            self._node_world_transforms,
            count=self.node_count,
            stream=stream,
        )

    def get_local_target_poses(
        self, node_index: int, output_poses: wp.array, stream: wp.Stream | None = None
    ) -> TargetPoseInferenceParams:
        """Write the node's four poses in its parent's space, and return their settings.

        Only the root moves by the node's local transform. Joint values are
        unchanged.

        Args:
            node_index: Node with a target pose.
            output_poses: Array that receives the four poses: local joint transforms for
                SOMA, or joint coordinates for G1.
            stream: Stream for the work. None uses the current stream.
        """
        self._validate_node_index(node_index)
        self._validate_target_pose_output(output_poses, "output_poses")
        return self._get_target_poses(
            node_index, output_poses, self._node_local_transforms, stream
        )

    def get_world_target_poses(
        self, node_index: int, output_poses: wp.array, stream: wp.Stream | None = None
    ) -> TargetPoseInferenceParams:
        """Write the node's four poses in world space, and return their settings.

        Only the root moves by the node's world transform. Joint values are
        unchanged.

        Args:
            node_index: Node with a target pose.
            output_poses: Array that receives the four poses: local joint transforms for
                SOMA, or joint coordinates for G1.
            stream: Stream for the work. None uses the current stream.
        """
        self._validate_node_index(node_index)
        self._validate_target_pose_output(output_poses, "output_poses")
        self._ensure_world_transforms(stream)
        return self._get_target_poses(
            node_index, output_poses, self._node_world_transforms, stream
        )

    def align_sequence_pivot(
        self,
        sequence_index: int,
        character: CharacterProperties,
        stream: wp.Stream | None = None,
    ) -> None:
        """Move and turn the sequence's pivot node to match the character's approach.

        Call it before reading the sequence's targets. A sequence without a pivot keeps
        its placement. The first call records the world transform of every pivot, and
        each alignment measures its limits from that record. Repeated alignments never
        drift, even when one pivot is a child of another. The call waits for stream to
        finish before it reads the node transforms.

        Args:
            sequence_index: Sequence whose pivot to align.
            character: The approaching character.
            stream: Stream for the work. None uses the current stream.
        """
        node_index = self.sequence_pivot_node_index(sequence_index)
        if node_index < 0:
            return
        self._ensure_world_transforms(stream)
        if stream is not None:
            wp.synchronize_stream(stream)
        transforms = self._node_world_transforms.numpy()
        if self._authored_pivot_transforms is None:
            self._authored_pivot_transforms = {
                node: wp.transform(*transforms[node])
                for node in self.pivot_node_indices
            }
        current = wp.transform(*transforms[node_index])
        rules = self.asset.pivots[self.asset.pivot_index_by_node[node_index]]
        self.set_node_world_transform(
            node_index,
            rules.aligned_transform(
                self._authored_pivot_transforms[node_index], current, character
            ),
            stream=stream,
        )

    def get_world_target_poses_for_sequence(
        self,
        sequence_index: int,
        output_target_poses: wp.array,
        stream: wp.Stream | None = None,
    ) -> tuple[TargetPoseInferenceParams, ...]:
        """Write one sequence in world space, and return the settings of each step.

        Args:
            sequence_index: Index of the sequence.
            output_target_poses: Array that receives four poses per step, in step order.
            stream: Stream for the work. None uses the current stream.
        """
        self._validate_sequence_index(sequence_index)
        node_indices = self.asset.sequences[sequence_index].node_indices
        required_frame_count = len(node_indices) * MODEL_POSES_PER_TOKEN
        self._validate_target_pose_output(
            output_target_poses, "output_target_poses", required_frame_count
        )
        self._ensure_world_transforms(stream)
        inference_params = []
        for target_index, node_index in enumerate(node_indices):
            frame_offset = target_index * MODEL_POSES_PER_TOKEN
            inference_params.append(
                self._get_target_poses(
                    node_index,
                    output_target_poses[
                        frame_offset : frame_offset + MODEL_POSES_PER_TOKEN
                    ],
                    self._node_world_transforms,
                    stream,
                )
            )
        return tuple(inference_params)

    def get_local_interaction_box(
        self, node_index: int, output_box: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Write the node's interaction box in its parent's space.

        Args:
            node_index: Node with an interaction box.
            output_box: Array whose first element receives the box.
            stream: Stream for the work. None uses the current stream.
        """
        self._get_interaction_box(
            node_index, output_box, self._node_local_transforms, stream
        )

    def get_world_interaction_box(
        self, node_index: int, output_box: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Write the node's interaction box in world space.

        Args:
            node_index: Node with an interaction box.
            output_box: Array whose first element receives the box.
            stream: Stream for the work. None uses the current stream.
        """
        self._ensure_world_transforms(stream)
        self._get_interaction_box(
            node_index, output_box, self._node_world_transforms, stream
        )

    def get_world_interaction_boxes(
        self, output_boxes: wp.array, stream: wp.Stream | None = None
    ) -> None:
        """Write every interaction box in world space, ready for overlap tests.

        Args:
            output_boxes: Array that receives the boxes in asset order. It needs room for
                interaction_box_count boxes.
            stream: Stream for the work. None uses the current stream.
        """
        box_count = self.interaction_box_count
        self._validate_box_output(output_boxes, "output_boxes", box_count)
        if box_count == 0:
            return
        self._ensure_world_transforms(stream)
        wp.launch(
            _write_interaction_boxes_kernel,
            dim=box_count,
            inputs=[
                0,
                self.asset.box_node_indices_device,
                self._node_world_transforms,
                self.asset.box_half_extents,
            ],
            outputs=[output_boxes],
            device=self.asset.device,
            stream=stream,
        )

    def _ensure_world_transforms(self, stream: wp.Stream | None) -> None:
        if self._world_transforms_dirty:
            pose_utils.compute_world_poses(
                self._node_local_transforms,
                self.asset.node_parent_indices,
                self._node_world_transforms,
                stream=stream,
            )
            self._world_transforms_dirty = False

    def _get_target_poses(
        self,
        node_index: int,
        output_poses: wp.array,
        node_transforms: wp.array,
        stream: wp.Stream | None,
    ) -> TargetPoseInferenceParams:
        target_index = self.asset.target_index_by_node[node_index]
        frame_offset = target_index * MODEL_POSES_PER_TOKEN
        kernel = (
            _write_skeleton_target_poses_kernel
            if self.rig is ModelRig.SOMA
            else _write_dof_target_poses_kernel
        )
        wp.launch(
            kernel,
            dim=(MODEL_POSES_PER_TOKEN, self.pose_size),
            inputs=[frame_offset, node_index, node_transforms, self.asset.target_poses],
            outputs=[output_poses],
            device=self.asset.device,
            stream=stream,
        )
        return self.asset.target_inference_params[target_index]

    def _get_interaction_box(
        self,
        node_index: int,
        output_box: wp.array,
        node_transforms: wp.array,
        stream: wp.Stream | None,
    ) -> None:
        self._validate_node_index(node_index)
        self._validate_box_output(output_box, "output_box", 1)
        box_index = self.asset.box_index_by_node[node_index]
        wp.launch(
            _write_interaction_boxes_kernel,
            dim=1,
            inputs=[
                box_index,
                self.asset.box_node_indices_device,
                node_transforms,
                self.asset.box_half_extents,
            ],
            outputs=[output_box],
            device=self.asset.device,
            stream=stream,
        )

    def _validate_node_index(self, node_index: int) -> None:
        if not 0 <= node_index < self.node_count:
            raise IndexError(f"Node index {node_index!r} is out of range.")

    def _validate_sequence_index(self, sequence_index: int) -> None:
        if not 0 <= sequence_index < self.sequence_count:
            raise IndexError(f"Sequence index {sequence_index!r} is out of range.")

    def _validate_target_pose_output(
        self, output_poses: wp.array, name: str, count: int = MODEL_POSES_PER_TOKEN
    ) -> None:
        if (
            output_poses.dtype != self.asset.target_poses.dtype
            or output_poses.ndim != 2
            or output_poses.shape[0] < count
            or (output_poses.shape[1] != self.pose_size)
        ):
            raise ValueError(
                f"{name} must be a 2D array of the target pose type with capacity for at least {count} poses and exactly {self.pose_size} values per pose."
            )

    def _validate_box_output(
        self, output_boxes: wp.array, name: str, count: int
    ) -> None:
        if (
            output_boxes.dtype != InteractionBox
            or output_boxes.ndim != 1
            or output_boxes.shape[0] < count
        ):
            raise ValueError(
                f"{name} must be a 1D InteractionBox array with capacity for at least {count} boxes."
            )
