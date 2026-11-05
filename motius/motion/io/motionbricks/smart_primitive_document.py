# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Shared Smart Primitive documents, clip loading, and runtime asset preparation."""

from __future__ import annotations

import colorsys
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MIN_TOKEN_COUNT,
    MODEL_POSES_PER_TOKEN,
    MODEL_SAMPLE_RATE,
    ModelRig,
)
from motius.motion.io.motionbricks import document_io as document_io
from motius.motion.io.motionbricks.document_io import (
    DocumentError,
    DocumentObject,
    DocumentType,
)
from motius.pipelines.motionbricks.scene_clip_registry import (
    ClipAsset,
    ClipLoadError,
    ClipRegistry,
)
from motius.pipelines.motionbricks.scene_pivot import (
    Axis,
    ConeSide,
    ConstraintType,
    OperationOrder,
    Pivot,
)
from motius.pipelines.motionbricks.scene_smart_primitive import (
    SmartPrimitive,
    SmartPrimitiveAsset,
    SmartPrimitiveData,
    SmartPrimitiveSequence,
)
from motius.pipelines.motionbricks.scene_target_pose import (
    MAX_TRANSITION_EARLY_BY_FRAMES,
    IKPin,
    TargetPoseInferenceParams,
)

_VELOCITY_EPSILON = 1e-06
MIN_TRAJECTORY_SCALE = 1.0
MAX_TRAJECTORY_SCALE = 4.0
_DEFAULT_TRAJECTORY_SCALE = 2.0
_DEFAULT_MIN_TOKENS = 6
_DEFAULT_MAX_TOKENS = 14
_TRAJECTORY_HISTORY_FRAMES = 10
_TRAJECTORY_FUTURE_FRAMES = 10
TRAJECTORY_POINT_COUNT = _TRAJECTORY_HISTORY_FRAMES + 1 + _TRAJECTORY_FUTURE_FRAMES
SMART_PRIMITIVE_DOCUMENT = DocumentType(
    kind="Smart Primitive",
    format="motionbricks.smart_primitive.authoring",
    schema_version=1,
    suffix=".smart_primitive.json",
    selected_field="selected_node_id",
)
_TARGET_POSE_COMPONENT_TYPE = "target_pose"
_INTERACTION_BOX_COMPONENT_TYPE = "interaction_box"
_PIVOT_COMPONENT_TYPE = "pivot"
_CONTENT_FIELDS = ("sequences", "root")
_NODE_FIELDS = (
    "id",
    "name",
    "local_transform",
    "euler_degrees",
    "visible",
    "translation_locks",
    "rotation_locks",
    "components",
    "children",
)
_TARGET_POSE_FIELDS = (
    "type",
    "id",
    "clip",
    "sample_time_seconds",
    "frame_index",
    "playback_mode",
    "hips_constrained",
    "num_poses_to_use",
    "transition_early_by_frames",
    "min_tokens",
    "max_tokens",
    "ik_pin_mask",
    "show_mesh",
    "show_skeletons",
    "show_velocity",
    "show_trajectory",
    "trajectory_scale",
    "color",
)
_INTERACTION_BOX_FIELDS = ("type", "id", "width", "depth", "height", "color")
_PIVOT_FIELDS = (
    "type",
    "id",
    "motion_direction",
    "operation_order",
    "translation_axis",
    "translation_constraint",
    "line_min_offset",
    "line_max_offset",
    "rotation_axis",
    "rotation_constraint",
    "cone_half_angle_degrees",
    "cone_side",
    "show_motion_direction",
    "show_translation_constraint",
    "show_rotation_constraint",
)
_SEQUENCE_FIELDS = ("id", "name", "node_ids", "pivot_node_id")


def _random_color() -> wp.vec3:
    return wp.vec3(*colorsys.hsv_to_rgb(random.random(), 0.65, 0.95))


class PlaybackMode(Enum):
    """How the editor shows a target pose's sample position: as a time or as a frame."""

    TIME = "time"
    FRAMES = "frames"


class InteractionBoxComponent:
    """An interaction box's size and color. Its node sets its placement."""

    def __init__(self, component_id: int) -> None:
        """Create a 1 m interaction box with a random color."""
        self.id = component_id
        self.width = 1.0
        self.depth = 1.0
        self.height = 1.0
        self.color = _random_color()

    def set_dimensions(self, width: float, depth: float, height: float) -> None:
        """Set the box size in meters. Values must be finite and non-negative."""
        dimensions = (float(width), float(depth), float(height))
        if any(not math.isfinite(value) or value < 0.0 for value in dimensions):
            raise ValueError(
                "Interaction Box dimensions must be non-negative and finite."
            )
        self.width, self.depth, self.height = dimensions


class TargetPoseComponent:
    """A target pose's settings and the clip samples used to build it.

    Poses use the layout of reference_pose, either local joint transforms or
    joint coordinates, with the root first. up_index names the clip axis that
    holds height.
    """

    def __init__(
        self, component_id: int, reference_pose: wp.array, up_index: int
    ) -> None:
        """Create a target pose without a clip, using all four poses and 6 to 14 tokens."""
        self.id = component_id
        self.clip: ClipAsset | None = None
        self.num_poses_to_use = MODEL_POSES_PER_TOKEN
        self.transition_early_by_frames = 0
        self.min_tokens = _DEFAULT_MIN_TOKENS
        self.max_tokens = _DEFAULT_MAX_TOKENS
        self.ik_pin_mask = IKPin.NONE
        self.show_mesh = True
        self.show_velocity = False
        self.show_trajectory = False
        self.show_skeletons = False
        self.trajectory_scale = _DEFAULT_TRAJECTORY_SCALE
        self.sample_time = 0.0
        self.playback_mode = PlaybackMode.TIME
        self.hips_constrained = False
        self.color = _random_color()
        self.set_reference_pose(reference_pose, up_index)

    def set_reference_pose(self, reference_pose: wp.array, up_index: int) -> None:
        """Allocate pose buffers that match reference_pose, then resample."""
        self.reference_pose = reference_pose
        self._up_index = up_index
        self.sampled_poses = wp.empty(
            (MODEL_POSES_PER_TOKEN, *reference_pose.shape),
            dtype=reference_pose.dtype,
            device=reference_pose.device,
        )
        self.pose = self.sampled_poses[0]
        self._trajectory_poses = wp.empty(
            (TRAJECTORY_POINT_COUNT, *reference_pose.shape),
            dtype=reference_pose.dtype,
            device=reference_pose.device,
        )
        self._trajectory_transforms = wp.empty(
            TRAJECTORY_POINT_COUNT, dtype=wp.transform, device=reference_pose.device
        )
        self._sample()

    @property
    def frame(self) -> int:
        """Clip frame nearest the sample time, or 0 without a clip."""
        if self.clip is None:
            return 0
        return min(
            round(self.sample_time * self.clip.animation.sample_rate),
            self.clip.last_frame,
        )

    @property
    def duration(self) -> float:
        """Clip duration in seconds, or 0 without a clip."""
        if self.clip is None:
            return 0.0
        return self.clip.duration

    @property
    def velocity(self) -> wp.vec3:
        """Root velocity between the first two sampled poses, in m/s.

        It is zero when only one pose is used.
        """
        if self.num_poses_to_use == 1:
            return wp.vec3(0.0)
        return self._sampled_velocity

    @property
    def velocity_magnitude(self) -> float:
        """Length of velocity, in m/s."""
        return float(wp.length(self.velocity))

    @property
    def has_velocity(self) -> bool:
        """Whether velocity is above a small threshold."""
        return self.velocity_magnitude > _VELOCITY_EPSILON

    @property
    def has_trajectory(self) -> bool:
        """Whether a root trajectory is available, which needs a clip."""
        return self.clip is not None

    @property
    def trajectory_transforms(self) -> wp.array:
        """Sample the preview path on demand, reusing it until the target changes."""
        if self._trajectory_dirty:
            self._update_trajectory_samples()
            self._trajectory_dirty = False
        return self._trajectory_transforms

    @property
    def skeleton_count(self) -> int:
        """Number of preview skeletons: the poses to use, or 1 without a clip."""
        if self.clip is None:
            return 1
        return self.num_poses_to_use

    def set_clip(self, clip: ClipAsset) -> None:
        """Use a clip and sample it from its start."""
        self.clip = clip
        self.sample_time = 0.0
        self._sample()

    def restore_sample_state(
        self,
        clip: ClipAsset | None,
        playback_mode: PlaybackMode,
        sample_time: float,
        frame_index: int,
        hips_constrained: bool,
    ) -> None:
        """Restore saved clip settings, then resample the target pose once."""
        self.clip = clip
        self.playback_mode = playback_mode
        self.hips_constrained = hips_constrained
        self.sample_time = (
            frame_index / clip.animation.sample_rate
            if clip is not None and playback_mode is PlaybackMode.FRAMES
            else sample_time
        )
        self._sample()

    def set_time(self, sample_time: float) -> None:
        """Sample the clip at sample_time (in seconds), clamped to its duration."""
        self.sample_time = min(max(float(sample_time), 0.0), self.duration)
        self._sample()

    def constrain_hips(self, enabled: bool) -> None:
        """Turn the root constraint on or off, then resample the clip."""
        enabled = bool(enabled)
        if enabled == self.hips_constrained:
            return
        self.hips_constrained = enabled
        self._sample()

    def set_num_poses_to_use(self, num_poses: int) -> None:
        """Set the poses to use, clamped to 1 to 4."""
        self.num_poses_to_use = min(max(int(num_poses), 1), MODEL_POSES_PER_TOKEN)

    def set_token_range(self, min_tokens: int, max_tokens: int) -> None:
        """Set the token counts the model may choose, clamped to 6 to 16.

        The maximum never drops below the minimum.
        """
        self.min_tokens = min(
            max(int(min_tokens), MODEL_MIN_TOKEN_COUNT), MODEL_MAX_TOKEN_COUNT
        )
        self.max_tokens = min(
            max(int(max_tokens), self.min_tokens), MODEL_MAX_TOKEN_COUNT
        )

    def set_ik_pin_enabled(self, pin: IKPin, enabled: bool) -> None:
        """Pin or unpin one hand or foot for this target pose."""
        pin = IKPin(pin)
        if enabled:
            self.ik_pin_mask |= pin
        else:
            self.ik_pin_mask &= ~pin

    def set_trajectory_scale(self, scale: float) -> None:
        """Set the spacing of the trajectory preview points, in frames at the model's sample rate.

        It must be MIN_TRAJECTORY_SCALE to MAX_TRAJECTORY_SCALE.
        """
        scale = float(scale)
        if (
            not math.isfinite(scale)
            or scale < MIN_TRAJECTORY_SCALE
            or scale > MAX_TRAJECTORY_SCALE
        ):
            raise ValueError(
                f"Trajectory scale must be finite, from {MIN_TRAJECTORY_SCALE:.1f} to {MAX_TRAJECTORY_SCALE:.1f}."
            )
        if scale == self.trajectory_scale:
            return
        self.trajectory_scale = scale
        self._trajectory_dirty = True

    def _sample(self) -> None:
        """Sample four 30 Hz poses from sample_time."""
        self._root_offset = wp.transform_identity()
        if self.clip is None:
            for frame in range(MODEL_POSES_PER_TOKEN):
                wp.copy(self.sampled_poses[frame], self.reference_pose)
        else:
            animation = self.clip.animation
            if self.hips_constrained:
                animation.sample(self.sample_time, self.pose)
                offset = -self.pose.numpy().reshape(-1)[:3]
                offset[self._up_index] = 0.0
                self._root_offset = wp.transform(wp.vec3(*offset), wp.quat_identity())
            for frame in range(MODEL_POSES_PER_TOKEN):
                animation.sample(
                    self.sample_time + frame / MODEL_SAMPLE_RATE,
                    self.sampled_poses[frame],
                    root_offset=self._root_offset,
                )
        roots = self.sampled_poses.numpy().reshape(MODEL_POSES_PER_TOKEN, -1)[:, :3]
        self.hips_position = wp.vec3(*roots[0])
        self._sampled_velocity = (
            wp.vec3(*roots[1]) - wp.vec3(*roots[0])
        ) * MODEL_SAMPLE_RATE
        self._trajectory_dirty = True

    def _update_trajectory_samples(self) -> None:
        if self.clip is None:
            return
        interval = self.trajectory_scale / MODEL_SAMPLE_RATE
        for point in range(TRAJECTORY_POINT_COUNT):
            self.clip.animation.sample(
                self.sample_time + (point - _TRAJECTORY_HISTORY_FRAMES) * interval,
                self._trajectory_poses[point],
                root_offset=self._root_offset,
            )
        roots = self._trajectory_poses.numpy().reshape(TRAJECTORY_POINT_COUNT, -1)[
            :, :7
        ]
        self._trajectory_transforms.assign(np.ascontiguousarray(roots))


class PivotComponent:
    """Pivot rules and preview visibility for a hierarchy node."""

    def __init__(self, component_id: int) -> None:
        """Create a pivot with default rules and every preview shown."""
        self.id = component_id
        self.pivot = Pivot()
        self.show_motion_direction = True
        self.show_translation_constraint = True
        self.show_rotation_constraint = True


class SceneNode:
    """A node's local transform, components, and children."""

    def __init__(
        self,
        node_id: int,
        name: str,
        parent: SceneNode | None,
        local_xform: wp.transform | None = None,
    ) -> None:
        """Create a visible node without components under parent."""
        self.id = node_id
        self.name = name
        self.parent = parent
        self.visible = True
        self.translation_locks = [False, False, False]
        self.rotation_locks = [False, False, False]
        self.euler_degrees: list[float] | None = None
        self.children: list[SceneNode] = []
        self.target_pose: TargetPoseComponent | None = None
        self.interaction_box: InteractionBoxComponent | None = None
        self.pivot: PivotComponent | None = None
        self.local_xform = (
            wp.transform_identity()
            if local_xform is None
            else wp.transform(local_xform)
        )

    @property
    def components(
        self,
    ) -> tuple[TargetPoseComponent | InteractionBoxComponent | PivotComponent, ...]:
        """The filled slots: Target Pose, Interaction Box, then Pivot."""
        return tuple(
            component
            for component in (self.target_pose, self.interaction_box, self.pivot)
            if component is not None
        )


@dataclass(slots=True)
class TargetPoseSequence:
    """Targets in generation order, with an optional pivot to align them."""

    id: int
    name: str
    node_ids: list[int]
    pivot_node_id: int | None = None


class SmartPrimitiveDocument:
    """The editable hierarchy, components, and target pose sequences."""

    def __init__(
        self, rig: ModelRig, reference_pose: wp.array, model_to_runtime: wp.transform
    ) -> None:
        """Create an empty document that holds only the root node.

        Args:
            rig: Rig the document is authored for.
            reference_pose: Reference pose that sets the pose layout, either local joint
                transforms or joint coordinates.
            model_to_runtime: Transform that places clip poses in world space.
        """
        self._next_id = 1
        self._nodes: dict[int, SceneNode] = {}
        self.root = SceneNode(self._new_id(), "Root", None)
        self._nodes[self.root.id] = self.root
        self.sequences: list[TargetPoseSequence] = []
        self.set_rig(rig, reference_pose, model_to_runtime)

    def set_rig(
        self, rig: ModelRig, reference_pose: wp.array, model_to_runtime: wp.transform
    ) -> None:
        """Switch the rig and reset every target to its reference pose.

        No target may have a clip.
        """
        rig = ModelRig(rig)
        targets = [component for _, component in self.iter_target_poses()]
        if any(target.clip is not None for target in targets):
            raise ValueError("Cannot change the rig while a target has a clip.")
        self.rig = rig
        self.reference_pose = reference_pose
        self.model_to_runtime = wp.transform(model_to_runtime)
        up = wp.quat_rotate_inv(self.model_to_runtime.q, wp.vec3(0.0, 0.0, 1.0))
        self.up_index = int(np.argmax(np.abs(np.asarray(up))))
        for target in targets:
            target.set_reference_pose(reference_pose, self.up_index)

    def set_contents(
        self, root: SceneNode, sequences: list[TargetPoseSequence], next_id: int
    ) -> None:
        """Replace the hierarchy and the sequences, such as with ones loaded from a file.

        next_id must exceed every ID in use. New nodes, components, and sequences take
        their IDs from it.
        """
        self.root = root
        self._nodes = {node.id: node for node in self.iter_subtree(root)}
        self.sequences = sequences
        self._next_id = next_id

    def _new_id(self) -> int:
        item_id = self._next_id
        self._next_id += 1
        return item_id

    def add_node(self, parent: SceneNode | None = None) -> SceneNode:
        """Add a node under parent, or under the root node, and return it."""
        parent = parent or self.root
        node_id = self._new_id()
        node = SceneNode(node_id, f"Node {node_id}", parent)
        parent.children.append(node)
        self._nodes[node.id] = node
        return node

    def remove_node(self, node: SceneNode) -> None:
        """Delete a node with its descendants, and drop them from every sequence.

        The root node stays.
        """
        if node is self.root:
            return
        node.parent.children.remove(node)
        removed_node_ids = {descendant.id for descendant in self.iter_subtree(node)}
        for descendant in self.iter_subtree(node):
            self._nodes.pop(descendant.id, None)
        for sequence in self.sequences:
            if sequence.pivot_node_id in removed_node_ids:
                sequence.pivot_node_id = None
            sequence.node_ids[:] = [
                node_id
                for node_id in sequence.node_ids
                if node_id not in removed_node_ids
            ]

    def find_node(self, node_id: int | None) -> SceneNode | None:
        """Return the node with node_id, or None."""
        return self._nodes.get(node_id)

    def iter_nodes(self) -> Iterator[SceneNode]:
        """Visit the root and descendants with every parent before its children."""
        return self.iter_subtree(self.root)

    @staticmethod
    def iter_subtree(root: SceneNode) -> Iterator[SceneNode]:
        """Visit parents before children, preserving the authored sibling order."""
        pending = [root]
        while pending:
            node = pending.pop()
            yield node
            pending.extend(reversed(node.children))

    @staticmethod
    def iter_ancestors(node: SceneNode) -> Iterator[SceneNode]:
        """Visit a node, then walk up through its parents."""
        current: SceneNode | None = node
        while current is not None:
            yield current
            current = current.parent

    def world_xform(self, node: SceneNode) -> wp.transform:
        """Return the node's transform composed through all of its parents."""
        world_xform = wp.transform(node.local_xform)
        parent = node.parent
        while parent is not None:
            world_xform = wp.mul(parent.local_xform, world_xform)
            parent = parent.parent
        return world_xform

    def target_pose_world_xform(self, node: SceneNode) -> wp.transform:
        """Return the transform that places the node's clip poses in world space."""
        return wp.mul(self.world_xform(node), self.model_to_runtime)

    def is_visible(self, node: SceneNode) -> bool:
        """Return whether the node and all of its ancestors are visible."""
        return all(ancestor.visible for ancestor in self.iter_ancestors(node))

    def can_reparent(self, node: SceneNode, new_parent: SceneNode) -> bool:
        """Return whether node can move under new_parent without forming a cycle."""
        if node is self.root or node.parent is new_parent:
            return False
        return all(ancestor is not node for ancestor in self.iter_ancestors(new_parent))

    def reparent(self, node: SceneNode, new_parent: SceneNode) -> bool:
        """Move node under new_parent, keeping its placement.

        Return whether the node moved.
        """
        if not self.can_reparent(node, new_parent):
            return False
        world_xform = self.world_xform(node)
        new_parent_world = self.world_xform(new_parent)
        node.parent.children.remove(node)
        node.parent = new_parent
        new_parent.children.append(node)
        node.local_xform = wp.mul(wp.transform_inverse(new_parent_world), world_xform)
        return True

    def add_target_pose(self, node: SceneNode) -> TargetPoseComponent:
        """Return the node's Target Pose component, adding one when it has none."""
        if node.target_pose is None:
            node.target_pose = TargetPoseComponent(
                self._new_id(), self.reference_pose, self.up_index
            )
        return node.target_pose

    def add_interaction_box(self, node: SceneNode) -> InteractionBoxComponent:
        """Return the node's Interaction Box component, adding one when it has none."""
        if node.interaction_box is None:
            node.interaction_box = InteractionBoxComponent(self._new_id())
        return node.interaction_box

    def add_pivot(self, node: SceneNode) -> PivotComponent:
        """Return the node's Pivot component, adding one when it has none."""
        if node.pivot is None:
            node.pivot = PivotComponent(self._new_id())
        return node.pivot

    def remove_component(
        self,
        node: SceneNode,
        component: TargetPoseComponent | InteractionBoxComponent | PivotComponent,
    ) -> None:
        """Remove a component from a node, and drop every sequence reference to it."""
        if component is node.target_pose:
            node.target_pose = None
            for sequence in self.sequences:
                sequence.node_ids[:] = [
                    node_id for node_id in sequence.node_ids if node_id != node.id
                ]
        elif component is node.interaction_box:
            node.interaction_box = None
        elif component is node.pivot:
            node.pivot = None
            for sequence in self.sequences:
                if sequence.pivot_node_id == node.id:
                    sequence.pivot_node_id = None

    def add_sequence(self) -> TargetPoseSequence:
        """Add an empty sequence with the next free name, and return it."""
        existing_names = {sequence.name for sequence in self.sequences}
        sequence_number = 1
        while f"Sequence {sequence_number}" in existing_names:
            sequence_number += 1
        sequence = TargetPoseSequence(self._new_id(), f"Sequence {sequence_number}", [])
        self.sequences.append(sequence)
        return sequence

    def find_sequence(self, sequence_id: int | None) -> TargetPoseSequence | None:
        """Return the sequence with sequence_id, or None."""
        return next(
            (sequence for sequence in self.sequences if sequence.id == sequence_id),
            None,
        )

    def remove_sequence(self, sequence: TargetPoseSequence) -> None:
        """Delete a sequence."""
        self.sequences.remove(sequence)

    def add_sequence_node(self, sequence: TargetPoseSequence, node: SceneNode) -> None:
        """Append a node that has a Target Pose to a sequence."""
        if self.find_sequence(sequence.id) is not sequence:
            raise ValueError("The sequence must belong to this Smart Primitive.")
        if self.find_node(node.id) is not node:
            raise ValueError("The node must belong to this Smart Primitive.")
        if node.target_pose is None:
            raise ValueError("Sequence nodes must own a Target Pose component.")
        sequence.node_ids.append(node.id)

    def set_sequence_pivot(
        self, sequence: TargetPoseSequence, node: SceneNode | None
    ) -> None:
        """Set the node whose pivot aligns the sequence. None removes the pivot."""
        if self.find_sequence(sequence.id) is not sequence:
            raise ValueError("The sequence must belong to this Smart Primitive.")
        if node is not None:
            if self.find_node(node.id) is not node or node.pivot is None:
                raise ValueError(
                    "The pivot node must belong here and own a Pivot component."
                )
        sequence.pivot_node_id = None if node is None else node.id

    @staticmethod
    def remove_sequence_node(sequence: TargetPoseSequence, index: int) -> None:
        """Remove the step at index from a sequence."""
        if not 0 <= index < len(sequence.node_ids):
            raise IndexError(f"Sequence node index {index!r} is out of range.")
        sequence.node_ids.pop(index)

    @staticmethod
    def move_sequence_node(
        sequence: TargetPoseSequence, source_index: int, destination_index: int
    ) -> None:
        """Move a sequence step from one index to another."""
        node_count = len(sequence.node_ids)
        if not 0 <= source_index < node_count:
            raise IndexError(f"Sequence node index {source_index!r} is out of range.")
        if not 0 <= destination_index < node_count:
            raise IndexError(
                f"Sequence destination index {destination_index!r} is out of range."
            )
        node_id = sequence.node_ids.pop(source_index)
        sequence.node_ids.insert(destination_index, node_id)

    def sequences_referencing(self, node: SceneNode) -> Iterator[TargetPoseSequence]:
        """Yield each sequence that uses node as a step or as its pivot."""
        for sequence in self.sequences:
            if node.id in sequence.node_ids or node.id == sequence.pivot_node_id:
                yield sequence

    def iter_target_poses(self) -> Iterator[tuple[SceneNode, TargetPoseComponent]]:
        """Yield each node with its Target Pose component, parents before children."""
        return (
            (node, node.target_pose)
            for node in self.iter_nodes()
            if node.target_pose is not None
        )

    def iter_interaction_boxes(
        self,
    ) -> Iterator[tuple[SceneNode, InteractionBoxComponent]]:
        """Yield each node with its Interaction Box component, parents before children."""
        return (
            (node, node.interaction_box)
            for node in self.iter_nodes()
            if node.interaction_box is not None
        )

    def iter_pivots(self) -> Iterator[tuple[SceneNode, PivotComponent]]:
        """Yield each node with its Pivot component, parents before children."""
        return (
            (node, node.pivot) for node in self.iter_nodes() if node.pivot is not None
        )


class SmartPrimitiveDocumentError(DocumentError):
    """Raised when a Smart Primitive breaks a document rule or cannot be prepared for runtime.

    A clip that fails to load also raises it. JSON and file problems raise DocumentError, the
    base class.
    """


def _serialize_target_pose(component: TargetPoseComponent, document_path: Path) -> dict:
    return {
        "type": _TARGET_POSE_COMPONENT_TYPE,
        "id": component.id,
        "clip": None
        if component.clip is None
        else document_io.serialize_reference(component.clip.path, document_path),
        "sample_time_seconds": component.sample_time,
        "frame_index": component.frame,
        "playback_mode": component.playback_mode.value,
        "hips_constrained": component.hips_constrained,
        "num_poses_to_use": component.num_poses_to_use,
        "transition_early_by_frames": component.transition_early_by_frames,
        "min_tokens": component.min_tokens,
        "max_tokens": component.max_tokens,
        "ik_pin_mask": int(component.ik_pin_mask),
        "show_mesh": component.show_mesh,
        "show_skeletons": component.show_skeletons,
        "show_velocity": component.show_velocity,
        "show_trajectory": component.show_trajectory,
        "trajectory_scale": component.trajectory_scale,
        "color": document_io.serialize_color(
            component.color, f"Target Pose {component.id}.color"
        ),
    }


def _serialize_interaction_box(component: InteractionBoxComponent) -> dict:
    return {
        "type": _INTERACTION_BOX_COMPONENT_TYPE,
        "id": component.id,
        "width": component.width,
        "depth": component.depth,
        "height": component.height,
        "color": document_io.serialize_color(
            component.color, f"Interaction Box {component.id}.color"
        ),
    }


def _serialize_pivot(component: PivotComponent) -> dict:
    pivot = component.pivot
    return {
        "type": _PIVOT_COMPONENT_TYPE,
        "id": component.id,
        "motion_direction": list(pivot.motion_direction),
        "operation_order": pivot.operation_order.value,
        "translation_axis": pivot.translation_axis.value,
        "translation_constraint": pivot.translation_constraint.value,
        "line_min_offset": pivot.line_min_offset,
        "line_max_offset": pivot.line_max_offset,
        "rotation_axis": pivot.rotation_axis.value,
        "rotation_constraint": pivot.rotation_constraint.value,
        "cone_half_angle_degrees": pivot.cone_half_angle_degrees,
        "cone_side": pivot.cone_side.value,
        "show_motion_direction": component.show_motion_direction,
        "show_translation_constraint": component.show_translation_constraint,
        "show_rotation_constraint": component.show_rotation_constraint,
    }


def _serialize_scene_node(node: SceneNode, document_path: Path) -> dict:
    components = []
    if node.target_pose is not None:
        components.append(_serialize_target_pose(node.target_pose, document_path))
    if node.interaction_box is not None:
        components.append(_serialize_interaction_box(node.interaction_box))
    if node.pivot is not None:
        components.append(_serialize_pivot(node.pivot))
    return {
        "id": node.id,
        "name": node.name,
        "local_transform": document_io.serialize_transform(
            node.local_xform, f"Node {node.name!r}.local_transform"
        ),
        "euler_degrees": None
        if node.euler_degrees is None
        else [float(value) for value in node.euler_degrees],
        "visible": node.visible,
        "translation_locks": list(node.translation_locks),
        "rotation_locks": list(node.rotation_locks),
        "components": components,
        "children": [
            _serialize_scene_node(child, document_path) for child in node.children
        ],
    }


def _serialize_sequence(sequence: TargetPoseSequence) -> dict:
    return {
        "id": sequence.id,
        "name": sequence.name,
        "node_ids": list(sequence.node_ids),
        "pivot_node_id": sequence.pivot_node_id,
    }


def _prepare_runtime_target_poses(
    authoring: SmartPrimitiveDocument,
    targets: list[tuple[SceneNode, TargetPoseComponent]],
) -> tuple[np.ndarray, tuple[TargetPoseInferenceParams, ...]]:
    pose_shape = authoring.reference_pose.numpy().shape
    packed_poses = np.empty(
        (len(targets) * MODEL_POSES_PER_TOKEN, *pose_shape), dtype=np.float32
    )
    inference_params = []
    for target_index, (node, component) in enumerate(targets):
        if component.clip is None:
            raise SmartPrimitiveDocumentError(
                f"Target Pose on node {node.name!r} has no clip."
            )
        sampled_poses = component.sampled_poses.numpy()
        document_io.validate_transforms(
            sampled_poses[..., :7], f"Target Pose on node {node.name!r}"
        )
        if not np.all(np.isfinite(sampled_poses)):
            raise SmartPrimitiveDocumentError(
                f"Target Pose on node {node.name!r} contains non-finite values."
            )
        frame_offset = target_index * MODEL_POSES_PER_TOKEN
        target_slice = packed_poses[frame_offset : frame_offset + MODEL_POSES_PER_TOKEN]
        target_slice[:] = sampled_poses
        roots = target_slice.reshape(MODEL_POSES_PER_TOKEN, -1)[:, :7]
        for frame in range(MODEL_POSES_PER_TOKEN):
            roots[frame] = wp.mul(
                authoring.model_to_runtime, wp.transform(*roots[frame])
            )
        inference_params.append(
            TargetPoseInferenceParams(
                transition_early_by_frames=component.transition_early_by_frames,
                num_poses_to_use=component.num_poses_to_use,
                ik_pin_mask=component.ik_pin_mask,
                allowed_token_counts=tuple(
                    range(component.min_tokens, component.max_tokens + 1)
                ),
            )
        )
    return (packed_poses, tuple(inference_params))


def _prepare_runtime_sequences(
    authoring: SmartPrimitiveDocument, node_indices: dict[SceneNode, int]
) -> tuple[SmartPrimitiveSequence, ...]:
    """Map saved node IDs to runtime indices while preserving sequence and step order."""
    prepared = []
    for sequence in authoring.sequences:
        if not sequence.node_ids:
            raise SmartPrimitiveDocumentError(
                f"Sequence {sequence.name!r} must contain at least one Target Pose."
            )
        nodes = [authoring.find_node(node_id) for node_id in sequence.node_ids]
        start_speed = nodes[0].target_pose.velocity_magnitude
        if not math.isfinite(start_speed):
            raise SmartPrimitiveDocumentError(
                f"Sequence {sequence.name!r} has a non-finite start speed."
            )
        pivot_node_index = -1
        if sequence.pivot_node_id is not None:
            pivot_node_index = node_indices[authoring.find_node(sequence.pivot_node_id)]
        prepared.append(
            SmartPrimitiveSequence(
                name=sequence.name,
                node_indices=tuple(node_indices[node] for node in nodes),
                start_speed=start_speed,
                pivot_node_index=pivot_node_index,
            )
        )
    return tuple(prepared)


def build_runtime_smart_primitive_data(
    authoring: SmartPrimitiveDocument,
) -> SmartPrimitiveData:
    """Prepare the hierarchy, sampled poses, and interaction settings for playback."""
    nodes = list(authoring.iter_nodes())
    node_indices = {node: index for index, node in enumerate(nodes)}
    node_local_transforms = np.array(
        [list(node.local_xform) for node in nodes], dtype=np.float32
    )
    node_parent_indices = np.asarray(
        [-1 if node.parent is None else node_indices[node.parent] for node in nodes],
        dtype=np.int32,
    )
    node_names = tuple(node.name for node in nodes)
    if len(set(node_names)) != len(node_names):
        raise SmartPrimitiveDocumentError("Runtime node names must be unique.")
    sequences = _prepare_runtime_sequences(authoring, node_indices)
    pivot_components = list(authoring.iter_pivots())
    pivot_node_indices = np.asarray(
        [node_indices[node] for node, _ in pivot_components], dtype=np.int32
    )
    pivots = tuple((component.pivot for _, component in pivot_components))
    targets = list(authoring.iter_target_poses())
    target_node_indices = np.asarray(
        [node_indices[node] for node, _ in targets], dtype=np.int32
    )
    target_poses, target_inference_params = _prepare_runtime_target_poses(
        authoring, targets
    )
    boxes = list(authoring.iter_interaction_boxes())
    box_node_indices = np.asarray(
        [node_indices[node] for node, _ in boxes], dtype=np.int32
    )
    box_half_extents = np.asarray(
        [
            (0.5 * component.width, 0.5 * component.depth, 0.5 * component.height)
            for _, component in boxes
        ],
        dtype=np.float32,
    ).reshape((len(boxes), 3))
    return SmartPrimitiveData(
        rig=authoring.rig,
        node_local_transforms=node_local_transforms,
        node_parent_indices=node_parent_indices,
        node_names=node_names,
        target_node_indices=target_node_indices,
        target_poses=target_poses,
        target_inference_params=target_inference_params,
        box_node_indices=box_node_indices,
        box_half_extents=box_half_extents,
        pivot_node_indices=pivot_node_indices,
        pivots=pivots,
        sequences=sequences,
    )


def save_smart_primitive_document(
    authoring: SmartPrimitiveDocument,
    path: str | Path,
    selected_node_id: int,
    camera_state: dict,
) -> Path:
    """Save the document and editor settings, then return the file path."""
    document_path = SMART_PRIMITIVE_DOCUMENT.normalize_path(path).resolve()
    document = SMART_PRIMITIVE_DOCUMENT.header(
        document_path, authoring.rig, selected_node_id, camera_state
    )
    document["sequences"] = [
        _serialize_sequence(sequence) for sequence in authoring.sequences
    ]
    document["root"] = _serialize_scene_node(authoring.root, document_path)
    document_io.save_json(document, document_path)
    return document_path


class _SmartPrimitiveDocumentLoader:
    def __init__(
        self,
        clip_registry: ClipRegistry,
        reference_pose: wp.array,
        model_to_runtime: wp.transform,
        document_path: Path,
    ) -> None:
        self.clip_registry = clip_registry
        self.reference_pose = reference_pose
        self.model_to_runtime = model_to_runtime
        self.document_path = document_path
        self.used_ids: set[int] = set()
        self.maximum_id = 0
        self.nodes: dict[int, SceneNode] = {}

    def load(self) -> tuple[SmartPrimitiveDocument, int, dict]:
        document, rig, selected_node_id, camera_state = SMART_PRIMITIVE_DOCUMENT.read(
            self.document_path, _CONTENT_FIELDS
        )
        if rig != self.clip_registry.rig:
            raise SmartPrimitiveDocumentError(
                f"{self.document_path.name} was saved for the '{rig}' rig, but this app uses '{self.clip_registry.rig}'."
            )
        authoring = SmartPrimitiveDocument(
            self.clip_registry.rig, self.reference_pose, self.model_to_runtime
        )
        root = self._load_node(document.object("root", _NODE_FIELDS), None, authoring)
        sequences = [
            self._load_sequence(data, authoring)
            for data in document.object_array("sequences", _SEQUENCE_FIELDS)
        ]
        sequence_names = [sequence.name for sequence in sequences]
        if len(set(sequence_names)) != len(sequence_names):
            raise SmartPrimitiveDocumentError("Sequence names must be unique.")
        authoring.set_contents(root, sequences, self.maximum_id + 1)
        if authoring.find_node(selected_node_id) is None:
            selected_node_id = authoring.root.id
        return (authoring, selected_node_id, camera_state)

    def _load_sequence(
        self, data: DocumentObject, authoring: SmartPrimitiveDocument
    ) -> TargetPoseSequence:
        sequence = TargetPoseSequence(self._claim_id(data), data.string("name"), [])
        for index, value in enumerate(data.array("node_ids")):
            location = f"{data.field_location('node_ids')}[{index}]"
            node_id = document_io.integer(value, location, minimum=1)
            node = self.nodes.get(node_id)
            if node is None:
                raise SmartPrimitiveDocumentError(
                    f"{location} does not identify a hierarchy node."
                )
            if node.target_pose is None:
                raise SmartPrimitiveDocumentError(
                    f"{location} identifies node {node.name!r}, which has no Target Pose."
                )
            sequence.node_ids.append(node_id)
        if data.value("pivot_node_id") is not None:
            pivot_id = data.integer("pivot_node_id", minimum=1)
            node = self.nodes.get(pivot_id)
            if node is None or node.pivot is None:
                raise SmartPrimitiveDocumentError(
                    f"{data.field_location('pivot_node_id')} must identify a node with a Pivot component."
                )
            sequence.pivot_node_id = pivot_id
        return sequence

    def _claim_id(self, data: DocumentObject) -> int:
        item_id = data.integer("id", minimum=1)
        if item_id in self.used_ids:
            raise SmartPrimitiveDocumentError(
                f"Duplicate ID {item_id} at {data.field_location('id')}."
            )
        self.used_ids.add(item_id)
        self.maximum_id = max(self.maximum_id, item_id)
        return item_id

    def _load_node(
        self,
        data: DocumentObject,
        parent: SceneNode | None,
        authoring: SmartPrimitiveDocument,
    ) -> SceneNode:
        node_id = self._claim_id(data)
        node = SceneNode(
            node_id,
            data.string("name"),
            parent,
            document_io.load_transform(
                data.object("local_transform", document_io.TRANSFORM_FIELDS)
            ),
        )
        node.visible = data.boolean("visible")
        node.translation_locks = data.bool_vector("translation_locks", 3)
        node.rotation_locks = data.bool_vector("rotation_locks", 3)
        euler_degrees = data.value("euler_degrees")
        node.euler_degrees = (
            None if euler_degrees is None else data.number_vector("euler_degrees", 3)
        )
        self.nodes[node.id] = node
        components_location = data.field_location("components")
        for index, value in enumerate(data.array("components")):
            location = f"{components_location}[{index}]"
            component_type = document_io.object_type(value, location)
            if component_type == _TARGET_POSE_COMPONENT_TYPE:
                if node.target_pose is not None:
                    raise SmartPrimitiveDocumentError(
                        f"{data.location} contains more than one Target Pose."
                    )
                node.target_pose = self._load_target_pose_component(
                    DocumentObject(value, location, _TARGET_POSE_FIELDS), authoring
                )
            elif component_type == _INTERACTION_BOX_COMPONENT_TYPE:
                if node.interaction_box is not None:
                    raise SmartPrimitiveDocumentError(
                        f"{data.location} contains more than one Interaction Box."
                    )
                node.interaction_box = self._load_interaction_box_component(
                    DocumentObject(value, location, _INTERACTION_BOX_FIELDS)
                )
            elif component_type == _PIVOT_COMPONENT_TYPE:
                if node.pivot is not None:
                    raise SmartPrimitiveDocumentError(
                        f"{data.location} contains more than one Pivot."
                    )
                node.pivot = self._load_pivot_component(
                    DocumentObject(value, location, _PIVOT_FIELDS)
                )
            else:
                raise SmartPrimitiveDocumentError(
                    f"Unsupported component type {component_type!r} at {location}."
                )
        for child_data in data.object_array("children", _NODE_FIELDS):
            child = self._load_node(child_data, node, authoring)
            node.children.append(child)
        return node

    def _load_pivot_component(self, data: DocumentObject) -> PivotComponent:
        component = PivotComponent(self._claim_id(data))
        try:
            component.pivot = Pivot(
                motion_direction=tuple(data.number_vector("motion_direction", 3)),
                operation_order=OperationOrder(data.string("operation_order")),
                translation_axis=Axis(data.string("translation_axis")),
                translation_constraint=ConstraintType(
                    data.string("translation_constraint")
                ),
                line_min_offset=data.number("line_min_offset"),
                line_max_offset=data.number("line_max_offset"),
                rotation_axis=Axis(data.string("rotation_axis")),
                rotation_constraint=ConstraintType(data.string("rotation_constraint")),
                cone_half_angle_degrees=data.number("cone_half_angle_degrees"),
                cone_side=ConeSide(data.string("cone_side")),
            )
        except ValueError as exc:
            raise SmartPrimitiveDocumentError(
                f"Invalid Pivot at {data.location}: {exc}"
            ) from exc
        component.show_motion_direction = data.boolean("show_motion_direction")
        component.show_translation_constraint = data.boolean(
            "show_translation_constraint"
        )
        component.show_rotation_constraint = data.boolean("show_rotation_constraint")
        return component

    def _load_target_pose_component(
        self, data: DocumentObject, authoring: SmartPrimitiveDocument
    ) -> TargetPoseComponent:
        component_id = self._claim_id(data)
        component = TargetPoseComponent(
            component_id, authoring.reference_pose, authoring.up_index
        )
        playback_mode = self._load_playback_mode(data)
        sample_time = data.number("sample_time_seconds")
        if sample_time < 0.0:
            raise SmartPrimitiveDocumentError(
                f"{data.field_location('sample_time_seconds')} cannot be negative."
            )
        frame_index = data.integer("frame_index", minimum=0)
        hips_constrained = data.boolean("hips_constrained")
        self._load_target_pose_properties(component, data)
        clip = self._load_clip(data)
        self._validate_sample_position(clip, sample_time, frame_index, data.location)
        component.restore_sample_state(
            clip, playback_mode, sample_time, frame_index, hips_constrained
        )
        return component

    def _load_interaction_box_component(
        self, data: DocumentObject
    ) -> InteractionBoxComponent:
        component = InteractionBoxComponent(self._claim_id(data))
        dimension_fields = ("width", "depth", "height")
        dimensions = [data.number(field) for field in dimension_fields]
        for field, value in zip(dimension_fields, dimensions):
            if value < 0.0:
                raise SmartPrimitiveDocumentError(
                    f"{data.field_location(field)} cannot be negative."
                )
        component.set_dimensions(*dimensions)
        component.color = data.color("color")
        return component

    @staticmethod
    def _load_playback_mode(data: DocumentObject) -> PlaybackMode:
        value = data.string("playback_mode")
        try:
            return PlaybackMode(value)
        except ValueError as exc:
            raise SmartPrimitiveDocumentError(
                f"{data.field_location('playback_mode')} must be 'time' or 'frames'."
            ) from exc

    def _load_target_pose_properties(
        self, component: TargetPoseComponent, data: DocumentObject
    ) -> None:
        component.num_poses_to_use = data.integer(
            "num_poses_to_use", minimum=1, maximum=MODEL_POSES_PER_TOKEN
        )
        component.transition_early_by_frames = data.integer(
            "transition_early_by_frames",
            minimum=0,
            maximum=MAX_TRANSITION_EARLY_BY_FRAMES,
        )
        component.min_tokens = data.integer(
            "min_tokens", minimum=MODEL_MIN_TOKEN_COUNT, maximum=MODEL_MAX_TOKEN_COUNT
        )
        component.max_tokens = data.integer(
            "max_tokens", minimum=component.min_tokens, maximum=MODEL_MAX_TOKEN_COUNT
        )
        ik_pin_mask = data.integer("ik_pin_mask", minimum=0)
        if ik_pin_mask & ~int(IKPin.ALL):
            raise SmartPrimitiveDocumentError(
                f"{data.field_location('ik_pin_mask')} contains unsupported bits."
            )
        component.ik_pin_mask = IKPin(ik_pin_mask)
        component.show_mesh = data.boolean("show_mesh")
        component.show_skeletons = data.boolean("show_skeletons")
        component.show_velocity = data.boolean("show_velocity")
        component.show_trajectory = data.boolean("show_trajectory")
        trajectory_scale = data.number("trajectory_scale")
        try:
            component.set_trajectory_scale(trajectory_scale)
        except ValueError as exc:
            raise SmartPrimitiveDocumentError(
                f"{data.field_location('trajectory_scale')} must be {MIN_TRAJECTORY_SCALE:.1f} to {MAX_TRAJECTORY_SCALE:.1f}."
            ) from exc
        component.color = data.color("color")

    @staticmethod
    def _validate_sample_position(
        clip: ClipAsset | None, sample_time: float, frame_index: int, location: str
    ) -> None:
        if clip is None:
            if sample_time != 0.0 or frame_index != 0:
                raise SmartPrimitiveDocumentError(
                    f"{location} cannot select a frame without a clip."
                )
            return
        if frame_index > clip.last_frame:
            raise SmartPrimitiveDocumentError(
                f"{location}.frame_index exceeds the clip's last frame."
            )
        if sample_time > clip.duration + 1e-06:
            raise SmartPrimitiveDocumentError(
                f"{location}.sample_time_seconds exceeds the clip duration."
            )

    def _load_clip(self, data: DocumentObject) -> ClipAsset | None:
        resolved_path = document_io.resolve_reference(
            data, "clip", self.document_path, kind="clip"
        )
        if resolved_path is None:
            return None
        try:
            return self.clip_registry.load(resolved_path)
        except ClipLoadError as exc:
            raise SmartPrimitiveDocumentError(str(exc)) from exc


def load_smart_primitive_rig(path: str | Path) -> ModelRig:
    """Return the rig a document was saved for, reading only its header."""
    _, rig, _, _ = SMART_PRIMITIVE_DOCUMENT.read(
        Path(path).expanduser().resolve(), _CONTENT_FIELDS
    )
    return rig


def load_smart_primitive_document(
    clip_registry: ClipRegistry,
    reference_pose: wp.array,
    model_to_runtime: wp.transform,
    path: str | Path,
) -> tuple[SmartPrimitiveDocument, int, dict]:
    """Load a document for the registry's rig, checking it before loading clips."""
    return _SmartPrimitiveDocumentLoader(
        clip_registry,
        reference_pose,
        model_to_runtime,
        Path(path).expanduser().resolve(),
    ).load()


class SmartPrimitiveAssetPool:
    """Load documents and clips once and share prepared data between runtime instances."""

    def __init__(
        self,
        clip_registry: ClipRegistry,
        reference_pose: wp.array,
        model_to_runtime: wp.transform,
    ) -> None:
        """Create an empty pool for one rig's clip registry."""
        self.clip_registry = clip_registry
        self.reference_pose = reference_pose
        self.device = clip_registry.device
        self.model_to_runtime = wp.transform(model_to_runtime)
        self._assets: dict[Path, SmartPrimitiveAsset] = {}

    def load(self, path: str | Path) -> SmartPrimitiveAsset:
        """Return the Smart Primitive asset for path, preparing it on first use."""
        resolved_path = Path(path).expanduser().resolve()
        asset = self._assets.get(resolved_path)
        if asset is None:
            with wp.ScopedDevice(self.device):
                authoring, _, _ = load_smart_primitive_document(
                    self.clip_registry,
                    self.reference_pose,
                    self.model_to_runtime,
                    resolved_path,
                )
                asset = SmartPrimitiveAsset(
                    build_runtime_smart_primitive_data(authoring), device=self.device
                )
            self._assets[resolved_path] = asset
        return asset

    def instantiate(
        self,
        path: str | Path,
        smart_primitive_type: type[SmartPrimitive] = SmartPrimitive,
    ) -> SmartPrimitive:
        """Return a new Smart Primitive instance of smart_primitive_type that shares the asset at path."""
        return smart_primitive_type(self.load(path))
