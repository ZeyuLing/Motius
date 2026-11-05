# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Scene documents shared by Scene Preview and the Scene Editor."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import ClassVar

import warp as wp

from motius.configuration.motionbricks_runtime_contract import ModelRig
from motius.motion.io.motionbricks import document_io as document_io
from motius.motion.io.motionbricks.document_io import (
    DocumentError,
    DocumentObject,
    DocumentType,
)
from motius.motion.io.motionbricks.smart_primitive_document import (
    SMART_PRIMITIVE_DOCUMENT,
    load_smart_primitive_rig,
)

SCENE_DOCUMENT = DocumentType(
    kind="Scene",
    format="motionbricks.scene",
    schema_version=1,
    suffix=".scene.json",
    selected_field="selected_item_id",
)
_CONTENT_FIELDS = ("character_start", "smart_primitives", "solid_cubes")
_CHARACTER_START_FIELDS = ("id", "transform", "euler_degrees")
_SMART_PRIMITIVE_FIELDS = (
    "id",
    "name",
    "parent_id",
    "transform",
    "euler_degrees",
    "reference",
    "visibility",
    "target_pose_color",
    "behavior",
)
_VISIBILITY_FIELDS = ("target_poses", "interaction_boxes")
_SOLID_CUBE_FIELDS = (
    "id",
    "name",
    "parent_id",
    "transform",
    "euler_degrees",
    "dimensions",
    "color",
)
_SOLID_CUBE_DIMENSION_FIELDS = ("width", "depth", "height")


class SceneDocumentError(DocumentError):
    """Raised when a scene breaks a scene rule, such as unique item IDs or a single rig.

    JSON and file problems raise DocumentError, the base class.
    """


def _default_solid_cube_transform() -> wp.transform:
    return wp.transform(wp.vec3(0.0, 0.0, 0.5), wp.quat_identity())


@dataclass(slots=True)
class CharacterStart:
    """Where the character starts in the scene. It always stays at the top level."""

    parent_id: ClassVar[None] = None
    name: ClassVar[str] = "CharacterStart"
    id: int = 1
    transform: wp.transform = field(default_factory=wp.transform_identity)
    euler_degrees: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass(slots=True)
class SmartPrimitiveInstance:
    """A placed Smart Primitive, including its behavior class and display settings.

    transform is relative to the parent item, or to world space without a parent.
    """

    id: int
    name: str
    transform: wp.transform = field(default_factory=wp.transform_identity)
    euler_degrees: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    reference: Path | None = None
    show_target_poses: bool = True
    show_interaction_boxes: bool = True
    behavior: str = ""
    target_pose_color: wp.vec3 = field(default_factory=lambda: wp.vec3(1.0, 0.3, 0.2))
    parent_id: int | None = None


@dataclass(slots=True)
class SolidCube:
    """A Solid Cube placed by its center, with its size in meters and its color.

    transform is relative to the parent item, or to world space without a parent.
    """

    id: int
    name: str
    transform: wp.transform = field(default_factory=_default_solid_cube_transform)
    euler_degrees: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    width: float = 1.0
    depth: float = 1.0
    height: float = 1.0
    color: wp.vec3 = field(default_factory=lambda: wp.vec3(0.45, 0.55, 0.7))
    parent_id: int | None = None


SceneItem = CharacterStart | SmartPrimitiveInstance | SolidCube


@dataclass(slots=True)
class SceneDefinition:
    """The scene's CharacterStart, Smart Primitives, and Solid Cubes.

    Every referenced Smart Primitive must use the scene's rig.
    Smart Primitives and cubes can sit under one another to edit them together.
    flatten() resolves that hierarchy into world transforms for runtime use.
    """

    rig: ModelRig = ModelRig.SOMA
    character_start: CharacterStart = field(default_factory=CharacterStart)
    smart_primitives: list[SmartPrimitiveInstance] = field(default_factory=list)
    solid_cubes: list[SolidCube] = field(default_factory=list)
    _next_id: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rig = ModelRig(self.rig)
        maximum_id = max(
            [self.character_start.id]
            + [instance.id for instance in self.smart_primitives]
            + [cube.id for cube in self.solid_cubes]
        )
        self._next_id = maximum_id + 1

    def add_smart_primitive(self) -> SmartPrimitiveInstance:
        """Add a Smart Primitive item with a new ID and name, and return it."""
        instance = SmartPrimitiveInstance(
            id=self._next_id, name=self._next_item_name("SmartPrimitive")
        )
        self._next_id += 1
        self.smart_primitives.append(instance)
        return instance

    def add_solid_cube(self) -> SolidCube:
        """Add a Solid Cube with a new ID and name, and return it."""
        cube = SolidCube(id=self._next_id, name=self._next_item_name("SolidCube"))
        self._next_id += 1
        self.solid_cubes.append(cube)
        return cube

    def remove_item(self, item_id: int) -> list[int]:
        """Delete an item and every item under it, and return their IDs.

        CharacterStart cannot be removed.
        """
        item = self.find_item(item_id)
        if item is None or item is self.character_start:
            return []
        removed = [descendant.id for descendant in self.subtree(item)]
        self.smart_primitives[:] = [
            instance for instance in self.smart_primitives if instance.id not in removed
        ]
        self.solid_cubes[:] = [
            cube for cube in self.solid_cubes if cube.id not in removed
        ]
        return removed

    def find_item(self, item_id: int | None) -> SceneItem | None:
        """Return the scene item with item_id, or None."""
        if item_id == self.character_start.id:
            return self.character_start
        for item in (*self.smart_primitives, *self.solid_cubes):
            if item.id == item_id:
                return item
        return None

    def children(
        self, parent_id: int | None
    ) -> list[SmartPrimitiveInstance | SolidCube]:
        """Return direct children in creation order. None selects the top-level items."""
        items = (*self.smart_primitives, *self.solid_cubes)
        return sorted(
            (item for item in items if item.parent_id == parent_id),
            key=lambda item: item.id,
        )

    def subtree(self, item: SceneItem) -> Iterator[SceneItem]:
        """Visit an item, then every item under it."""
        yield item
        for child in self.children(item.id):
            yield from self.subtree(child)

    def world_transform(self, item: SceneItem) -> wp.transform:
        """Return the item's position and rotation in world space."""
        transform = wp.transform(item.transform)
        parent = self.find_item(item.parent_id)
        while parent is not None:
            transform = wp.mul(parent.transform, transform)
            parent = self.find_item(parent.parent_id)
        return transform

    def local_transform(
        self, item: SceneItem, world_transform: wp.transform
    ) -> wp.transform:
        """Convert a world transform to the item's parent space."""
        parent = self.find_item(item.parent_id)
        if parent is None:
            return wp.transform(world_transform)
        return wp.mul(
            wp.transform_inverse(self.world_transform(parent)), world_transform
        )

    def set_parent(self, item: SceneItem, parent_id: int | None) -> bool:
        """Change an item's parent while keeping its world placement.

        None moves it to the top level. Return False if the parent is unchanged or invalid.
        """
        parent = self.find_item(parent_id)
        if (
            item is self.character_start
            or parent is self.character_start
            or (parent is None and parent_id is not None)
            or (item.parent_id == parent_id)
            or any(descendant is parent for descendant in self.subtree(item))
        ):
            return False
        world_transform = self.world_transform(item)
        item.parent_id = parent_id
        item.transform = self.local_transform(item, world_transform)
        return True

    def flatten(self) -> SceneDefinition:
        """Return a copy where every item is at the top level, placed by its world transform."""
        return SceneDefinition(
            rig=self.rig,
            character_start=replace(self.character_start),
            smart_primitives=[
                replace(
                    instance, transform=self.world_transform(instance), parent_id=None
                )
                for instance in self.smart_primitives
            ],
            solid_cubes=[
                replace(cube, transform=self.world_transform(cube), parent_id=None)
                for cube in self.solid_cubes
            ],
        )

    def _next_item_name(self, base_name: str) -> str:
        names = {item.name for item in (*self.smart_primitives, *self.solid_cubes)}
        if base_name not in names:
            return base_name
        suffix = 2
        while f"{base_name}{suffix}" in names:
            suffix += 1
        return f"{base_name}{suffix}"


def save_scene_document(
    scene: SceneDefinition, path: str | Path, selected_item_id: int, camera_state: dict
) -> Path:
    """Save the scene and return the path it was saved to.

    The path gets the .scene.json suffix when it lacks it. The previous file is replaced
    only after the write succeeds.
    """
    _validate_scene(scene)
    document_path = SCENE_DOCUMENT.normalize_path(path).resolve()
    document = SCENE_DOCUMENT.header(
        document_path, scene.rig, selected_item_id, camera_state
    )
    document["character_start"] = _serialize_character_start(scene.character_start)
    document["smart_primitives"] = [
        _serialize_smart_primitive(instance, document_path)
        for instance in scene.smart_primitives
    ]
    document["solid_cubes"] = [
        _serialize_solid_cube(cube) for cube in scene.solid_cubes
    ]
    document_io.save_json(document, document_path)
    return document_path


def load_scene_document(path: str | Path) -> tuple[SceneDefinition, int, dict]:
    """Load a scene plus its editor selection and camera state."""
    document_path = Path(path).expanduser().resolve()
    document, rig, selected_item_id, camera_state = SCENE_DOCUMENT.read(
        document_path, _CONTENT_FIELDS
    )
    scene = SceneDefinition(
        rig=rig,
        character_start=_load_character_start(
            document.object("character_start", _CHARACTER_START_FIELDS)
        ),
        smart_primitives=[
            _load_smart_primitive(data, document_path)
            for data in document.object_array(
                "smart_primitives", _SMART_PRIMITIVE_FIELDS
            )
        ],
        solid_cubes=[
            _load_solid_cube(data)
            for data in document.object_array("solid_cubes", _SOLID_CUBE_FIELDS)
        ],
    )
    _validate_scene(scene)
    if scene.find_item(selected_item_id) is None:
        selected_item_id = scene.character_start.id
    return (scene, selected_item_id, camera_state)


def check_reference_rigs(scene: SceneDefinition) -> None:
    """Check that every referenced Smart Primitive was saved for the scene's rig.

    Only each document's rig is read, so this runs before any character or clip loads.
    """
    for instance in scene.smart_primitives:
        if instance.reference is None:
            continue
        rig = load_smart_primitive_rig(instance.reference)
        if rig is not scene.rig:
            raise SceneDocumentError(
                f"{instance.reference.name} was saved for the '{rig}' rig, but this scene uses '{scene.rig}'."
            )


def _serialize_character_start(item: CharacterStart) -> dict:
    return {
        "id": item.id,
        "transform": document_io.serialize_transform(
            item.transform, "CharacterStart.transform"
        ),
        "euler_degrees": list(item.euler_degrees),
    }


def _serialize_smart_primitive(
    instance: SmartPrimitiveInstance, document_path: Path
) -> dict:
    location = f"Smart Primitive {instance.name!r}"
    return {
        "id": instance.id,
        "name": instance.name,
        "parent_id": instance.parent_id,
        "transform": document_io.serialize_transform(
            instance.transform, f"{location}.transform"
        ),
        "euler_degrees": list(instance.euler_degrees),
        "reference": None
        if instance.reference is None
        else document_io.serialize_reference(instance.reference, document_path),
        "visibility": {
            "target_poses": instance.show_target_poses,
            "interaction_boxes": instance.show_interaction_boxes,
        },
        "target_pose_color": document_io.serialize_color(
            instance.target_pose_color, f"{location}.target_pose_color"
        ),
        "behavior": instance.behavior,
    }


def _serialize_solid_cube(cube: SolidCube) -> dict:
    location = f"Solid Cube {cube.name!r}"
    return {
        "id": cube.id,
        "name": cube.name,
        "parent_id": cube.parent_id,
        "transform": document_io.serialize_transform(
            cube.transform, f"{location}.transform"
        ),
        "euler_degrees": list(cube.euler_degrees),
        "dimensions": {"width": cube.width, "depth": cube.depth, "height": cube.height},
        "color": document_io.serialize_color(cube.color, f"{location}.color"),
    }


def _load_character_start(data: DocumentObject) -> CharacterStart:
    return CharacterStart(
        id=data.integer("id", minimum=1),
        transform=document_io.load_transform(
            data.object("transform", document_io.TRANSFORM_FIELDS)
        ),
        euler_degrees=data.number_vector("euler_degrees", 3),
    )


def _load_smart_primitive(
    data: DocumentObject, document_path: Path
) -> SmartPrimitiveInstance:
    visibility = data.object("visibility", _VISIBILITY_FIELDS)
    return SmartPrimitiveInstance(
        id=data.integer("id", minimum=1),
        name=data.string("name"),
        transform=document_io.load_transform(
            data.object("transform", document_io.TRANSFORM_FIELDS)
        ),
        euler_degrees=data.number_vector("euler_degrees", 3),
        reference=document_io.resolve_reference(
            data, "reference", document_path, kind="Smart Primitive"
        ),
        show_target_poses=visibility.boolean("target_poses"),
        show_interaction_boxes=visibility.boolean("interaction_boxes"),
        target_pose_color=data.color("target_pose_color"),
        behavior=data.string("behavior", allow_empty=True),
        parent_id=_load_parent_id(data),
    )


def _load_solid_cube(data: DocumentObject) -> SolidCube:
    dimensions = data.object("dimensions", _SOLID_CUBE_DIMENSION_FIELDS)
    return SolidCube(
        id=data.integer("id", minimum=1),
        name=data.string("name"),
        transform=document_io.load_transform(
            data.object("transform", document_io.TRANSFORM_FIELDS)
        ),
        euler_degrees=data.number_vector("euler_degrees", 3),
        width=dimensions.number("width"),
        depth=dimensions.number("depth"),
        height=dimensions.number("height"),
        color=data.color("color"),
        parent_id=_load_parent_id(data),
    )


def _load_parent_id(data: DocumentObject) -> int | None:
    return (
        None
        if data.value("parent_id") is None
        else data.integer("parent_id", minimum=1)
    )


def _validate_scene(scene: SceneDefinition) -> None:
    if scene.character_start.id != 1:
        raise SceneDocumentError("CharacterStart must use scene item ID 1.")
    used_ids = {scene.character_start.id}
    for index, instance in enumerate(scene.smart_primitives):
        location = f"smart_primitives[{index}]"
        if instance.id < 1 or instance.id in used_ids:
            raise SceneDocumentError(f"{location}.id must be unique and positive.")
        used_ids.add(instance.id)
        if not instance.name.strip():
            raise SceneDocumentError(f"{location}.name cannot be empty.")
        if instance.reference is not None:
            _validate_reference_path(instance.reference, f"{location}.reference")
    for index, cube in enumerate(scene.solid_cubes):
        location = f"solid_cubes[{index}]"
        if cube.id < 1 or cube.id in used_ids:
            raise SceneDocumentError(f"{location}.id must be unique and positive.")
        used_ids.add(cube.id)
        if not cube.name.strip():
            raise SceneDocumentError(f"{location}.name cannot be empty.")
        for attribute, value in (
            ("width", cube.width),
            ("depth", cube.depth),
            ("height", cube.height),
        ):
            if value <= 0.0:
                raise SceneDocumentError(
                    f"{location}.{attribute} must be greater than zero."
                )
    _validate_hierarchy(scene)


def _validate_hierarchy(scene: SceneDefinition) -> None:
    items = {item.id: item for item in (*scene.smart_primitives, *scene.solid_cubes)}
    for item in items.values():
        ancestors = {item.id}
        parent_id = item.parent_id
        while parent_id is not None:
            if parent_id not in items:
                raise SceneDocumentError(
                    f"{item.name!r} must have a Smart Primitive or Solid Cube as its parent."
                )
            if parent_id in ancestors:
                raise SceneDocumentError(f"{item.name!r} cannot sit under itself.")
            ancestors.add(parent_id)
            parent_id = items[parent_id].parent_id


def _validate_reference_path(path: Path, location: str) -> None:
    if not path.name.lower().endswith(SMART_PRIMITIVE_DOCUMENT.suffix):
        raise SceneDocumentError(
            f"{location} must reference a {SMART_PRIMITIVE_DOCUMENT.suffix} file."
        )
    if not path.is_file():
        raise SceneDocumentError(
            f"Could not find the file {location} references: {path}."
        )


__all__ = [
    "SCENE_DOCUMENT",
    "CharacterStart",
    "SceneDefinition",
    "SceneDocumentError",
    "SceneItem",
    "SmartPrimitiveInstance",
    "SolidCube",
    "check_reference_rigs",
    "load_scene_document",
    "save_scene_document",
]
