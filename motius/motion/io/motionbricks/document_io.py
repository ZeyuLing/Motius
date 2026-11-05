# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""File handling and JSON validation for shared application documents."""

from __future__ import annotations

import json
import math
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from motius.configuration.motionbricks_runtime_contract import ModelRig

_CAMERA_FIELDS = ("pos", "pivot", "pitch", "yaw", "fov", "near", "far")
_HEADER_FIELDS = ("format", "schema_version", "name", "rig", "editor")
TRANSFORM_FIELDS = ("translation", "rotation_xyzw")


class DocumentError(ValueError):
    """Raised for invalid document data or failed file operations."""


def integer(
    value, location: str, minimum: int | None = None, maximum: int | None = None
) -> int:
    """Return value as an integer within the given limits, or raise DocumentError."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DocumentError(f"{location} must be an integer.")
    if minimum is not None and value < minimum:
        raise DocumentError(f"{location} must be at least {minimum}.")
    if maximum is not None and value > maximum:
        raise DocumentError(f"{location} must be at most {maximum}.")
    return value


def number(value, location: str) -> float:
    """Return value as a finite float, or raise DocumentError."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or (not math.isfinite(value))
    ):
        raise DocumentError(f"{location} must be a finite number.")
    return float(value)


def number_vector(value, size: int, location: str) -> list[float]:
    """Return value as a list of size finite floats, or raise DocumentError."""
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise DocumentError(f"{location} must be an array of {size} numbers.")
    return [number(item, f"{location}[{index}]") for index, item in enumerate(value)]


def string(value, location: str, *, allow_empty: bool = False) -> str:
    """Return value as a string, or raise DocumentError.

    It must be non-empty unless allow_empty is set.
    """
    if not isinstance(value, str) or (not allow_empty and (not value.strip())):
        kind = "string" if allow_empty else "non-empty string"
        raise DocumentError(f"{location} must be a {kind}.")
    return value


def _boolean(value, location: str) -> bool:
    if not isinstance(value, bool):
        raise DocumentError(f"{location} must be a boolean.")
    return value


def _color(value, location: str) -> list[float]:
    values = number_vector(value, 3, location)
    if any(not 0.0 <= channel <= 1.0 for channel in values):
        raise DocumentError(f"{location} values must be 0 to 1.")
    return values


def serialize_color(color: wp.vec3, location: str) -> list[float]:
    """Return an RGB color as JSON values, or raise DocumentError when a channel is outside 0 to 1."""
    return _color(list(color), location)


class DocumentObject:
    """Read JSON fields, with errors that point to the invalid field."""

    def __init__(self, value, location: str, fields) -> None:
        """Check that a JSON object holds exactly the given fields."""
        if not isinstance(value, dict):
            raise DocumentError(f"{location} must be an object.")
        for field in fields:
            if field not in value:
                raise DocumentError(f"Missing {location}.{field}.")
        unexpected = sorted(set(value) - set(fields))
        if unexpected:
            names = ", ".join(repr(field) for field in unexpected)
            raise DocumentError(f"Unexpected fields in {location}: {names}.")
        self._data = value
        self.location = location

    def field_location(self, field: str) -> str:
        """Return the location of a field, for error messages."""
        return f"{self.location}.{field}"

    def value(self, field: str):
        """Return a field's value as stored."""
        return self._data[field]

    def object(self, field: str, fields) -> DocumentObject:
        """Return a field as a DocumentObject with the given fields."""
        return DocumentObject(self.value(field), self.field_location(field), fields)

    def array(self, field: str) -> list:
        """Return a field as a list, or raise DocumentError."""
        value = self.value(field)
        if not isinstance(value, list):
            raise DocumentError(f"{self.field_location(field)} must be an array.")
        return value

    def object_array(self, field: str, fields) -> list[DocumentObject]:
        """Return a field as a list of DocumentObject values with the given fields."""
        return [
            DocumentObject(value, f"{self.field_location(field)}[{index}]", fields)
            for index, value in enumerate(self.array(field))
        ]

    def string(self, field: str, *, allow_empty: bool = False) -> str:
        """Return a field checked with string()."""
        return string(
            self.value(field), self.field_location(field), allow_empty=allow_empty
        )

    def boolean(self, field: str) -> bool:
        """Return a field as a boolean, or raise DocumentError."""
        return _boolean(self.value(field), self.field_location(field))

    def integer(
        self, field: str, minimum: int | None = None, maximum: int | None = None
    ) -> int:
        """Return a field checked with integer()."""
        return integer(self.value(field), self.field_location(field), minimum, maximum)

    def number(self, field: str) -> float:
        """Return a field checked with number()."""
        return number(self.value(field), self.field_location(field))

    def number_vector(self, field: str, size: int) -> list[float]:
        """Return a field checked with number_vector()."""
        return number_vector(self.value(field), size, self.field_location(field))

    def color(self, field: str) -> wp.vec3:
        """Return a field as an RGB color with channels from 0 to 1, or raise DocumentError."""
        return wp.vec3(*_color(self.value(field), self.field_location(field)))

    def bool_vector(self, field: str, size: int) -> list[bool]:
        """Return a field as a list of size booleans, or raise DocumentError."""
        values = self.array(field)
        location = self.field_location(field)
        if len(values) != size:
            raise DocumentError(f"{location} must contain exactly {size} booleans.")
        return [
            _boolean(value, f"{location}[{index}]")
            for index, value in enumerate(values)
        ]


def validate_transforms(values: np.ndarray, location: str) -> None:
    """Require finite transforms with unit quaternions (within 1e-3), or raise DocumentError.

    Args:
        values: Transforms as 7 values each: position, then XYZW quaternion.
        location: Where the transforms are, for error messages.
    """
    if not np.all(np.isfinite(values)):
        raise DocumentError(f"{location} contains non-finite values.")
    quaternion_norms = np.linalg.norm(values[..., 3:7], axis=-1)
    if np.any(np.abs(quaternion_norms - 1.0) > 0.001):
        raise DocumentError(f"{location} contains a non-unit quaternion.")


def object_type(value, location: str) -> str:
    """Return the type field of a JSON object whose other fields depend on its type.

    A value that is not an object, or that has no type field, raises DocumentError.
    """
    if not isinstance(value, dict):
        raise DocumentError(f"{location} must be an object.")
    if "type" not in value:
        raise DocumentError(f"Missing {location}.type.")
    return string(value["type"], f"{location}.type")


def serialize_transform(transform: wp.transform, location: str) -> dict:
    """Return a transform as JSON values: translation and rotation_xyzw."""
    validate_transforms(np.asarray(transform), location)
    return {"translation": list(transform.p), "rotation_xyzw": list(transform.q)}


def load_transform(data: DocumentObject) -> wp.transform:
    """Read a transform from its translation and rotation_xyzw fields."""
    translation = data.number_vector("translation", 3)
    rotation = data.number_vector("rotation_xyzw", 4)
    transform = wp.transform(wp.vec3(*translation), wp.quat(*rotation))
    validate_transforms(np.asarray(transform), data.location)
    return transform


def _load_camera(data: DocumentObject) -> dict:
    """Read the saved camera state, and check its field of view and clip planes."""
    state = {
        "pos": data.number_vector("pos", 3),
        "pivot": data.number_vector("pivot", 3),
        "pitch": data.number("pitch"),
        "yaw": data.number("yaw"),
        "fov": data.number("fov"),
        "near": data.number("near"),
        "far": data.number("far"),
    }
    if not 0.0 < state["fov"] < 180.0:
        raise DocumentError(
            f"{data.field_location('fov')} must be above 0 and below 180 degrees."
        )
    if state["near"] <= 0.0:
        raise DocumentError(f"{data.field_location('near')} must be greater than zero.")
    if state["far"] <= state["near"]:
        raise DocumentError(
            f"{data.field_location('far')} must be greater than the near plane."
        )
    return state


def serialize_reference(path: Path, document_path: Path) -> str:
    """Store a path relative to the document, or absolute when no relative path exists."""
    absolute_path = path.expanduser().resolve()
    try:
        return Path(os.path.relpath(absolute_path, document_path.parent)).as_posix()
    except ValueError:
        return absolute_path.as_posix()


def resolve_reference(
    data: DocumentObject, field: str, document_path: Path, *, kind: str
) -> Path | None:
    """Return the referenced file, or None when the document has no reference."""
    if data.value(field) is None:
        return None
    path = (document_path.parent / data.string(field)).resolve()
    if not path.is_file():
        raise DocumentError(
            f"Could not find the {kind} referenced by {data.field_location(field)}: {path}."
        )
    return path


@contextmanager
def atomic_write(path: Path):
    """Replace the destination only after the temporary file is fully written."""
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary_path.open("xb") as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_json(path: Path):
    """Read a JSON document, or raise DocumentError."""
    if not path.is_file():
        raise DocumentError(f"Could not find document: {path}.")
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise DocumentError(f"Invalid JSON in {path}: {exc}") from exc
    except OSError as exc:
        raise DocumentError(f"Could not read {path}: {exc}") from exc


def save_json(document: dict, path: Path) -> None:
    """Write a JSON document atomically, or raise DocumentError."""
    try:
        data = (json.dumps(document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DocumentError(f"Could not serialize document for {path}: {exc}") from exc
    try:
        with atomic_write(path) as stream:
            stream.write(data)
    except OSError as exc:
        raise DocumentError(f"Could not save document to {path}: {exc}") from exc


@dataclass(frozen=True)
class DocumentType:
    """The header and file name rules of one kind of document.

    Every document starts with the same header. Its editor field holds the selection and
    the camera.

    Attributes:
        kind: Name of the document kind in messages and file dialogs, such as "Scene".
        format: Value of the format field.
        schema_version: Value of the schema_version field.
        suffix: File name suffix, such as ".scene.json".
        selected_field: Editor field that holds the selected item's ID.
    """

    kind: str
    format: str
    schema_version: int
    suffix: str
    selected_field: str

    def normalize_path(self, path: str | Path) -> Path:
        """Return path with the document suffix, completing a partial suffix such as .json."""
        path = Path(path).expanduser()
        lower_name = path.name.lower()
        if lower_name.endswith(self.suffix):
            return path
        if lower_name.endswith(self.suffix.removesuffix(".json")):
            return path.with_name(path.name + ".json")
        if lower_name.endswith(".json"):
            return path.with_name(path.name[:-5] + self.suffix)
        return path.with_name(path.name + self.suffix)

    def name_from_path(self, path: str | Path) -> str:
        """Return the file name without the document suffix, or raise DocumentError."""
        filename = Path(path).name
        if not filename.lower().endswith(self.suffix):
            raise DocumentError(f"{self.kind} documents must end in {self.suffix}.")
        name = filename[: -len(self.suffix)]
        if not name:
            raise DocumentError(f"A {self.kind} document must have a name.")
        return name

    def header(
        self, document_path: Path, rig: ModelRig, selected_id: int, camera_state: dict
    ) -> dict:
        """Return the header of a document saved to document_path.

        The camera is checked here, so a bad camera never reaches the file.
        """
        return {
            "format": self.format,
            "schema_version": self.schema_version,
            "name": self.name_from_path(document_path),
            "rig": rig.value,
            "editor": {
                self.selected_field: selected_id,
                "camera": _load_camera(
                    DocumentObject(
                        camera_state, "document.editor.camera", _CAMERA_FIELDS
                    )
                ),
            },
        }

    def read(
        self, document_path: Path, content_fields
    ) -> tuple[DocumentObject, ModelRig, int, dict]:
        """Read a document and check its header.

        Args:
            document_path: Resolved path of the document.
            content_fields: Top-level fields that follow the header.

        Returns:
            The document, its rig, the selected item's ID, and the camera state.
        """
        self.name_from_path(document_path)
        document = DocumentObject(
            load_json(document_path), "document", (*_HEADER_FIELDS, *content_fields)
        )
        document_format = document.string("format")
        if document_format != self.format:
            raise DocumentError(f"Unsupported {self.kind} format {document_format!r}.")
        schema_version = document.integer("schema_version", minimum=1)
        if schema_version != self.schema_version:
            raise DocumentError(
                f"Unsupported {self.kind} schema version {schema_version}."
            )
        rig = document.string("rig")
        if rig not in ModelRig:
            raise DocumentError(f"{document_path.name} uses unknown rig '{rig}'.")
        editor = document.object("editor", (self.selected_field, "camera"))
        selected_id = editor.integer(self.selected_field, minimum=1)
        camera_state = _load_camera(editor.object("camera", _CAMERA_FIELDS))
        return (document, ModelRig(rig), selected_id, camera_state)
