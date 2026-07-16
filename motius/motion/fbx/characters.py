"""Resolution helpers for caller-provided rigged character FBX files."""

from __future__ import annotations

from pathlib import Path


def resolve_character_fbx(value: str | Path) -> Path:
    """Resolve and validate a caller-provided rigged character FBX path."""

    path = Path(value).expanduser()
    if path.suffix.casefold() != ".fbx":
        raise ValueError(f"Character asset must be an .fbx file: {path}.")
    if not path.is_file():
        raise FileNotFoundError(f"Character FBX does not exist: {path}.")
    return path.resolve()


__all__ = ["resolve_character_fbx"]
