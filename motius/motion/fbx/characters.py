"""Packaged, redistributable Mixamo-compatible character assets."""

from __future__ import annotations

from pathlib import Path


CHARACTER_DIR = Path(__file__).with_name("characters")
BUILTIN_MIXAMO_CHARACTERS = ("atlas", "nova", "gear")


def list_mixamo_characters() -> tuple[str, ...]:
    """Return the packaged character slugs."""

    return tuple(
        name for name in BUILTIN_MIXAMO_CHARACTERS
        if (CHARACTER_DIR / f"{name}.fbx").is_file()
    )


def resolve_mixamo_character(value: str | Path) -> Path:
    """Resolve a packaged slug or a caller-provided FBX path."""

    path = Path(value).expanduser()
    if path.is_file():
        if path.suffix.casefold() != ".fbx":
            raise ValueError(f"Character asset must be an .fbx file: {path}.")
        return path.resolve()
    slug = str(value).casefold().removesuffix(".fbx")
    if slug in BUILTIN_MIXAMO_CHARACTERS:
        packaged = CHARACTER_DIR / f"{slug}.fbx"
        if packaged.is_file():
            return packaged.resolve()
        raise FileNotFoundError(
            f"Packaged character {slug!r} is missing from this Motius installation."
        )
    available = ", ".join(list_mixamo_characters()) or "none"
    raise FileNotFoundError(
        f"Character FBX does not exist: {path}. Packaged slugs: {available}."
    )


__all__ = [
    "BUILTIN_MIXAMO_CHARACTERS",
    "CHARACTER_DIR",
    "list_mixamo_characters",
    "resolve_mixamo_character",
]
