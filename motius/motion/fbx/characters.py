"""Discovery and resolution for locally installed rigged character FBX files."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


_IDENTIFIER_PART = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CharacterAsset:
    """A locally installed character addressed as ``provider/slug``."""

    provider: str
    slug: str
    path: Path

    @property
    def identifier(self) -> str:
        return f"{self.provider}/{self.slug}"


def resolve_character_root(value: str | Path | None = None) -> Path:
    """Resolve the character checkpoint root without requiring it to exist."""

    candidate = value or os.environ.get("MOTIUS_CHARACTER_DIR")
    if candidate is None:
        candidate = Path(__file__).resolve().parents[3] / "checkpoints" / "characters"
    return Path(candidate).expanduser().resolve()


def list_character_assets(
    root: str | Path | None = None,
) -> tuple[CharacterAsset, ...]:
    """List installed ``<provider>/<slug>/character.fbx`` assets."""

    character_root = resolve_character_root(root)
    if not character_root.is_dir():
        return ()
    assets = [
        CharacterAsset(path.parent.parent.name, path.parent.name, path.resolve())
        for path in character_root.glob("*/*/character.fbx")
        if path.is_file()
    ]
    return tuple(sorted(assets, key=lambda asset: asset.identifier.casefold()))


def _identifier_parts(value: str | Path) -> tuple[str, ...]:
    identifier = str(value).replace("\\", "/").strip("/")
    parts = PurePosixPath(identifier).parts
    if len(parts) not in {1, 2} or any(
        _IDENTIFIER_PART.fullmatch(part) is None for part in parts
    ):
        raise ValueError(
            "Character identifier must be '<slug>' or '<provider>/<slug>', got "
            f"{value!r}."
        )
    return parts


def resolve_character_fbx(
    value: str | Path,
    *,
    root: str | Path | None = None,
) -> Path:
    """Resolve an FBX path or an installed ``provider/slug`` identifier."""

    path = Path(value).expanduser()
    if path.is_file():
        if path.suffix.casefold() != ".fbx":
            raise ValueError(f"Character asset must be an .fbx file: {path}.")
        return path.resolve()
    if path.suffix:
        if path.suffix.casefold() != ".fbx":
            raise ValueError(f"Character asset must be an .fbx file: {path}.")
        raise FileNotFoundError(f"Character FBX does not exist: {path}.")

    character_root = resolve_character_root(root)
    parts = _identifier_parts(value)
    if len(parts) == 2:
        candidates = [character_root / parts[0] / parts[1] / "character.fbx"]
    else:
        candidates = list(character_root.glob(f"*/{parts[0]}/character.fbx"))
    matches = sorted(candidate.resolve() for candidate in candidates if candidate.is_file())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        providers = [path.parent.parent.name for path in matches]
        raise ValueError(
            f"Character slug {parts[0]!r} exists under multiple providers {providers}; "
            "use '<provider>/<slug>'."
        )
    raise FileNotFoundError(
        f"Character {str(value)!r} is not installed under {character_root}. Expected "
        "<provider>/<slug>/character.fbx."
    )


__all__ = [
    "CharacterAsset",
    "list_character_assets",
    "resolve_character_fbx",
    "resolve_character_root",
]
