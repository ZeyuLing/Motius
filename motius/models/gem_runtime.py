"""Shared artifact and process helpers for repository-native GEM runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Optional


PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def project_root() -> Path:
    """Resolve the caller's project root for generated outputs."""

    configured = os.environ.get("MOTIUS_PROJECT_ROOT")
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path.cwd().resolve()
    )


def sha256_file(
    path: str | Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> str:
    """Hash one artifact without deserializing it."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or download a complete Hugging Face snapshot."""

    candidate = Path(pretrained_model_name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            token=token,
            local_files_only=local_files_only,
        )
    ).resolve()


def load_manifest(
    artifact_root: str | Path,
    *,
    filename: str,
    artifact_format: str,
    source_revision: str,
) -> dict:
    """Load and validate an immutable Motius artifact manifest."""

    root = Path(artifact_root)
    path = root / filename
    if not path.is_file():
        raise FileNotFoundError(f"Artifact is missing {filename}: {root}")
    payload = json.loads(path.read_text())
    if payload.get("artifact_format") != artifact_format:
        raise ValueError(
            f"Unsupported artifact format {payload.get('artifact_format')!r}; "
            f"expected {artifact_format!r}."
        )
    if payload.get("source_revision") != source_revision:
        raise ValueError("Artifact source revision does not match vendored Motius source.")
    return payload


def ensure_outputs_path(path: str | Path, *, method: str) -> Path:
    """Require generated files to remain under the caller project's outputs/."""

    output = Path(path).expanduser().resolve()
    try:
        output.relative_to(project_root() / "outputs")
    except ValueError as exc:
        raise ValueError(f"{method} output_root must live under outputs/.") from exc
    return output


def resolve_python(executable: str | Path) -> str:
    """Resolve the interpreter used by the method's isolated environment."""

    candidate = Path(executable).expanduser()
    if candidate.is_file():
        return os.path.abspath(candidate)
    resolved = shutil.which(str(candidate))
    if resolved is None:
        raise FileNotFoundError(f"Python executable not found: {executable}")
    return os.path.abspath(resolved)


def verify_manifest_assets(
    artifact_root: str | Path,
    manifest: Mapping[str, object],
) -> dict[str, str]:
    """Verify every manifest-declared asset and return its computed digest."""

    root = Path(artifact_root)
    expected = manifest.get("sha256")
    if not isinstance(expected, Mapping):
        raise ValueError("Artifact manifest must contain a sha256 mapping.")
    actual: dict[str, str] = {}
    for relative, digest in expected.items():
        path = root / str(relative)
        if not path.is_file():
            raise FileNotFoundError(f"Artifact asset is missing: {path}")
        value = sha256_file(path)
        if value != digest:
            raise RuntimeError(
                f"Artifact SHA256 mismatch for {relative}: expected {digest}, found {value}."
            )
        actual[str(relative)] = value
    return actual


def process_env(
    *,
    vendor_root: str | Path,
    artifact_root: str | Path,
    method_prefix: str,
) -> dict[str, str]:
    """Build an isolated subprocess environment using only Motius source."""

    env = os.environ.copy()
    env[f"MOTIUS_{method_prefix}_ARTIFACT_ROOT"] = str(Path(artifact_root))
    env["PYTHONPATH"] = os.pathsep.join(
        (str(Path(vendor_root).resolve()), str(PACKAGE_ROOT))
    )
    return env


__all__ = [
    "PACKAGE_ROOT",
    "ensure_outputs_path",
    "load_manifest",
    "process_env",
    "project_root",
    "resolve_artifact",
    "resolve_python",
    "sha256_file",
    "verify_manifest_assets",
]
