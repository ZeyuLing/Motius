"""Pinned external runtime contract for NVIDIA GEM-X / SOMA-77."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Sequence


SOURCE_REPOSITORY = "https://github.com/NVlabs/GEM-X.git"
SOURCE_REVISION = "32992550dba114c62243fb55e361311972dce8f9"
SOMA_SOURCE_REVISION = "e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"
HF_REPOSITORY = "nvidia/GEM-X"
HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
CHECKPOINT_FILENAME = "gem_soma.ckpt"
CHECKPOINT_SHA256 = "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
OFFICIAL_OUTPUT_FILENAME = "hpe_results.pt"


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a checkpoint without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path) -> Path:
    """Require the exact official GEM-X checkpoint."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"GEM-X checkpoint not found: {checkpoint}. "
            "Run models/gem_x/setup_runtime.sh with DOWNLOAD_WEIGHTS=1."
        )
    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            "GEM-X checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {actual}."
        )
    return checkpoint


def _git_revision(root: Path, relative_path: str = ".") -> str:
    completed = subprocess.run(
        ["git", "-C", str(root / relative_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def verify_runtime_checkout(runtime_root: str | Path) -> Path:
    """Reject wrong GEM-X or SOMA submodule revisions."""

    root = Path(runtime_root).expanduser().resolve()
    if not (root / "scripts" / "demo" / "demo_soma.py").is_file():
        raise FileNotFoundError(f"Official GEM-X runtime is incomplete: {root}")
    actual = _git_revision(root)
    if actual != SOURCE_REVISION:
        raise RuntimeError(f"GEM-X runtime must be at {SOURCE_REVISION}, got {actual}.")
    soma_root = root / "third_party" / "soma"
    if not soma_root.is_dir():
        raise FileNotFoundError(f"GEM-X SOMA submodule is missing: {soma_root}")
    soma_actual = _git_revision(root, "third_party/soma")
    if soma_actual != SOMA_SOURCE_REVISION:
        raise RuntimeError(
            f"SOMA runtime must be at {SOMA_SOURCE_REVISION}, got {soma_actual}."
        )
    return root


def runtime_python(
    runtime_root: str | Path,
    python_executable: Optional[str | Path] = None,
) -> Path:
    """Resolve only the interpreter belonging to the GEM-X environment."""

    python = (
        Path(python_executable).expanduser()
        if python_executable is not None
        else Path(runtime_root).expanduser() / ".venv" / "bin" / "python"
    ).absolute()
    if not python.is_file():
        raise FileNotFoundError(f"GEM-X isolated Python not found: {python}")
    return python


def expected_output_path(video: str | Path, output_root: str | Path) -> Path:
    """Return the output path from the fixed demo config."""

    return (
        Path(output_root).expanduser().resolve()
        / Path(video).stem
        / OFFICIAL_OUTPUT_FILENAME
    )


def build_demo_command(
    *,
    runtime_root: str | Path,
    video: str | Path,
    output_root: str | Path,
    checkpoint: str | Path,
    python_executable: Optional[str | Path] = None,
    static_camera: bool = False,
    extra_args: Sequence[str] = (),
) -> tuple[str, ...]:
    """Build the exact fixed-revision official GEM-X demo command."""

    root = Path(runtime_root).expanduser().resolve()
    command = [
        str(runtime_python(root, python_executable)),
        str(root / "scripts" / "demo" / "demo_soma.py"),
        "--video",
        str(Path(video).expanduser().resolve()),
        "--ckpt",
        str(Path(checkpoint).expanduser().resolve()),
        "--output_root",
        str(Path(output_root).expanduser().resolve()),
    ]
    if static_camera:
        command.append("--static_cam")
    command.extend(str(value) for value in extra_args)
    return tuple(command)


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SHA256",
    "HF_REPOSITORY",
    "HF_REVISION",
    "OFFICIAL_OUTPUT_FILENAME",
    "SOMA_SOURCE_REVISION",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "build_demo_command",
    "expected_output_path",
    "runtime_python",
    "sha256_file",
    "verify_checkpoint",
    "verify_runtime_checkout",
]
