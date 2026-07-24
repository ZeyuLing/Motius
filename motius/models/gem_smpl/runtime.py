"""Pinned external runtime contract for NVIDIA GEM-SMPL (formerly GENMO)."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Optional, Sequence


SOURCE_REPOSITORY = "https://github.com/NVlabs/GENMO.git"
SOURCE_REVISION = "16bebf402d8893184249ee206d957b8248cd8310"
HF_REPOSITORY = "nvidia/GEM-X"
HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
CHECKPOINT_FILENAME = "gem_smpl.ckpt"
CHECKPOINT_SHA256 = "1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a"
OFFICIAL_OUTPUT_FILENAME = "smpl_params.pt"


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Hash a checkpoint without loading it into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: str | Path) -> Path:
    """Require the exact official GEM-SMPL checkpoint."""

    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"GEM-SMPL checkpoint not found: {checkpoint}. "
            "Run models/gem_smpl/setup_runtime.sh with DOWNLOAD_WEIGHTS=1."
        )
    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            "GEM-SMPL checkpoint SHA-256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {actual}."
        )
    return checkpoint


def verify_runtime_checkout(runtime_root: str | Path) -> Path:
    """Reject mutable or wrong upstream checkouts."""

    root = Path(runtime_root).expanduser().resolve()
    if not (root / "scripts" / "demo" / "demo_smpl_hpe.py").is_file():
        raise FileNotFoundError(f"Official GEM-SMPL runtime is incomplete: {root}")
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = completed.stdout.strip()
    if actual != SOURCE_REVISION:
        raise RuntimeError(
            f"GEM-SMPL runtime must be at {SOURCE_REVISION}, got {actual}."
        )
    return root


def runtime_python(
    runtime_root: str | Path,
    python_executable: Optional[str | Path] = None,
) -> Path:
    """Resolve only the interpreter belonging to the GEM-SMPL environment."""

    python = (
        Path(python_executable).expanduser()
        if python_executable is not None
        else Path(runtime_root).expanduser() / ".venv" / "bin" / "python"
    ).absolute()
    if not python.is_file():
        raise FileNotFoundError(f"GEM-SMPL isolated Python not found: {python}")
    return python


def expected_output_path(video: str | Path, output_root: str | Path) -> Path:
    """Return the path hard-coded by the fixed official demo."""

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
    """Build the fixed-revision official, parameter-only demo command."""

    root = Path(runtime_root).expanduser().resolve()
    command = [
        str(runtime_python(root, python_executable)),
        str(root / "scripts" / "demo" / "demo_smpl_hpe.py"),
        "--video",
        str(Path(video).expanduser().resolve()),
        "--ckpt_path",
        str(Path(checkpoint).expanduser().resolve()),
        "--output_root",
        str(Path(output_root).expanduser().resolve()),
        "--no_render",
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
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "build_demo_command",
    "expected_output_path",
    "runtime_python",
    "sha256_file",
    "verify_checkpoint",
    "verify_runtime_checkout",
]
