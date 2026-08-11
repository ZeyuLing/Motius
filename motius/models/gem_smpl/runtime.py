"""Public provenance constants for the repository-native GEM-SMPL runtime."""

from __future__ import annotations

from pathlib import Path

from motius.models.gem_runtime import sha256_file
from motius.models.gem_smpl.bundle import (
    GEM_SMPL_CHECKPOINT_SHA256,
    OFFICIAL_HF_REVISION,
    OFFICIAL_REPOSITORY,
    OFFICIAL_SOURCE_REVISION,
)


SOURCE_REPOSITORY = OFFICIAL_REPOSITORY
SOURCE_REVISION = OFFICIAL_SOURCE_REVISION
HF_REPOSITORY = "nvidia/GEM-X"
HF_REVISION = OFFICIAL_HF_REVISION
CHECKPOINT_FILENAME = "gem_smpl.ckpt"
CHECKPOINT_SHA256 = GEM_SMPL_CHECKPOINT_SHA256
OFFICIAL_OUTPUT_FILENAME = "smpl_params.pt"
VENDORED_RUNTIME_ROOT = Path(__file__).resolve().parent / "vendor"


def verify_checkpoint(path: str | Path) -> Path:
    checkpoint = Path(path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"GEM-SMPL checkpoint not found: {checkpoint}")
    actual = sha256_file(checkpoint)
    if actual != CHECKPOINT_SHA256:
        raise ValueError(
            "GEM-SMPL checkpoint SHA256 mismatch: "
            f"expected {CHECKPOINT_SHA256}, got {actual}."
        )
    return checkpoint


def verify_runtime_checkout(runtime_root: str | Path = VENDORED_RUNTIME_ROOT) -> Path:
    """Validate Motius-owned source; external checkouts are not accepted."""

    root = Path(runtime_root).expanduser().resolve()
    if root != VENDORED_RUNTIME_ROOT.resolve():
        raise ValueError(
            "GEM-SMPL no longer accepts an external runtime checkout; "
            "use GemSmplPipeline.from_pretrained(...)."
        )
    if not (root / "scripts/demo/demo_smpl_hpe.py").is_file():
        raise FileNotFoundError(f"Vendored GEM-SMPL source is incomplete: {root}")
    return root


def expected_output_path(video: str | Path, output_root: str | Path) -> Path:
    return Path(output_root).expanduser().resolve() / Path(video).stem / OFFICIAL_OUTPUT_FILENAME


__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SHA256",
    "HF_REPOSITORY",
    "HF_REVISION",
    "OFFICIAL_OUTPUT_FILENAME",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "VENDORED_RUNTIME_ROOT",
    "expected_output_path",
    "sha256_file",
    "verify_checkpoint",
    "verify_runtime_checkout",
]
