"""GEM-SMPL external runtime metadata."""

from motius.models.gem_smpl.bundle import GemSmplBundle
from motius.models.gem_smpl.runtime import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    HF_REPOSITORY,
    HF_REVISION,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    build_demo_command,
    expected_output_path,
    verify_checkpoint,
    verify_runtime_checkout,
)

__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SHA256",
    "HF_REPOSITORY",
    "HF_REVISION",
    "GemSmplBundle",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "build_demo_command",
    "expected_output_path",
    "verify_checkpoint",
    "verify_runtime_checkout",
]
