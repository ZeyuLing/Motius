"""GEM-X / SOMA-77 external runtime metadata."""

from motius.models.gem_x.bundle import GemXBundle
from motius.models.gem_x.runtime import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    HF_REPOSITORY,
    HF_REVISION,
    SOMA_SOURCE_REVISION,
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
    "GemXBundle",
    "SOMA_SOURCE_REVISION",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "build_demo_command",
    "expected_output_path",
    "verify_checkpoint",
    "verify_runtime_checkout",
]
