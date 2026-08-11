"""Repository-native GEM-SMPL model bundle."""

from motius.models.gem_smpl.bundle import (
    GEM_SMPL_ARTIFACT_FORMAT,
    GEM_SMPL_CHECKPOINT_SHA256,
    GEM_SMPL_REPO_ID,
    GemSmplBundle,
)
from motius.models.gem_smpl.runtime import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_SHA256,
    HF_REPOSITORY,
    HF_REVISION,
    SOURCE_REPOSITORY,
    SOURCE_REVISION,
    expected_output_path,
    verify_checkpoint,
    verify_runtime_checkout,
)

__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_SHA256",
    "GEM_SMPL_ARTIFACT_FORMAT",
    "GEM_SMPL_CHECKPOINT_SHA256",
    "GEM_SMPL_REPO_ID",
    "HF_REPOSITORY",
    "HF_REVISION",
    "GemSmplBundle",
    "SOURCE_REPOSITORY",
    "SOURCE_REVISION",
    "expected_output_path",
    "verify_checkpoint",
    "verify_runtime_checkout",
]
