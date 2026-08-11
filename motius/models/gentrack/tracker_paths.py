"""Canonical runtime paths for GenTrack's physical judge backends.

The public repository ships the tracker source snapshots but not third-party
pretrained weights.  Every artifact path can therefore be overridden without
editing code.  Legacy ``PHYSFLOW_*`` variables remain accepted so checkpoints
and launch manifests produced by the research code stay reproducible.
"""

from __future__ import annotations

import os
from pathlib import Path


MOTIUS_ROOT = Path(__file__).resolve().parents[3]


def _path_from_env(primary: str, legacy: str, default: Path) -> Path:
    value = os.environ.get(primary) or os.environ.get(legacy)
    return Path(value).expanduser().absolute() if value else default.absolute()


PROTOMOTIONS_ROOT = _path_from_env(
    "MOTIUS_GENTRACK_PROTOMOTIONS_ROOT",
    "PHYSFLOW_PROTOMOTIONS_ROOT",
    MOTIUS_ROOT / "motius" / "trainers" / "protomotions" / "vendor",
)
PROTOMOTIONS_G1_TRACKER_ROOT = _path_from_env(
    "MOTIUS_GENTRACK_PROTOMOTIONS_ARTIFACT",
    "PHYSFLOW_PROTOMOTIONS_ARTIFACT",
    PROTOMOTIONS_ROOT
    / "data"
    / "pretrained_models"
    / "motion_tracker"
    / "g1-bones-deploy",
)
PROTOMOTIONS_G1_ONNX = _path_from_env(
    "MOTIUS_GENTRACK_PROTOMOTIONS_ONNX",
    "PHYSFLOW_PROTOMOTIONS_ONNX",
    PROTOMOTIONS_G1_TRACKER_ROOT / "compiled_models" / "unified_pipeline.onnx",
)
PROTOMOTIONS_G1_YAML = PROTOMOTIONS_G1_ONNX.with_suffix(".yaml")
PROTOMOTIONS_G1_CKPT = _path_from_env(
    "MOTIUS_GENTRACK_PROTOMOTIONS_CHECKPOINT",
    "PHYSFLOW_PROTOMOTIONS_CHECKPOINT",
    PROTOMOTIONS_G1_TRACKER_ROOT / "last.ckpt",
)
PROTOMOTIONS_G1_MJCF = _path_from_env(
    "MOTIUS_GENTRACK_G1_MJCF",
    "PHYSFLOW_G1_MJCF",
    PROTOMOTIONS_ROOT
    / "protomotions"
    / "data"
    / "assets"
    / "mjcf"
    / "g1_holo_compat.xml",
)
PROTOMOTIONS_G1_URDF = _path_from_env(
    "MOTIUS_GENTRACK_G1_URDF",
    "PHYSFLOW_G1_URDF",
    PROTOMOTIONS_ROOT
    / "protomotions"
    / "data"
    / "assets"
    / "urdf"
    / "for_retargeting"
    / "g1.urdf",
)
PROTOMOTIONS_G1_MESH_DIR = (
    PROTOMOTIONS_ROOT
    / "protomotions"
    / "data"
    / "assets"
    / "mesh"
    / "G1"
)

ANY2TRACK_ROOT = MOTIUS_ROOT / "motius" / "models" / "any2track"
ANY2TRACK_ONNX = _path_from_env(
    "MOTIUS_GENTRACK_ANY2TRACK_ONNX",
    "PHYSFLOW_ANY2TRACK_ONNX",
    ANY2TRACK_ROOT / "model.onnx",
)
ANY2TRACK_CONFIG = _path_from_env(
    "MOTIUS_GENTRACK_ANY2TRACK_CONFIG",
    "PHYSFLOW_ANY2TRACK_CONFIG",
    ANY2TRACK_ROOT / "config.json",
)
ANY2TRACK_CHECKPOINT_CONFIG = ANY2TRACK_CONFIG
ANY2TRACK_G1_MJCF = PROTOMOTIONS_G1_MJCF

HUMANOID_GPT_ROOT = MOTIUS_ROOT / "motius" / "models" / "humanoid_gpt"
HUMANOID_GPT_ONNX = _path_from_env(
    "MOTIUS_GENTRACK_HUMANOID_GPT_ONNX",
    "PHYSFLOW_HGPT_ONNX",
    HUMANOID_GPT_ROOT / "policy.onnx",
)
HUMANOID_GPT_VENV_PYTHON = _path_from_env(
    "MOTIUS_GENTRACK_HUMANOID_GPT_PYTHON",
    "PHYSFLOW_HGPT_PYTHON",
    Path(os.sys.executable),
)


def resolve_project_path(
    path: str | Path | None,
    default: Path,
    project_root: Path = MOTIUS_ROOT,
) -> Path:
    candidate = Path(path) if path else default
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.expanduser().absolute()


TRACKERS_ROOT = MOTIUS_ROOT / "motius" / "models"
