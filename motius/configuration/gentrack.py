"""Self-describing metadata for the GenTrack specialization of HYMotion."""

import json
from pathlib import Path


def specialize_artifact(directory):
    directory = Path(directory)
    classes = dict(
        pipeline_class="motius.pipelines.gentrack.GenTrackPipeline",
        bundle_class="motius.models.gentrack.GenTrackG1Bundle",
    )
    for filename in ["hymotion_t2m_config.json", "model_index.json"]:
        path = directory / filename
        metadata = json.loads(path.read_text())
        metadata.update(classes)
        metadata.update(
            _class_name="GenTrackPipeline",
            tasks=["text_to_motion"],
            specialization="gentrack_g1",
        )
        if "api" in metadata:
            metadata["api"] = {
                key: classes["bundle_class"] + "." + key
                for key in ["from_pretrained", "from_config"]
            }
        path.write_text(json.dumps(metadata, indent=2) + "\n")


import os
from pathlib import Path


MOTIUS_ROOT = Path(__file__).resolve().parents[2]


def _path_from_env(primary: str, legacy: str, default: Path) -> Path:
    value = os.environ.get(primary) or os.environ.get(legacy)
    return Path(value).expanduser().absolute() if value else default.absolute()


PROTOMOTIONS_ROOT = _path_from_env(
    "MOTIUS_GENTRACK_PROTOMOTIONS_ROOT",
    "PHYSFLOW_PROTOMOTIONS_ROOT",
    MOTIUS_ROOT,
)
PROTOMOTIONS_ASSETS_ROOT = MOTIUS_ROOT / "checkpoints" / "robots" / "protomotions"

PROTOMOTIONS_G1_TRACKER_ROOT = _path_from_env(
    "MOTIUS_GENTRACK_PROTOMOTIONS_ARTIFACT",
    "PHYSFLOW_PROTOMOTIONS_ARTIFACT",
    MOTIUS_ROOT / "checkpoints" / "models" / "protomotions",
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
    PROTOMOTIONS_ASSETS_ROOT
    / "mjcf"
    / "g1_holo_compat.xml",
)
PROTOMOTIONS_G1_URDF = _path_from_env(
    "MOTIUS_GENTRACK_G1_URDF",
    "PHYSFLOW_G1_URDF",
    PROTOMOTIONS_ASSETS_ROOT
    / "urdf"
    / "for_retargeting"
    / "g1.urdf",
)
PROTOMOTIONS_G1_MESH_DIR = (
    PROTOMOTIONS_ASSETS_ROOT
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
