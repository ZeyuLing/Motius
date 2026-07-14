"""Motius wrapper for NVIDIA MotionBricks checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import sys
import importlib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


_REPO_ROOT = Path(__file__).resolve().parents[3]
_NETWORK_ROOT = Path(__file__).resolve().parent / "network"
_DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "motionbricks"
_ARTIFACT_FORMAT = "motius-motionbricks-wrapper-v1"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    if "/" not in name_or_path:
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _copy_tree(src: Path, dst: Path, copy_mode: str = "copy") -> None:
    if not src.exists():
        raise FileNotFoundError(f"MotionBricks artifact source does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_function = shutil.copy2
    if copy_mode == "hardlink":
        def copy_function(src, dst):
            try:
                os.link(Path(src).resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
    elif copy_mode != "copy":
        raise ValueError(f"Unsupported MotionBricks artifact copy mode: {copy_mode}")
    shutil.copytree(src, dst, symlinks=False, copy_function=copy_function)


def _is_local_relative_path(value: Optional[str]) -> bool:
    if not value:
        return False
    path = str(value)
    return "://" not in path and not Path(path).is_absolute()


def _resolve_relative_path(value: Optional[str], base: Path) -> Optional[str]:
    if not _is_local_relative_path(value):
        return value
    return str(base / str(value))


def _register_motionbricks_alias() -> None:
    """Expose the vendored runtime under the package name used in Hydra configs."""
    if "motionbricks" not in sys.modules:
        sys.modules["motionbricks"] = importlib.import_module("motius.models.motionbricks.network")


@dataclass(frozen=True)
class MotionBricksCheckpointLayout:
    """Official MotionBricks checkpoint files expected by the runtime."""

    root: Path
    clip: Path
    vqvae: Path
    pose: Path
    root_model: Path

    @classmethod
    def from_root(cls, root: Path) -> "MotionBricksCheckpointLayout":
        return cls(
            root=root,
            clip=root / "G1-clip.ckpt",
            vqvae=root / "motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt",
            pose=root / "motionbricks_pose/version_1/checkpoints/model-step=2000000.ckpt",
            root_model=root / "motionbricks_root/version_1/checkpoints/model-step=2000000.ckpt",
        )

    def missing(self) -> list[Path]:
        return [path for path in (self.clip, self.vqvae, self.pose, self.root_model) if not path.is_file()]

    def lfs_pointers(self) -> list[Path]:
        pointers: list[Path] = []
        for path in (self.clip, self.vqvae, self.pose, self.root_model):
            if not path.is_file() or path.stat().st_size > 4096:
                continue
            try:
                if path.read_text(errors="ignore").startswith("version https://git-lfs.github.com/spec/v1"):
                    pointers.append(path)
            except OSError:
                pass
        return pointers


@MODEL_BUNDLES.register_module()
class MotionBricksBundle(ModelBundle):
    """Own MotionBricks G1 runtime paths and lazily construct the demo agent.

    MotionBricks ships three learned components: a VQVAE motion tokenizer, a
    pose model, and a root model. The official release keeps their checkpoints
    in a Git LFS `out/` directory. Motius vendors the Apache-2.0 source runtime
    but intentionally does not vendor the multi-GB pretrained weights.
    """

    SUPPORTED_TASKS = {
        "g1_realtime_navigation": "real-time G1 locomotion controlled by WASD/random primitives",
        "g1_qpos_generation": "stream MotionBricks runtime frames as Unitree G1 qpos-36",
    }

    def __init__(
        self,
        checkpoint_dir: str | Path | None = None,
        device: str = "cuda",
        controller: str = "wasd",
        clips: str = "G1",
        load_model: bool = False,
        **kwargs: Any,
    ):
        super().__init__()
        self.checkpoint_dir = Path(checkpoint_dir or _DEFAULT_CHECKPOINT_DIR)
        self.device_name = str(device)
        self.controller = str(controller)
        self.clips = str(clips)
        self.runtime_kwargs = dict(kwargs)
        self.network_root = _NETWORK_ROOT
        self.asset_root = _NETWORK_ROOT / "assets"
        self.layout = MotionBricksCheckpointLayout.from_root(self.checkpoint_dir)
        self._agent = None
        if load_model:
            self.load_model()

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        config = dict(kwargs)
        config.setdefault("checkpoint_dir", pretrained_model_name_or_path)
        return config

    @property
    def representation(self) -> str:
        return "motionbricks_g1_414"

    @property
    def fps(self) -> int:
        return 30

    @property
    def required_checkpoint_files(self) -> MotionBricksCheckpointLayout:
        return self.layout

    def validate_checkpoints(self) -> None:
        missing = self.layout.missing()
        pointers = self.layout.lfs_pointers()
        if missing or pointers:
            details = []
            if missing:
                details.append("missing: " + ", ".join(str(path) for path in missing))
            if pointers:
                details.append("git-lfs pointers: " + ", ".join(str(path) for path in pointers))
            raise FileNotFoundError(
                "MotionBricks pretrained files are not ready under "
                f"{self.checkpoint_dir}. Fetch them with "
                "`git lfs pull --include=\"motionbricks/out/**\" --exclude=\"\"` "
                "from the official GR00T-WholeBodyControl checkout, then copy or "
                "symlink motionbricks/out to checkpoints/motionbricks. "
                + " | ".join(details)
            )

    def _runtime_args(self, **overrides: Any) -> SimpleNamespace:
        args = {
            "humanoid_scene_xml": str(self.asset_root / "skeletons/g1/scene_29dof.xml"),
            "skeleton_xml": str(self.asset_root / "skeletons/g1/g1.xml"),
            "result_dir": str(self.checkpoint_dir),
            "data_root": str(self.checkpoint_dir / "datasets"),
            "explicit_dataset_folder": None,
            "clips_ckpt": str(self.layout.clip),
            "reprocess_clips": 0,
            "controller": self.controller,
            "lookat_movement_direction": 0,
            "has_viewer": 0,
            "pre_filter_qpos": 1,
            "source_root_realignment": 1,
            "target_root_realignment": 1,
            "force_canonicalization": 1,
            "skip_ending_target_cond": 0,
            "random_speed_scale": 0,
            "speed_scale": [1.0, 1.0],
            "generate_dt": 2.0,
            "max_steps": 120,
            "random_seed": 1234,
            "num_runs": 1,
            "use_qpos": 1,
            "planner": "default",
            "allowed_mode": None,
            "clips": self.clips,
            "return_model_configs": True,
            "return_dataloader": True,
            "recording_dir": None,
            "EXP": "default",
        }
        args.update(self.runtime_kwargs)
        args.update(overrides)
        return SimpleNamespace(**args)

    def load_model(self, **overrides: Any):
        self.validate_checkpoints()
        if self._agent is not None and not overrides:
            return self._agent
        _register_motionbricks_alias()
        try:
            from motius.models.motionbricks.network.motion_backbone.demo.utils import navigation_demo
        except ImportError as exc:
            raise ImportError(
                "MotionBricks requires optional runtime dependencies. Install with "
                "`pip install -e '.[motionbricks]'`."
            ) from exc
        self._agent = navigation_demo(self._runtime_args(**overrides))
        return self._agent

    def save_pretrained(
        self,
        save_directory: str,
        *,
        include_checkpoint: bool = True,
        checkpoint_source: str | Path | None = None,
        checkpoint_subdir: str = "motionbricks_checkpoint",
        copy_mode: str = "copy",
        **kwargs: Any,
    ):
        """Save a HuggingFace-style Motius MotionBricks artifact."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "device": self.device_name,
            "controller": self.controller,
            "clips": self.clips,
            "runtime_kwargs": self.runtime_kwargs,
        }
        artifacts = {}
        if include_checkpoint:
            src = Path(checkpoint_source or self.checkpoint_dir)
            _copy_tree(src, save_dir / checkpoint_subdir, copy_mode=copy_mode)
            config["checkpoint_dir"] = checkpoint_subdir
            artifacts["motionbricks_checkpoint"] = checkpoint_subdir

        meta = {
            "model_type": "motionbricks",
            "format": _ARTIFACT_FORMAT,
            "bundle_class": "motius.models.motionbricks.MotionBricksBundle",
            "pipeline_class": "motius.pipelines.motionbricks.MotionBricksPipeline",
            "artifacts": artifacts,
            "config": config,
            "source": {
                "repo": "NVlabs/GR00T-WholeBodyControl",
                "subdir": "motionbricks",
                "paper": "https://arxiv.org/abs/2604.24833",
            },
        }
        (save_dir / "motionbricks_config.json").write_text(json.dumps(meta, indent=2) + "\n")
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MotionBricksPipeline",
                    "_library_name": "motius",
                    "model_type": "motionbricks",
                    "format": _ARTIFACT_FORMAT,
                    "bundle_class": meta["bundle_class"],
                    "pipeline_class": meta["pipeline_class"],
                    "artifacts": artifacts,
                    "supported_tasks": self.SUPPORTED_TASKS,
                    "api": {
                        "from_pretrained": "motius.pipelines.motionbricks.MotionBricksPipeline.from_pretrained",
                    },
                },
                indent=2,
            )
            + "\n"
        )
        readme = save_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "---\n"
                "library_name: motius\n"
                "tags:\n"
                "- motion-generation\n"
                "- robotics\n"
                "- unitree-g1\n"
                "- motionbricks\n"
                "license: other\n"
                "---\n\n"
                "# MotionBricks Motius Artifact\n\n"
                "This artifact stores the official MotionBricks checkpoint "
                "layout for the Motius `MotionBricksPipeline` wrapper.\n\n"
                "```python\n"
                "from motius.pipelines.motionbricks import MotionBricksPipeline\n\n"
                "pipe = MotionBricksPipeline.from_pretrained(\"<artifact-path-or-repo>\", bundle_kwargs={\"device\": \"cuda\", \"controller\": \"random\"})\n"
                "out = pipe.rollout(steps=240)\n"
                "qpos = out[\"qpos\"]\n"
                "```\n"
            )
        return save_directory

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        base_dir = None
        if isinstance(cfg, (str, Path)):
            cfg_path = Path(cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "motionbricks_config.json"
            base_dir = cfg_path.parent
            cfg = _read_json(cfg_path)
        cfg_dict = cls._to_plain_dict(cfg)
        if cfg_dict.get("model_type") == "motionbricks" and "config" in cfg_dict:
            cfg_dict = dict(cfg_dict["config"])
        runtime_kwargs = cfg_dict.pop("runtime_kwargs", None) or {}
        cfg_dict.update(runtime_kwargs)
        cfg_dict.pop("format", None)
        cfg_dict.pop("artifacts", None)
        if base_dir is not None:
            cfg_dict["checkpoint_dir"] = _resolve_relative_path(
                cfg_dict.get("checkpoint_dir"),
                base_dir,
            )
        return super().from_config(cfg_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(str(pretrained_model_name_or_path))
        if not (path / "motionbricks_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "motionbricks_config.json"
        if cfg_file.exists():
            return cls.from_config(cfg_file, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)


__all__ = ["MotionBricksBundle", "MotionBricksCheckpointLayout"]
