"""Motius wrapper for NVIDIA MotionBricks checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


_REPO_ROOT = Path(__file__).resolve().parents[3]
_NETWORK_ROOT = Path(__file__).resolve().parent / "network"
_DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "motionbricks"


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
        try:
            from motius.models.motionbricks.network.motion_backbone.demo.utils import navigation_demo
        except ImportError as exc:
            raise ImportError(
                "MotionBricks requires optional runtime dependencies. Install with "
                "`pip install -e '.[motionbricks]'`."
            ) from exc
        self._agent = navigation_demo(self._runtime_args(**overrides))
        return self._agent


__all__ = ["MotionBricksBundle", "MotionBricksCheckpointLayout"]
