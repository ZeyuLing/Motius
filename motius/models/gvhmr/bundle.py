"""Isolated launcher for the unmodified official GVHMR demo runtime."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

import numpy as np

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


OFFICIAL_REPOSITORY = "https://github.com/zju3dv/GVHMR"
OFFICIAL_RUNTIME_REVISION = "6ec3ca39336c50492c0fae65fba2fb831fc7d866"
_OFFICIAL_CHECKPOINT = Path(
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
)
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_RUNTIME_ROOT = _REPO_ROOT / "outputs" / "tmp" / "gvhmr" / "upstream"


def _bbox_xys_from_xyxy(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Match GVHMR's official 192:256, 1.2x bbox conversion."""

    boxes = np.asarray(bbox_xyxy, dtype=np.float32)
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 4
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, 2:] <= boxes[:, :2])
    ):
        raise ValueError("bbox_xyxy must have finite shape (frames,4) and positive area.")
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    aspect_ratio = 192.0 / 256.0
    wide = width > aspect_ratio * height
    height = height.copy()
    width = width.copy()
    height[wide] = width[wide] / aspect_ratio
    tall = width < aspect_ratio * height
    width[tall] = height[tall] * aspect_ratio
    size = np.maximum(height, width) * 1.2
    return np.concatenate((center, size[:, None]), axis=1).astype(np.float32)


def sha256_file(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Hash a local checkpoint without loading or deserializing it."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


@MODEL_BUNDLES.register_module()
class GVHMRBundle(ModelBundle):
    """Configuration and subprocess boundary for official GVHMR inference.

    The model code and weights stay in the separately installed official
    runtime. This bundle never imports ``hmr4d`` into the Motius process.
    """

    def __init__(
        self,
        *,
        runtime_root: Optional[str | Path] = None,
        python_executable: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.runtime_root = Path(
            runtime_root
            or os.environ.get("MOTIUS_GVHMR_ROOT", _DEFAULT_RUNTIME_ROOT)
        ).expanduser().resolve()
        self.python_executable = (
            str(python_executable)
            if python_executable is not None
            else os.environ.get("MOTIUS_GVHMR_PYTHON")
        )
        self.checkpoint_path = self.runtime_root / _OFFICIAL_CHECKPOINT

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        **kwargs,
    ) -> dict:
        return {"runtime_root": pretrained_model_name_or_path, **kwargs}

    @property
    def checkpoint_sha256(self) -> str:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                "Official GVHMR checkpoint not found at "
                f"{self.checkpoint_path}. Follow GVHMR docs/INSTALL.md and place "
                f"the release checkpoint at {_OFFICIAL_CHECKPOINT}."
            )
        return sha256_file(self.checkpoint_path)

    def _resolved_python(self) -> str:
        if not self.python_executable:
            raise RuntimeError(
                "Set MOTIUS_GVHMR_PYTHON (or python_executable) to the Python "
                "binary from the isolated GVHMR environment."
            )
        executable = str(Path(self.python_executable).expanduser())
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"GVHMR Python executable not found: {self.python_executable}"
            )
        return resolved

    def validate_runtime(self, *, require_checkpoint: bool = True) -> None:
        demo = self.runtime_root / "tools" / "demo" / "demo.py"
        if not demo.is_file():
            raise FileNotFoundError(
                f"Official GVHMR runtime not found at {self.runtime_root}. "
                "Run tools/setup_gvhmr_env.sh or provide MOTIUS_GVHMR_ROOT."
            )
        revision = subprocess.run(
            ["git", "-C", str(self.runtime_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != OFFICIAL_RUNTIME_REVISION:
            raise RuntimeError(
                "GVHMR runtime revision mismatch: wrapped code requires "
                f"{OFFICIAL_RUNTIME_REVISION}, found {revision}."
            )
        self._resolved_python()
        if require_checkpoint:
            _ = self.checkpoint_sha256

    def run_official_demo(
        self,
        video: str | Path,
        output_root: str | Path,
        *,
        static_camera: bool = False,
        use_dpvo: bool = False,
        focal_length_mm: Optional[int] = None,
        verbose: bool = False,
        bbox_xyxy: Optional[np.ndarray] = None,
    ) -> Path:
        """Run the pinned official demo and return its ``hmr4d_results.pt``."""

        self.validate_runtime(require_checkpoint=True)
        video_path = Path(video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        if focal_length_mm is not None and int(focal_length_mm) <= 0:
            raise ValueError("focal_length_mm must be positive.")

        run_root = (
            Path(output_root).expanduser().resolve()
            / f"motius_{video_path.stem}_{uuid.uuid4().hex}"
        )
        run_root.mkdir(parents=True, exist_ok=False)
        if bbox_xyxy is not None:
            import torch

            boxes = np.asarray(bbox_xyxy, dtype=np.float32)
            bbox_cache = run_root / video_path.stem / "preprocess" / "bbx.pt"
            bbox_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "bbx_xyxy": torch.from_numpy(boxes),
                    "bbx_xys": torch.from_numpy(_bbox_xys_from_xyxy(boxes)),
                },
                bbox_cache,
            )
        command = [
            self._resolved_python(),
            str(self.runtime_root / "tools" / "demo" / "demo.py"),
            f"--video={video_path}",
            f"--output_root={run_root}",
        ]
        if static_camera:
            command.append("--static_cam")
        if use_dpvo:
            command.append("--use_dpvo")
        if focal_length_mm is not None:
            command.append(f"--f_mm={int(focal_length_mm)}")
        if verbose:
            command.append("--verbose")
        subprocess.run(command, cwd=self.runtime_root, check=True)

        result = run_root / video_path.stem / "hmr4d_results.pt"
        if not result.is_file():
            raise RuntimeError(
                "Official GVHMR demo completed without producing "
                f"the documented result file: {result}"
            )
        return result

    def convert_official_result(
        self,
        result_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Materialize SMPL joints/vertices inside the isolated runtime."""

        self.validate_runtime(require_checkpoint=True)
        source = Path(result_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GVHMR result not found: {source}")
        destination = (
            Path(output_path).expanduser().resolve()
            if output_path is not None
            else source.with_name("motius_monocular_capture.npz")
        )
        converter = _REPO_ROOT / "tools" / "convert_gvhmr_results.py"
        subprocess.run(
            [
                self._resolved_python(),
                str(converter),
                f"--input={source}",
                f"--output={destination}",
                f"--checkpoint={self.checkpoint_path}",
                f"--runtime-root={self.runtime_root}",
                f"--runtime-revision={OFFICIAL_RUNTIME_REVISION}",
            ],
            cwd=self.runtime_root,
            check=True,
        )
        if not destination.is_file():
            raise RuntimeError(
                f"GVHMR result converter did not produce {destination}."
            )
        return destination

    def forward(self, video: str | Path, output_root: str | Path, **kwargs) -> Path:
        return self.run_official_demo(video, output_root, **kwargs)


__all__ = [
    "GVHMRBundle",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_RUNTIME_REVISION",
    "sha256_file",
]
