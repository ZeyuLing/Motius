"""Repo-native, source-pinned GVHMR inference bundle."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


OFFICIAL_REPOSITORY = "https://github.com/zju3dv/GVHMR"
OFFICIAL_RUNTIME_REVISION = "6ec3ca39336c50492c0fae65fba2fb831fc7d866"
GVHMR_REPO_ID = "ZeyuLing/Motius-GVHMR"
GVHMR_ARTIFACT_FORMAT = "motius-gvhmr-v1"
_ARTIFACT_CONFIG = "gvhmr_config.json"
_OFFICIAL_CHECKPOINT = Path(
    "inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt"
)
_REQUIRED_PUBLIC_ASSETS = (
    _OFFICIAL_CHECKPOINT,
    Path("inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"),
    Path("inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    Path("inputs/checkpoints/yolo/yolov8x.pt"),
)
_BODY_MODELS = Path("inputs/checkpoints/body_models")
_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDORED_DEMO_MODULE = "motius.models.gvhmr.vendor.official_demo"
_VENDORED_SOURCE = Path(__file__).resolve().parent / "vendor" / "official_demo.py"


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


def sha256_file(path: str | Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    """Hash a local artifact without loading or deserializing it."""

    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_artifact(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    candidate = Path(pretrained_model_name_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=revision,
            cache_dir=None if cache_dir is None else str(cache_dir),
            token=token,
            local_files_only=local_files_only,
        )
    ).resolve()


def _ensure_output_root(path: str | Path) -> Path:
    output = Path(path).expanduser().resolve()
    try:
        output.relative_to((_REPO_ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("GVHMR output_root must live under repository outputs/.") from exc
    return output


@MODEL_BUNDLES.register_module()
class GVHMRBundle(ModelBundle):
    """GVHMR source, checkpoint, and isolated dependency boundary.

    Motius owns the source code in ``motius.models.gvhmr.vendor``. The artifact
    root contains only official checkpoint assets and user-supplied licensed
    body models. A subprocess remains useful because the pinned GVHMR package
    requires a tightly constrained CUDA/PyTorch environment; it does not imply
    a dependency on another source checkout.
    """

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        python_executable: Optional[str | Path] = None,
        body_models_root: Optional[str | Path] = None,
        manifest: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.python_executable = str(
            python_executable
            or os.environ.get("MOTIUS_GVHMR_PYTHON")
            or sys.executable
        )
        self.body_models_root = Path(
            body_models_root or self.artifact_root / _BODY_MODELS
        ).expanduser().resolve()
        self.manifest = dict(manifest or self._load_manifest())
        self.checkpoint_path = self.artifact_root / _OFFICIAL_CHECKPOINT

    def _load_manifest(self) -> dict:
        config = self.artifact_root / _ARTIFACT_CONFIG
        if not config.is_file():
            raise FileNotFoundError(
                f"GVHMR artifact is missing {_ARTIFACT_CONFIG}: {self.artifact_root}"
            )
        payload = json.loads(config.read_text())
        if payload.get("artifact_format") != GVHMR_ARTIFACT_FORMAT:
            raise ValueError(
                "Unsupported GVHMR artifact format "
                f"{payload.get('artifact_format')!r}."
            )
        if payload.get("source_revision") != OFFICIAL_RUNTIME_REVISION:
            raise ValueError("GVHMR artifact source revision does not match Motius.")
        return payload

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "GVHMRBundle":
        root = _resolve_artifact(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        return cls(artifact_root=root, **kwargs)

    @property
    def checkpoint_sha256(self) -> str:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"GVHMR checkpoint not found: {self.checkpoint_path}"
            )
        expected = (
            self.manifest.get("sha256", {})
            if isinstance(self.manifest.get("sha256"), Mapping)
            else {}
        ).get(_OFFICIAL_CHECKPOINT.as_posix())
        actual = sha256_file(self.checkpoint_path)
        if expected is not None and actual != expected:
            raise RuntimeError(
                "GVHMR checkpoint SHA256 mismatch: "
                f"expected {expected}, found {actual}."
            )
        return actual

    def _resolved_python(self) -> str:
        executable_path = Path(self.python_executable).expanduser()
        if executable_path.exists() or executable_path.parent != Path("."):
            executable = os.path.abspath(executable_path)
        else:
            executable = str(executable_path)
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError(
                f"GVHMR Python executable not found: {self.python_executable}"
            )
        return resolved

    def validate_runtime(
        self,
        *,
        require_checkpoint: bool = True,
        require_body_models: bool = False,
    ) -> None:
        if not _VENDORED_SOURCE.is_file():
            raise FileNotFoundError(
                f"Vendored GVHMR source is missing: {_VENDORED_SOURCE}"
            )
        self._resolved_python()
        missing = [
            path.as_posix()
            for path in _REQUIRED_PUBLIC_ASSETS
            if not (self.artifact_root / path).is_file()
        ]
        if require_checkpoint and missing:
            raise FileNotFoundError(
                "GVHMR artifact is incomplete; missing: " + ", ".join(missing)
            )
        if require_checkpoint:
            _ = self.checkpoint_sha256
        if require_body_models and not self.body_models_root.is_dir():
            raise FileNotFoundError(
                "GVHMR geometry materialization requires licensed SMPL/SMPL-X "
                f"files under {self.body_models_root}."
            )

    def _subprocess_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["MOTIUS_GVHMR_ARTIFACT_ROOT"] = str(self.artifact_root)
        env["MOTIUS_GVHMR_BODY_MODELS_ROOT"] = str(self.body_models_root)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(_REPO_ROOT)
            if not pythonpath
            else f"{_REPO_ROOT}{os.pathsep}{pythonpath}"
        )
        return env

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
        parity_trace: Optional[str | Path] = None,
        render: bool = False,
    ) -> Path:
        """Run the source-pinned demo and return its ``hmr4d_results.pt``."""

        self.validate_runtime(
            require_checkpoint=True,
            require_body_models=True,
        )
        video_path = Path(video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        if focal_length_mm is not None and int(focal_length_mm) <= 0:
            raise ValueError("focal_length_mm must be positive.")

        output = _ensure_output_root(output_root)
        run_root = output / f"motius_{video_path.stem}_{uuid.uuid4().hex}"
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
        trace_path = (
            run_root / video_path.stem / "gvhmr_parity_trace.npz"
            if parity_trace is True
            else None if parity_trace in (None, False) else Path(parity_trace).resolve()
        )
        if trace_path is not None:
            _ensure_output_root(trace_path.parent)
            trace_path.parent.mkdir(parents=True, exist_ok=True)

        command = [
            self._resolved_python(),
            "-m",
            _VENDORED_DEMO_MODULE,
            f"--video={video_path}",
            f"--output_root={run_root}",
            f"--ckpt_path={self.checkpoint_path}",
        ]
        if static_camera:
            command.append("--static_cam")
        if use_dpvo:
            command.append("--use_dpvo")
        if focal_length_mm is not None:
            command.append(f"--f_mm={int(focal_length_mm)}")
        if verbose:
            command.append("--verbose")
        if trace_path is not None:
            command.append(f"--parity_trace={trace_path}")
        if render:
            command.append("--render")

        subprocess.run(
            command,
            cwd=self.artifact_root,
            env=self._subprocess_env(),
            check=True,
        )

        result = run_root / video_path.stem / "hmr4d_results.pt"
        if not result.is_file():
            raise RuntimeError(
                "GVHMR completed without producing its documented result: "
                f"{result}"
            )
        return result

    def convert_official_result(
        self,
        result_path: str | Path,
        output_path: Optional[str | Path] = None,
    ) -> Path:
        """Materialize SMPL joints and vertices with vendored GVHMR code."""

        self.validate_runtime(require_checkpoint=True, require_body_models=True)
        source = Path(result_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GVHMR result not found: {source}")
        destination = (
            source.with_name("motius_monocular_capture.npz")
            if output_path is None
            else Path(output_path).expanduser().resolve()
        )
        _ensure_output_root(destination.parent)
        converter = _REPO_ROOT / "tools" / "convert_gvhmr_results.py"
        subprocess.run(
            [
                self._resolved_python(),
                str(converter),
                f"--input={source}",
                f"--output={destination}",
                f"--checkpoint={self.checkpoint_path}",
                f"--artifact-root={self.artifact_root}",
                f"--runtime-revision={OFFICIAL_RUNTIME_REVISION}",
            ],
            cwd=self.artifact_root,
            env=self._subprocess_env(),
            check=True,
        )
        if not destination.is_file():
            raise RuntimeError(f"GVHMR converter did not produce {destination}.")
        return destination

    def save_pretrained(
        self,
        save_directory: str,
        *,
        source_assets: Optional[Mapping[str, str | Path]] = None,
    ) -> None:
        """Export exact official checkpoint bytes as one HF-style artifact."""

        output = Path(save_directory).expanduser().resolve()
        _ensure_output_root(output)
        output.mkdir(parents=True, exist_ok=True)
        assets = {
            path.as_posix(): (
                Path(source_assets[path.as_posix()]).expanduser().resolve()
                if source_assets and path.as_posix() in source_assets
                else self.artifact_root / path
            )
            for path in _REQUIRED_PUBLIC_ASSETS
        }
        digests = {}
        for relative, source in assets.items():
            if not source.is_file():
                raise FileNotFoundError(f"Missing GVHMR export asset: {source}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            digests[relative] = sha256_file(destination)

        config = {
            "artifact_format": GVHMR_ARTIFACT_FORMAT,
            "model_type": "gvhmr",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": OFFICIAL_RUNTIME_REVISION,
            "checkpoint": _OFFICIAL_CHECKPOINT.as_posix(),
            "assets": list(assets),
            "sha256": digests,
            "external_assets": {
                _BODY_MODELS.as_posix(): (
                    "Licensed SMPL and SMPL-X files supplied by the user."
                )
            },
        }
        (output / _ARTIFACT_CONFIG).write_text(
            json.dumps(config, indent=2) + "\n"
        )
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "GVHMRPipeline",
                    "_library_name": "motius",
                    "artifact_format": GVHMR_ARTIFACT_FORMAT,
                    "bundle_class": "motius.models.gvhmr.bundle.GVHMRBundle",
                    "pipeline_class": (
                        "motius.pipelines.gvhmr.pipeline.GVHMRPipeline"
                    ),
                    "tasks": ["monocular_motion_capture"],
                },
                indent=2,
            )
            + "\n"
        )
        package_root = Path(__file__).resolve().parent
        for filename, source in {
            "ATTRIBUTIONS.md": package_root / "ATTRIBUTIONS.md",
            "GVHMR_LICENSE": package_root / "vendor" / "GVHMR_LICENSE",
        }.items():
            shutil.copy2(source, output / filename)
        preview_sources = sorted(
            (_REPO_ROOT / "assets/model_zoo/gvhmr").glob("case_*_global.*")
        )
        for source in preview_sources:
            destination = output / "assets" / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        preview = (
            """| Clip 01 | Clip 02 | Clip 03 |
| --- | --- | --- |
| [![](assets/case_01_global.webp)](assets/case_01_global.mp4) | [![](assets/case_02_global.webp)](assets/case_02_global.mp4) | [![](assets/case_03_global.webp)](assets/case_03_global.mp4) |"""
            if preview_sources
            else (
                "See the full Motius model card for verified preview media."
            )
        )
        (output / "README.md").write_text(
            f"""---
library_name: motius
pipeline_tag: other
license: other
tags:
- monocular-motion-capture
- gvhmr
---

# Motius GVHMR

Repository-native, source-pinned [GVHMR]({OFFICIAL_REPOSITORY}) for monocular
camera-relative and world-grounded human motion recovery.

- Motius model card:
  <https://github.com/ZeyuLing/Motius/blob/main/docs/model_zoo/gvhmr.md>
- Pinned source revision: `{OFFICIAL_RUNTIME_REVISION}`
- GVHMR checkpoint SHA-256:
  `4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91`

```python
from motius.pipelines.gvhmr import GVHMRPipeline

pipeline = GVHMRPipeline.from_pretrained(
    "{GVHMR_REPO_ID}",
    bundle_kwargs={{
        "python_executable": "outputs/envs/gvhmr/bin/python",
        "body_models_root": "checkpoints/body_models",
    }},
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4",
    output_root="outputs/gvhmr/run_001",
)
```

## 3DPW Test

| Coverage | MPJPE | PA-MPJPE | Acceleration |
| ---: | ---: | ---: | ---: |
| 100.00% | 62.65 mm | 47.66 mm | 5.47 m/s2 |

The source migration passes exact parity over 39 fields across tracking,
camera construction, visual features, model input, and decoded model output.

## Preview

{preview}

This artifact is restricted to educational, research, and non-profit use.
Licensed SMPL and SMPL-X body models are not included.
"""
        )

    def forward(self, video: str | Path, output_root: str | Path, **kwargs) -> Path:
        return self.run_official_demo(video, output_root, **kwargs)


__all__ = [
    "GVHMR_ARTIFACT_FORMAT",
    "GVHMR_REPO_ID",
    "GVHMRBundle",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_RUNTIME_REVISION",
    "sha256_file",
]
