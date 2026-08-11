"""Repository-native, source-pinned GEM-X inference bundle."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Optional

from motius.models.base_model_bundle import ModelBundle
from motius.models.gem_runtime import (
    ensure_outputs_path,
    load_manifest,
    process_env,
    resolve_artifact,
    resolve_python,
    sha256_file,
    verify_manifest_assets,
)
from motius.registry import MODEL_BUNDLES


OFFICIAL_REPOSITORY = "https://github.com/NVlabs/GEM-X"
OFFICIAL_SOURCE_REVISION = "32992550dba114c62243fb55e361311972dce8f9"
OFFICIAL_SOMA_REVISION = "e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5"
OFFICIAL_SAM3D_REVISION = "b5c765a0d89d789985e186d396315e7590887b94"
OFFICIAL_DINOV3_REVISION = "6876159a11b4df116f30f667f8c9888617df0751"
OFFICIAL_HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
GEM_X_REPO_ID = "ZeyuLing/Motius-GEM-X"
GEM_X_ARTIFACT_FORMAT = "motius-gem-x-v1"
GEM_X_CHECKPOINT_SHA256 = (
    "4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e"
)
_ARTIFACT_CONFIG = "gem_x_config.json"
_CHECKPOINT = Path("inputs/pretrained/gem_soma.ckpt")
_YOLOX_NAME = "yolox_x_8xb8-300e_humanart-a39d44ed.onnx"
_REQUIRED_PUBLIC_ASSETS = (
    _CHECKPOINT,
    Path("inputs/checkpoints/vitpose/vitpose.pth"),
    Path("inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt"),
    Path("inputs/checkpoints/sam-3d-body-dinov3/model_config.yaml"),
    Path("inputs/checkpoints/yolox") / _YOLOX_NAME,
    Path("inputs/mhr_data/mhr_model.pt"),
    Path("inputs/soma_data/scale_mean.pth"),
    Path("inputs/soma_data/scale_comps.pth"),
    Path("inputs/soma_assets/SOMA_neutral.npz"),
    Path("inputs/soma_assets/correctives_model.pt"),
    Path("inputs/soma_assets/MHR/mhr_model_lod6.pt"),
    Path("inputs/soma_assets/MHR/base_body_lod6.obj"),
    Path("inputs/soma_assets/MHR/SOMA_wrap_lod1.obj"),
)
_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
_RUNNER_MODULE = "motius.models.gem_x.vendor.runner"


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Runtime path already exists: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


@MODEL_BUNDLES.register_module()
class GemXBundle(ModelBundle):
    """Own complete GEM-X/SOMA-77 assets without an external checkout."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        python_executable: Optional[str | Path] = None,
        export_device: str = "cuda",
        manifest: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.python_executable = str(
            python_executable
            or os.environ.get("MOTIUS_GEM_X_PYTHON")
            or sys.executable
        )
        self.export_device = str(export_device)
        self.manifest = dict(
            manifest
            if manifest is not None
            else load_manifest(
                self.artifact_root,
                filename=_ARTIFACT_CONFIG,
                artifact_format=GEM_X_ARTIFACT_FORMAT,
                source_revision=OFFICIAL_SOURCE_REVISION,
            )
        )
        self.checkpoint_path = self.artifact_root / _CHECKPOINT
        self.soma_assets = self.artifact_root / "inputs/soma_assets"

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
    ) -> "GemXBundle":
        root = resolve_artifact(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        return cls(artifact_root=root, **kwargs)

    @property
    def source_revision(self) -> str:
        return OFFICIAL_SOURCE_REVISION

    @property
    def checkpoint_sha256(self) -> str:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"GEM-X checkpoint not found: {self.checkpoint_path}")
        actual = sha256_file(self.checkpoint_path)
        if actual != GEM_X_CHECKPOINT_SHA256:
            raise RuntimeError(
                "GEM-X checkpoint SHA256 mismatch: "
                f"expected {GEM_X_CHECKPOINT_SHA256}, found {actual}."
            )
        return actual

    def validate_runtime(self, *, verify_all_assets: bool = False) -> None:
        if not (_VENDOR_ROOT / "scripts/demo/demo_soma.py").is_file():
            raise FileNotFoundError("Motius GEM-X vendored source is incomplete.")
        resolve_python(self.python_executable)
        missing = [
            path.as_posix()
            for path in _REQUIRED_PUBLIC_ASSETS
            if not (self.artifact_root / path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "GEM-X artifact is incomplete; missing: " + ", ".join(missing)
            )
        _ = self.checkpoint_sha256
        if verify_all_assets:
            verify_manifest_assets(self.artifact_root, self.manifest)

    def _env(
        self,
        runtime_root: Optional[Path] = None,
        *,
        render: bool = False,
    ) -> dict[str, str]:
        env = process_env(
            vendor_root=_VENDOR_ROOT,
            artifact_root=self.artifact_root,
            method_prefix="GEM_X",
        )
        third_party = (
            _VENDOR_ROOT / "third_party/sam-3d-body",
            _VENDOR_ROOT / "third_party/soma",
            _VENDOR_ROOT / "third_party/dinov3-repo",
        )
        env["PYTHONPATH"] = os.pathsep.join(
            (str(_VENDOR_ROOT), *(str(path) for path in third_party), env["PYTHONPATH"])
        )
        env["MOTIUS_GEM_X_SKIP_RENDER"] = "0" if render else "1"
        if runtime_root is not None:
            env["TORCH_HOME"] = str(runtime_root / "inputs/torch_cache")
        return env

    def _prepare_runtime(self, runtime_root: Path) -> None:
        runtime_root.mkdir(parents=True, exist_ok=False)
        for relative in _REQUIRED_PUBLIC_ASSETS:
            _symlink(self.artifact_root / relative, runtime_root / relative)
        detector = self.artifact_root / "inputs/checkpoints/yolox" / _YOLOX_NAME
        _symlink(
            detector,
            runtime_root / "inputs/torch_cache/hub/checkpoints" / _YOLOX_NAME,
        )

    def run_official_demo(
        self,
        video: str | Path,
        output_root: str | Path,
        *,
        static_camera: bool = False,
        render: bool = False,
        seed: Optional[int] = None,
        deterministic: bool = False,
        precomputed_stage_dir: Optional[str | Path] = None,
        extra_args: Sequence[str] = (),
        timeout: Optional[float] = None,
    ) -> Path:
        """Run vendored official inference and return ``hpe_results.pt``."""

        self.validate_runtime()
        video_path = Path(video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        output = ensure_outputs_path(output_root, method="GEM-X")
        run_root = output / f"motius_{video_path.stem}_{uuid.uuid4().hex}"
        runtime_root = run_root / "runtime"
        result_root = run_root / "results"
        self._prepare_runtime(runtime_root)
        if precomputed_stage_dir is not None:
            source = Path(precomputed_stage_dir).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Precomputed GEM-X stages not found: {source}")
            shutil.copytree(
                source,
                result_root / video_path.stem / "preprocess",
                dirs_exist_ok=True,
            )

        command = [
            resolve_python(self.python_executable),
            "-m",
            _RUNNER_MODULE,
            "--video",
            str(video_path),
            "--ckpt",
            str(runtime_root / _CHECKPOINT),
            "--sam3d_ckpt_path",
            str(
                runtime_root
                / "inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt"
            ),
            "--sam3d_mhr_path",
            str(runtime_root / "inputs/mhr_data/mhr_model.pt"),
            "--output_root",
            str(result_root),
        ]
        if static_camera:
            command.append("--static_cam")
        command.extend(str(value) for value in extra_args)
        env = self._env(runtime_root, render=render)
        env["MOTIUS_GEM_X_MODEL_INPUT_TRACE"] = str(
            result_root / video_path.stem / "preprocess/model_input.pt"
        )
        if seed is not None:
            env["MOTIUS_GEM_X_SEED"] = str(int(seed))
        if deterministic:
            if seed is None:
                raise ValueError("deterministic GEM-X inference requires a seed.")
            env["MOTIUS_GEM_X_DETERMINISTIC"] = "1"
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        subprocess.run(
            command,
            cwd=runtime_root,
            env=env,
            check=True,
            timeout=timeout,
        )
        result = result_root / video_path.stem / "hpe_results.pt"
        legacy = result_root / video_path.stem / "preprocess/hpe_results.pt"
        if not result.is_file() and legacy.is_file():
            result = legacy
        if not result.is_file():
            raise RuntimeError(f"GEM-X did not produce its documented result: {result}")
        return result

    def convert_official_result(
        self,
        result_path: str | Path,
        output_path: Optional[str | Path] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Path:
        """Materialize SOMA-77 joints and vertices with vendored SOMA-X."""

        self.validate_runtime()
        source = Path(result_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GEM-X result not found: {source}")
        destination = (
            source.with_name("motius_soma77_native.npz")
            if output_path is None
            else Path(output_path).expanduser().resolve()
        )
        ensure_outputs_path(destination.parent, method="GEM-X")
        runtime_root = source.parents[2] / "runtime"
        subprocess.run(
            [
                resolve_python(self.python_executable),
                "-m",
                "motius.pipelines.gem_x.export_native",
                "--input",
                str(source),
                "--output",
                str(destination),
                "--device",
                self.export_device,
                "--soma-assets",
                str(runtime_root / "inputs/soma_assets"),
            ],
            cwd=runtime_root,
            env=self._env(runtime_root),
            check=True,
            timeout=timeout,
        )
        if not destination.is_file():
            raise RuntimeError(f"GEM-X exporter did not produce {destination}.")
        return destination

    def verify(self) -> dict[str, str]:
        self.validate_runtime(verify_all_assets=True)
        return {
            "source_revision": self.source_revision,
            "soma_source_revision": OFFICIAL_SOMA_REVISION,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def save_pretrained(
        self,
        save_directory: str,
        *,
        source_assets: Optional[Mapping[str, str | Path]] = None,
    ) -> None:
        """Export exact official bytes as one complete Motius HF artifact."""

        output = ensure_outputs_path(save_directory, method="GEM-X")
        output.mkdir(parents=True, exist_ok=True)
        digests: dict[str, str] = {}
        for relative in _REQUIRED_PUBLIC_ASSETS:
            key = relative.as_posix()
            source = (
                Path(source_assets[key]).expanduser().resolve()
                if source_assets and key in source_assets
                else self.artifact_root / relative
            )
            if not source.is_file():
                raise FileNotFoundError(f"Missing GEM-X export asset: {source}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            digests[key] = sha256_file(destination)
        if digests[_CHECKPOINT.as_posix()] != GEM_X_CHECKPOINT_SHA256:
            raise RuntimeError("Refusing to export a non-official GEM-X checkpoint.")

        config = {
            "artifact_format": GEM_X_ARTIFACT_FORMAT,
            "model_type": "gem_x",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": OFFICIAL_SOURCE_REVISION,
            "soma_revision": OFFICIAL_SOMA_REVISION,
            "sam3d_revision": OFFICIAL_SAM3D_REVISION,
            "dinov3_revision": OFFICIAL_DINOV3_REVISION,
            "upstream_hf_revision": OFFICIAL_HF_REVISION,
            "checkpoint": _CHECKPOINT.as_posix(),
            "assets": [path.as_posix() for path in _REQUIRED_PUBLIC_ASSETS],
            "sha256": digests,
            "external_assets": {},
        }
        (output / _ARTIFACT_CONFIG).write_text(json.dumps(config, indent=2) + "\n")
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "GemXPipeline",
                    "_library_name": "motius",
                    "artifact_format": GEM_X_ARTIFACT_FORMAT,
                    "bundle_class": "motius.models.gem_x.GemXBundle",
                    "pipeline_class": "motius.pipelines.gem_x.GemXPipeline",
                    "tasks": ["monocular_motion_capture"],
                },
                indent=2,
            )
            + "\n"
        )

    def forward(self, video: str | Path, output_root: str | Path, **kwargs) -> Path:
        return self.run_official_demo(video, output_root, **kwargs)


__all__ = [
    "GEM_X_ARTIFACT_FORMAT",
    "GEM_X_CHECKPOINT_SHA256",
    "GEM_X_REPO_ID",
    "GemXBundle",
    "OFFICIAL_DINOV3_REVISION",
    "OFFICIAL_HF_REVISION",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_SAM3D_REVISION",
    "OFFICIAL_SOMA_REVISION",
    "OFFICIAL_SOURCE_REVISION",
]
