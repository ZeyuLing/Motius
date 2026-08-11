"""Repository-native, source-pinned GEM-SMPL inference bundle."""

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


OFFICIAL_REPOSITORY = "https://github.com/NVlabs/GENMO"
OFFICIAL_SOURCE_REVISION = "16bebf402d8893184249ee206d957b8248cd8310"
OFFICIAL_HF_REVISION = "5ccf5ca3746c3620aa4016114f069a5f6ae399cd"
GEM_SMPL_REPO_ID = "ZeyuLing/Motius-GEM-SMPL"
GEM_SMPL_ARTIFACT_FORMAT = "motius-gem-smpl-v1"
GEM_SMPL_CHECKPOINT_SHA256 = (
    "1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a"
)
_ARTIFACT_CONFIG = "gem_smpl_config.json"
_CHECKPOINT = Path("inputs/pretrained/gem_smpl.ckpt")
_REQUIRED_PUBLIC_ASSETS = (
    _CHECKPOINT,
    Path("inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt"),
    Path("inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth"),
    Path("yolov8x.pt"),
)
_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
_RUNNER_MODULE = "motius.models.gem_smpl.vendor.runner"
_VENDORED_BODY_ASSETS = {
    "coco_aug_dict.pth": "9d045cc3e507e1f7d91ef89904cdfa26e29cf18710f52f9fd305a88ae5d4e539",
    "smpl_3dpw14_J_regressor_sparse.pt": "f3f4cf476e206d36ac806beab9b728b886c10f1534a74318839391f76ad5ab0a",
    "smpl_coco17_J_regressor.pt": "bacdaf756629493994cc869f4c27d179f5e4a5d06b8797ee3dcb94571522079f",
    "smpl_neutral_J_regressor.pt": "70e3213bd30fe8d8ce37b54675282745e406f915a51511a003aeff99b6da04cf",
    "smplx2smpl_sparse.pt": "0fc821a9e79ec3e76d6a9796b96d5bef8cd67055e18497bbe370d8aed9e07e06",
    "smplx_verts437.pt": "ef0ea64c470a1fea80adb5e4b866c78a792a04f5f98744c9367b15e57bfb4a4d",
}


def _symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        if destination.resolve() == source.resolve():
            return
        raise FileExistsError(f"Runtime path already exists: {destination}")
    destination.symlink_to(source.resolve(), target_is_directory=source.is_dir())


@MODEL_BUNDLES.register_module()
class GemSmplBundle(ModelBundle):
    """Own all public GEM-SMPL assets and the isolated runtime boundary."""

    def __init__(
        self,
        *,
        artifact_root: str | Path,
        python_executable: Optional[str | Path] = None,
        body_models_root: Optional[str | Path] = None,
        export_device: str = "cuda",
        manifest: Optional[Mapping[str, object]] = None,
    ) -> None:
        super().__init__()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        self.python_executable = str(
            python_executable
            or os.environ.get("MOTIUS_GEM_SMPL_PYTHON")
            or sys.executable
        )
        self.body_models_root = (
            None
            if body_models_root is None
            else Path(body_models_root).expanduser().resolve()
        )
        self.export_device = str(export_device)
        self.manifest = dict(
            manifest
            if manifest is not None
            else load_manifest(
                self.artifact_root,
                filename=_ARTIFACT_CONFIG,
                artifact_format=GEM_SMPL_ARTIFACT_FORMAT,
                source_revision=OFFICIAL_SOURCE_REVISION,
            )
        )
        self.checkpoint_path = self.artifact_root / _CHECKPOINT

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
    ) -> "GemSmplBundle":
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
            raise FileNotFoundError(f"GEM-SMPL checkpoint not found: {self.checkpoint_path}")
        actual = sha256_file(self.checkpoint_path)
        if actual != GEM_SMPL_CHECKPOINT_SHA256:
            raise RuntimeError(
                "GEM-SMPL checkpoint SHA256 mismatch: "
                f"expected {GEM_SMPL_CHECKPOINT_SHA256}, found {actual}."
            )
        return actual

    def validate_runtime(
        self,
        *,
        require_body_models: bool = True,
        verify_all_assets: bool = False,
    ) -> None:
        if not (_VENDOR_ROOT / "scripts/demo/demo_smpl_hpe.py").is_file():
            raise FileNotFoundError("Motius GEM-SMPL vendored source is incomplete.")
        body_asset_root = _VENDOR_ROOT / "gem/utils/body_model"
        for filename, expected in _VENDORED_BODY_ASSETS.items():
            path = body_asset_root / filename
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(
                    f"Motius GEM-SMPL vendored body asset is missing or changed: {filename}"
                )
        resolve_python(self.python_executable)
        missing = [
            path.as_posix()
            for path in _REQUIRED_PUBLIC_ASSETS
            if not (self.artifact_root / path).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "GEM-SMPL artifact is incomplete; missing: " + ", ".join(missing)
            )
        _ = self.checkpoint_sha256
        if verify_all_assets:
            verify_manifest_assets(self.artifact_root, self.manifest)
        if require_body_models and (
            self.body_models_root is None or not self.body_models_root.is_dir()
        ):
            raise FileNotFoundError(
                "GEM-SMPL requires user-licensed SMPL and SMPL-X files. Pass "
                "body_models_root='checkpoints/body_models'."
            )

    def _env(self) -> dict[str, str]:
        return process_env(
            vendor_root=_VENDOR_ROOT,
            artifact_root=self.artifact_root,
            method_prefix="GEM_SMPL",
        )

    def _prepare_runtime(self, runtime_root: Path) -> None:
        runtime_root.mkdir(parents=True, exist_ok=False)
        for relative in _REQUIRED_PUBLIC_ASSETS:
            _symlink(self.artifact_root / relative, runtime_root / relative)
        if self.body_models_root is not None:
            _symlink(
                self.body_models_root,
                runtime_root / "inputs/checkpoints/body_models",
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
        """Run vendored official inference and return ``smpl_params.pt``."""

        self.validate_runtime(require_body_models=True)
        video_path = Path(video).expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Input video not found: {video_path}")
        output = ensure_outputs_path(output_root, method="GEM-SMPL")
        run_root = output / f"motius_{video_path.stem}_{uuid.uuid4().hex}"
        runtime_root = run_root / "runtime"
        result_root = run_root / "results"
        self._prepare_runtime(runtime_root)
        if precomputed_stage_dir is not None:
            source = Path(precomputed_stage_dir).expanduser().resolve()
            if not source.is_dir():
                raise FileNotFoundError(f"Precomputed GEM-SMPL stages not found: {source}")
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
            "--ckpt_path",
            str(runtime_root / _CHECKPOINT),
            "--output_root",
            str(result_root),
        ]
        if not render:
            command.append("--no_render")
        if static_camera:
            command.append("--static_cam")
        command.extend(str(value) for value in extra_args)
        env = self._env()
        env["MOTIUS_GEM_SMPL_MODEL_INPUT_TRACE"] = str(
            result_root / video_path.stem / "preprocess/model_input.pt"
        )
        if seed is not None:
            env["MOTIUS_GEM_SMPL_SEED"] = str(int(seed))
        if deterministic:
            if seed is None:
                raise ValueError("deterministic GEM-SMPL inference requires a seed.")
            env["MOTIUS_GEM_SMPL_DETERMINISTIC"] = "1"
            env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        subprocess.run(
            command,
            cwd=runtime_root,
            env=env,
            check=True,
            timeout=timeout,
        )
        result = result_root / video_path.stem / "smpl_params.pt"
        if not result.is_file():
            raise RuntimeError(f"GEM-SMPL did not produce its documented result: {result}")
        return result

    def convert_official_result(
        self,
        result_path: str | Path,
        output_path: Optional[str | Path] = None,
        *,
        timeout: Optional[float] = None,
    ) -> Path:
        """Materialize official SMPL joints and vertices without another repo."""

        self.validate_runtime(require_body_models=True)
        source = Path(result_path).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(f"GEM-SMPL result not found: {source}")
        destination = (
            source.with_name("motius_smpl_native.npz")
            if output_path is None
            else Path(output_path).expanduser().resolve()
        )
        ensure_outputs_path(destination.parent, method="GEM-SMPL")
        runtime_root = source.parents[2] / "runtime"
        subprocess.run(
            [
                resolve_python(self.python_executable),
                "-m",
                "motius.pipelines.gem_smpl.export_native",
                "--input",
                str(source),
                "--output",
                str(destination),
                "--device",
                self.export_device,
            ],
            cwd=runtime_root,
            env=self._env(),
            check=True,
            timeout=timeout,
        )
        if not destination.is_file():
            raise RuntimeError(f"GEM-SMPL exporter did not produce {destination}.")
        return destination

    def verify(self) -> dict[str, str]:
        self.validate_runtime(require_body_models=False, verify_all_assets=True)
        return {
            "source_revision": self.source_revision,
            "checkpoint_sha256": self.checkpoint_sha256,
        }

    def save_pretrained(
        self,
        save_directory: str,
        *,
        source_assets: Optional[Mapping[str, str | Path]] = None,
    ) -> None:
        """Export exact official bytes as one complete Motius HF artifact."""

        output = ensure_outputs_path(save_directory, method="GEM-SMPL")
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
                raise FileNotFoundError(f"Missing GEM-SMPL export asset: {source}")
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
            digests[key] = sha256_file(destination)
        if digests[_CHECKPOINT.as_posix()] != GEM_SMPL_CHECKPOINT_SHA256:
            raise RuntimeError("Refusing to export a non-official GEM-SMPL checkpoint.")

        config = {
            "artifact_format": GEM_SMPL_ARTIFACT_FORMAT,
            "model_type": "gem_smpl",
            "source_repository": OFFICIAL_REPOSITORY,
            "source_revision": OFFICIAL_SOURCE_REVISION,
            "upstream_hf_revision": OFFICIAL_HF_REVISION,
            "checkpoint": _CHECKPOINT.as_posix(),
            "assets": [path.as_posix() for path in _REQUIRED_PUBLIC_ASSETS],
            "sha256": digests,
            "external_assets": {
                "checkpoints/body_models": "User-licensed SMPL and SMPL-X files."
            },
        }
        (output / _ARTIFACT_CONFIG).write_text(json.dumps(config, indent=2) + "\n")
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "GemSmplPipeline",
                    "_library_name": "motius",
                    "artifact_format": GEM_SMPL_ARTIFACT_FORMAT,
                    "bundle_class": "motius.models.gem_smpl.GemSmplBundle",
                    "pipeline_class": "motius.pipelines.gem_smpl.GemSmplPipeline",
                    "tasks": ["monocular_motion_capture"],
                },
                indent=2,
            )
            + "\n"
        )

    def forward(self, video: str | Path, output_root: str | Path, **kwargs) -> Path:
        return self.run_official_demo(video, output_root, **kwargs)


__all__ = [
    "GEM_SMPL_ARTIFACT_FORMAT",
    "GEM_SMPL_CHECKPOINT_SHA256",
    "GEM_SMPL_REPO_ID",
    "GemSmplBundle",
    "OFFICIAL_HF_REVISION",
    "OFFICIAL_REPOSITORY",
    "OFFICIAL_SOURCE_REVISION",
]
