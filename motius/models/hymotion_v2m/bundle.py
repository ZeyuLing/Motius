"""HyMotion-V2M Bundle: video(feature)-to-motion generation via flow matching.

This bundle is a thin wrapper around the vendored ``MotionGenerationV2M``
pipeline (see ``vendor/hymotion/pipeline/motion_diffusion_v2m.py``).  The
vendored module is a self-contained, parity-preserving integration of the
original HunyuanMotion V2M inference stack. The original ``epoch*.ckpt``
(``model_state_dict``) loads with ``strict=True`` and produces numerically
identical inference outputs.

The bundle exposes the framework-friendly surface:

  - ``generate_from_feature(...)`` -- atomic forward: pre-extracted SAM-3D
    feature + camera -> flow-matching ODE -> 349-dim motion -> SMPL decode.
  - ``train_frames`` / ``body_model`` properties used by the Pipeline for
    sliding-window inference and SMPL forward kinematics.

Stage 1 (this file) only needs pre-extracted features.  Video preprocessing
(YOLOX + SAM-3D-Body) is added in stage 2 on top of the same bundle.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Mapping, Optional

import torch
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

# Dotted import prefix of the vendored, self-contained V2M source package.
_VENDOR_PREFIX = "motius.models.hymotion_v2m.vendor.hymotion"
# Motius repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
HYMOTION_V2M_SOURCE_REVISION = "motius_hymotion_v2m_release_v1"
HYMOTION_V2M_REPO_ID = "ZeyuLing/Motius-HYMotion-V2M"
HYMOTION_V2M_ARTIFACT_FORMAT = "motius-hymotion-v2m-v1"
_EXCLUDED_BODY_MODEL_PREFIXES = (
    "vertex_loss.body_model.",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _rewrite_module_path(path: str) -> str:
    """Rewrite an original ``hymotion/...`` module path to the vendored package.

    The V2M ``config.yml`` stores module references like
    ``hymotion/network/hymotion_mmdit_for_v2m.HunyuanMotionMMDiT`` which the
    vendored ``load_object`` resolves via ``importlib``.  Since the vendored
    code lives under ``motius...vendor.hymotion`` (there is no top-level
    ``hymotion`` package inside Motius), rewrite the leading namespace.
    """
    norm = path.replace("/", ".")
    if norm.startswith("hymotion."):
        norm = _VENDOR_PREFIX + norm[len("hymotion"):]
    return norm


def _resolve_path(path: Optional[str]) -> Optional[str]:
    """Resolve a possibly repo-relative path to an absolute path."""
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() or p.exists():
        return str(p)
    cand = _REPO_ROOT / path
    if cand.exists():
        return str(cand)
    return str(path)


def _embedded_body_model_state(checkpoint_path: Optional[str]) -> Optional[dict]:
    """Read only the SMPL-H FK/mesh buffers embedded in a safetensors artifact."""
    if checkpoint_path is None:
        return None
    logical_path = Path(checkpoint_path).expanduser()
    if logical_path.suffix != ".safetensors" or not logical_path.exists():
        return None
    from safetensors import safe_open

    groups = {"body_model": {}, "mesh_model": {}}
    with safe_open(str(logical_path), framework="pt", device="cpu") as stream:
        keys = set(stream.keys())
        for group in groups:
            prefix = f"{group}."
            for key in sorted(key for key in keys if key.startswith(prefix)):
                groups[group][key[len(prefix) :]] = stream.get_tensor(key)
    if not groups["body_model"] or not groups["mesh_model"]:
        return None
    return groups


@MODEL_BUNDLES.register_module()
class HyMotionV2MBundle(ModelBundle):
    """ModelBundle wrapping the vendored ``MotionGenerationV2M`` pipeline.

    Args:
        v2m_config_path: path to the original V2M ``config.yml``.  When given,
            ``network_module``, ``network_module_args`` and the pipeline args
            are read from it (under ``train_pipeline_args``).
        ckpt_path: path to the original ``epoch*.ckpt`` whose
            ``model_state_dict`` is loaded into the vendored pipeline.
        mean_std_path: override for the mean/std JSON asset.  Defaults to the
            value declared in the config / pipeline args.
        network_module: explicit network module path (overrides config).
        network_module_args: explicit network kwargs (overrides config); used
            by the smoke config to build a tiny transformer.
        pipeline_args: explicit pipeline kwargs when no config file is given.
        pipeline_overrides: nested overrides merged onto the pipeline args
            (e.g. shrink ``infer_noise_scheduler_cfg.validation_steps``).
        strict_load: whether checkpoint loading is strict (default True).
        device: optional device to move the bundle onto after construction.
    """

    def __init__(
        self,
        v2m_config_path: Optional[str] = None,
        ckpt_path: Optional[str] = None,
        mean_std_path: Optional[str] = None,
        body_model_path: Optional[str] = None,
        network_module: Optional[str] = None,
        network_module_args: Optional[dict] = None,
        pipeline_args: Optional[dict] = None,
        pipeline_overrides: Optional[dict] = None,
        strict_load: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        self._checkpoint_path: Optional[Path] = None
        self._checkpoint_sha256: Optional[str] = None

        from .vendor.hymotion.pipeline.motion_diffusion_v2m import (
            MotionGenerationV2M,
        )
        from .vendor.hymotion.utils.loaders import read_yaml

        if v2m_config_path is not None:
            cfg = read_yaml(_resolve_path(v2m_config_path))
            if network_module is None:
                network_module = cfg["network_module"]
            if network_module_args is None:
                network_module_args = deepcopy(cfg["network_module_args"])
            base_pipeline_args = deepcopy(cfg["train_pipeline_args"])
        else:
            base_pipeline_args = deepcopy(pipeline_args or {})

        if network_module is None:
            network_module = (
                "hymotion/network/hymotion_mmdit_for_v2m.HunyuanMotionMMDiT"
            )
        artifact_network_module = network_module
        network_module = _rewrite_module_path(network_module)
        network_module_args = deepcopy(network_module_args or {})

        if pipeline_overrides:
            base_pipeline_args = self._merge_nested_dict(
                base_pipeline_args, pipeline_overrides
            )

        # Resolve the mean/std asset (load_mean_std opens it as a file path).
        mean_std = mean_std_path or base_pipeline_args.get("mean_std")
        base_pipeline_args["mean_std"] = _resolve_path(mean_std)
        embedded_body_state = (
            None
            if body_model_path is not None
            else _embedded_body_model_state(_resolve_path(ckpt_path))
        )
        if body_model_path is not None:
            base_pipeline_args["body_model_path"] = _resolve_path(body_model_path)
        elif embedded_body_state is not None:
            base_pipeline_args["body_model_path"] = None
        else:
            base_pipeline_args["body_model_path"] = _resolve_path(
                "checkpoints/body_models/smplh/neutral/model.npz"
            )

        self._network_module = network_module
        self._artifact_network_module = artifact_network_module
        self._network_module_args = deepcopy(network_module_args)
        self._pipeline_args = deepcopy(base_pipeline_args)
        self._mean_std_path = (
            None
            if base_pipeline_args["mean_std"] is None
            else Path(base_pipeline_args["mean_std"]).expanduser().resolve()
        )
        self._motion_rep = base_pipeline_args.get("motion_rep")
        self._pred_type = base_pipeline_args.get("pred_type")

        # Build the vendored pipeline (an nn.Module).  Its ``__init__`` builds
        # the transformer via load_object, SMPL body/mesh models, losses and
        # registers mean/std buffers from the JSON.
        build_pipeline_args = deepcopy(base_pipeline_args)
        build_pipeline_args["body_model_state"] = embedded_body_state
        self.model = MotionGenerationV2M(
            network_module=network_module,
            network_module_args=network_module_args,
            **build_pipeline_args,
        )

        # Register ``model`` as the single framework-managed sub-module so the
        # AccelerateRunner save/load path treats it like a normal module.
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self._frozen_modules = []
        self._module_checkpoint_formats = {"model": "full"}

        if ckpt_path is not None:
            self.load_v2m_checkpoint(_resolve_path(ckpt_path), strict=strict_load)

        if device is not None:
            self.to(torch.device(device))

    # ------------------------------------------------------------------
    # HF-style construction
    # ------------------------------------------------------------------
    #: checkpoint filenames searched (in order) when ``ckpt_name`` is omitted.
    _CKPT_CANDIDATES = (
        "model.safetensors",
        "epoch100.ckpt",
        "latest.ckpt",
        "model.ckpt",
        "pytorch_model.bin",
    )
    #: mean/std asset filenames searched inside a pretrained directory.
    _MEAN_STD_CANDIDATES = (
        "mean_std.json",
        "v2m_wv_mean_std_1200h_step10.json",
    )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        config_name: str = "config.yml",
        ckpt_name: Optional[str] = None,
        mean_std_path: Optional[str] = None,
        strict_load: bool = True,
        device: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs,
    ) -> "HyMotionV2MBundle":
        """Build the bundle from a released V2M artifact directory (or ckpt file).

        The V2M artifact is *not* a diffusers/transformers layout, so this
        overrides the declarative ``ModelBundle.from_pretrained`` with a small
        path resolver.  Accepted inputs:

        - a **directory** containing ``config.yml`` + a checkpoint
          (``epoch*.ckpt`` / ``latest.ckpt`` / ``model.ckpt``) and, optionally,
          a ``*mean_std*.json`` asset;
        - a **checkpoint file** whose sibling ``config.yml`` is used.

        Args:
            pretrained_model_name_or_path: artifact dir or ``*.ckpt`` path
                (repo-relative paths are resolved against the repo root).
            config_name: config filename inside the directory (``config.yml``).
            ckpt_name: explicit checkpoint filename (otherwise auto-detected).
            mean_std_path: override for the mean/std asset; when omitted, a
                ``*mean_std*.json`` inside the directory is used if present,
                else the path declared in ``config.yml`` is resolved.
            strict_load / device: forwarded to ``__init__``.
        """
        resolved = _resolve_path(pretrained_model_name_or_path)
        if resolved is None:
            raise ValueError("pretrained_model_name_or_path must not be None")
        root = Path(resolved)
        if not root.exists():
            from huggingface_hub import snapshot_download

            root = Path(
                snapshot_download(
                    repo_id=pretrained_model_name_or_path,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        root = root.resolve()

        if root.is_dir():
            ckpt_path = cls._find_checkpoint(root, ckpt_name)
            if mean_std_path is None:
                mean_std_path = cls._find_mean_std(root)
            artifact_config_path = root / "hymotion_v2m_config.json"
            if artifact_config_path.exists():
                artifact = json.loads(artifact_config_path.read_text())
                if artifact.get("artifact_format") != HYMOTION_V2M_ARTIFACT_FORMAT:
                    raise ValueError(
                        "Unsupported HYMotion-V2M artifact format "
                        f"{artifact.get('artifact_format')!r}."
                    )
                bundle = cls(
                    ckpt_path=str(ckpt_path),
                    mean_std_path=mean_std_path,
                    network_module=artifact["network_module"],
                    network_module_args=artifact["network_module_args"],
                    pipeline_args=artifact["pipeline_args"],
                    strict_load=strict_load,
                    device=device,
                    **kwargs,
                )
                source_digest = artifact.get("source_checkpoint_sha256")
                if source_digest:
                    bundle._checkpoint_sha256 = str(source_digest).lower()
                return bundle
            config_path = root / config_name
            if not config_path.exists():
                raise FileNotFoundError(
                    f"V2M artifact dir missing config: {config_path}"
                )
        elif root.is_file():
            ckpt_path = root
            config_path = root.parent / config_name
            if not config_path.exists():
                raise FileNotFoundError(
                    f"V2M config '{config_name}' not found beside checkpoint: "
                    f"{config_path}"
                )
            if mean_std_path is None:
                mean_std_path = cls._find_mean_std(root.parent)
        else:
            raise FileNotFoundError(
                f"V2M pretrained path does not exist: {root}"
            )

        return cls(
            v2m_config_path=str(config_path),
            ckpt_path=str(ckpt_path),
            mean_std_path=mean_std_path,
            strict_load=strict_load,
            device=device,
            **kwargs,
        )

    @classmethod
    def _find_checkpoint(cls, root: Path, ckpt_name: Optional[str]) -> Path:
        if ckpt_name is not None:
            cand = root / ckpt_name
            if not cand.exists():
                raise FileNotFoundError(f"Checkpoint not found: {cand}")
            return cand
        for name in cls._CKPT_CANDIDATES:
            cand = root / name
            if cand.exists():
                return cand
        # last resort: any single *.ckpt in the directory
        ckpts = sorted(root.glob("*.ckpt"))
        if len(ckpts) == 1:
            return ckpts[0]
        raise FileNotFoundError(
            f"No checkpoint found in {root}. Looked for {cls._CKPT_CANDIDATES} "
            f"and *.ckpt (found {len(ckpts)})."
        )

    @classmethod
    def _find_mean_std(cls, root: Path) -> Optional[str]:
        for name in cls._MEAN_STD_CANDIDATES:
            cand = root / name
            if cand.exists():
                return str(cand)
        # fall back to any *mean_std*.json in the directory
        hits = sorted(root.glob("*mean_std*.json"))
        if hits:
            return str(hits[0])
        return None  # let __init__ resolve the path declared in config.yml

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def load_v2m_checkpoint(self, ckpt_path: str, strict: bool = True):
        """Load the original V2M ``epoch*.ckpt`` ``model_state_dict``."""
        logical_path = Path(ckpt_path).expanduser()
        checkpoint_path = logical_path.resolve()
        # Hugging Face snapshots expose logical filenames as symlinks to
        # extensionless cache blobs. Preserve the logical suffix for format
        # dispatch while retaining the resolved path for immutable provenance.
        checkpoint_suffix = logical_path.suffix or checkpoint_path.suffix
        if checkpoint_suffix == ".safetensors":
            from safetensors.torch import load_file

            state = load_file(str(checkpoint_path), device="cpu")
            incompatible = self.model.load_state_dict(state, strict=False)
            if strict:
                invalid_missing = [
                    key
                    for key in incompatible.missing_keys
                    if not key.startswith(_EXCLUDED_BODY_MODEL_PREFIXES)
                ]
                if invalid_missing or incompatible.unexpected_keys:
                    raise RuntimeError(
                        "HYMotion-V2M artifact state mismatch: "
                        f"missing={invalid_missing}, "
                        f"unexpected={incompatible.unexpected_keys}"
                    )
            result = incompatible
        else:
            ckpt = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                state = ckpt["model_state_dict"]
            else:
                state = ckpt
            result = self.model.load_state_dict(state, strict=strict)
        self._checkpoint_path = checkpoint_path
        self._checkpoint_sha256 = None
        return result

    # ------------------------------------------------------------------
    # Properties used by the Pipeline
    # ------------------------------------------------------------------
    @property
    def train_frames(self) -> int:
        return int(self.model.train_frames)

    @property
    def body_model(self):
        return self.model.body_model

    @property
    def feature_dim(self) -> int:
        """Context feature dim (SAM-3D token dim) expected as ``feature['feature']``."""
        ctxt = self.model.motion_transformer.ctxt_input_dim
        if isinstance(ctxt, dict):
            return int(ctxt.get("feature", next(iter(ctxt.values()))))
        return int(ctxt)

    @property
    def motion_rep(self) -> Optional[str]:
        return self._motion_rep

    @property
    def source_revision(self) -> str:
        return HYMOTION_V2M_SOURCE_REVISION

    @property
    def checkpoint_sha256(self) -> str:
        if self._checkpoint_sha256 is not None:
            return self._checkpoint_sha256
        if self._checkpoint_path is None:
            raise RuntimeError(
                "No checkpoint was loaded; pass checkpoint_sha256 explicitly."
            )
        self._checkpoint_sha256 = _sha256_file(self._checkpoint_path)
        return self._checkpoint_sha256

    def save_pretrained(
        self,
        save_directory: str,
        *,
        preprocessor_assets: Optional[Mapping[str, str]] = None,
        safe_serialization: bool = True,
    ) -> None:
        """Export the model, stats, and optional video front end as one artifact."""

        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        if self._mean_std_path is None or not self._mean_std_path.is_file():
            raise FileNotFoundError("HYMotion-V2M mean/std JSON is unavailable.")
        shutil.copy2(self._mean_std_path, output / "mean_std.json")

        weights_name = "model.safetensors" if safe_serialization else "pytorch_model.bin"
        state = {
            key: value.detach().cpu().contiguous()
            for key, value in self.model.state_dict().items()
            if not key.startswith(_EXCLUDED_BODY_MODEL_PREFIXES)
        }
        if safe_serialization:
            from safetensors.torch import save_file

            save_file(state, str(output / weights_name))
        else:
            torch.save(state, output / weights_name)

        pipeline_args = deepcopy(self._pipeline_args)
        pipeline_args["mean_std"] = "mean_std.json"
        pipeline_args.pop("body_model_path", None)
        if isinstance(pipeline_args.get("test_cfg"), dict):
            pipeline_args["test_cfg"]["mean_std_dir"] = "mean_std.json"

        stored_preprocessor_assets = {}
        if preprocessor_assets:
            asset_dir = output / "preprocessor"
            asset_dir.mkdir(parents=True, exist_ok=True)
            filenames = {
                "sam3d_ckpt": "sam3d_body.ckpt",
                "sam3d_mhr": "mhr_model.pt",
                "sam3d_config": "model_config.yaml",
                "yolox_ckpt": "yolox_l.pth",
            }
            for key, filename in filenames.items():
                source = preprocessor_assets.get(key)
                if source is None:
                    continue
                source_path = Path(source).expanduser().resolve()
                if not source_path.is_file():
                    raise FileNotFoundError(f"Missing {key} asset: {source_path}")
                target = asset_dir / filename
                shutil.copy2(source_path, target)
                stored_preprocessor_assets[key] = target.relative_to(output).as_posix()

        artifact = {
            "artifact_format": HYMOTION_V2M_ARTIFACT_FORMAT,
            "model_type": "hymotion_v2m",
            "source_repository": "https://github.com/Tencent-Hunyuan/HY-Motion-1.0",
            "source_revision": HYMOTION_V2M_SOURCE_REVISION,
            "source_checkpoint_sha256": self.checkpoint_sha256,
            "network_module": self._artifact_network_module,
            "network_module_args": self._network_module_args,
            "pipeline_args": pipeline_args,
            "weights": weights_name,
            "mean_std": "mean_std.json",
            "preprocessor_assets": stored_preprocessor_assets,
            "components": {
                "smplh_kinematics": {
                    "stored_in_artifact": True,
                    "path": weights_name,
                    "state_prefixes": ["body_model.", "mesh_model."],
                }
            },
        }
        (output / "hymotion_v2m_config.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
        (output / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "HyMotionV2MPipeline",
                    "_library_name": "motius",
                    "artifact_format": HYMOTION_V2M_ARTIFACT_FORMAT,
                    "bundle_class": (
                        "motius.models.hymotion_v2m.bundle.HyMotionV2MBundle"
                    ),
                    "pipeline_class": (
                        "motius.pipelines.hymotion_v2m."
                        "hymotion_v2m_pipeline.HyMotionV2MPipeline"
                    ),
                    "tasks": ["monocular_motion_capture"],
                    "required_files": [
                        "hymotion_v2m_config.json",
                        weights_name,
                        "mean_std.json",
                        *stored_preprocessor_assets.values(),
                    ],
                    "artifacts": {
                        "model": weights_name,
                        "mean_std": "mean_std.json",
                        **stored_preprocessor_assets,
                    },
                },
                indent=2,
            )
            + "\n"
        )
        package_root = Path(__file__).resolve().parent
        legal_files = {
            "License.txt": package_root / "License.txt",
            "NOTICE": package_root / "NOTICE",
            "ATTRIBUTIONS.md": package_root / "ATTRIBUTIONS.md",
            "SAM3D_LICENSE": package_root / "vendor" / "SAM3D_LICENSE",
            "DINOV3_LICENSE.md": package_root / "vendor" / "DINOV3_LICENSE.md",
            "YOLOX_LICENSE": package_root / "vendor" / "YOLOX_LICENSE",
        }
        for filename, source in legal_files.items():
            if not source.is_file():
                raise FileNotFoundError(f"Missing required legal file: {source}")
            shutil.copy2(source, output / filename)

        (output / "README.md").write_text(
            f"""---
library_name: motius
pipeline_tag: other
license: other
tags:
  - monocular-motion-capture
  - smpl-h
  - video-to-motion
---

# HYMotion-V2M for Motius

This artifact packages the HYMotion-V2M generator, motion statistics,
SAM-3D-Body/MHR image encoder assets, and YOLOX detector for the first-class
Motius pipeline. It does not require an external source checkout.

- Paper: https://arxiv.org/abs/2512.23464
- Official source: https://github.com/Tencent-Hunyuan/HY-Motion-1.0
- Motius source: https://github.com/ZeyuLing/Motius
- Source checkpoint SHA-256: `{self.checkpoint_sha256}`

## Usage

```python
from motius.pipelines.hymotion_v2m import HyMotionV2MPipeline

pipeline = HyMotionV2MPipeline.from_pretrained(
    "{HYMOTION_V2M_REPO_ID}",
    bundle_kwargs={{
            "device": "cuda",
    }},
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4",
    work_dir="outputs/hymotion_v2m/run_001",
)
```

The checkpoint embeds the FK and mesh buffers needed by inference. It does not
redistribute the original licensed SMPL-H parameter file. See
`ATTRIBUTIONS.md`, `License.txt`, and the bundled third-party license files
before use or redistribution.

## Numerical parity

Motius compares named intermediate stages with exact element-wise equality
(`rtol=0`, `atol=0`). The release check covers tracking, camera conditioning,
SAM-3D tokens, model conditioning, every inference window, stitching,
SMPL-H forward kinematics, ground alignment, and the public result contract.
"""
        )

    # ------------------------------------------------------------------
    # Atomic forward (shared by Pipeline)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate_from_feature(
        self,
        feature: Dict[str, Tensor],
        seeds: List[int],
        length: int,
        camera_is_static: bool = True,
        cfg_scale: float = 1.0,
        do_postproc: bool = False,
        debug: bool = False,
    ) -> Dict[str, Tensor]:
        """Run flow-matching ODE for one window of pre-extracted features.

        Args:
            feature: dict with ``feature`` (B, T, Dctx), ``camera_R`` (B, T, 9)
                and ``camera_T`` (B, T, 3).  ``T`` should equal ``train_frames``.
            seeds: list of integer seeds; one sample per seed.
            length: number of valid frames in the padded window
                (``1 <= length <= train_frames``).
            camera_is_static: whether the camera is static for this clip.
            cfg_scale: classifier-free guidance scale (1.0 = off).
            do_postproc: forwarded to the vendored ``generate``.

        Returns:
            Decoded motion dict: ``rot6d``, ``shapes``, ``trans``,
            ``global_orient``, ``local_transl_vel``, ``end_effector_vel``.
        """
        requested_length = int(length)
        train_frames = self.train_frames
        if not 1 <= requested_length <= train_frames:
            raise ValueError(
                f"length must be in [1, {train_frames}], got {requested_length}"
            )
        for key in ("feature", "camera_R", "camera_T"):
            value = feature.get(key)
            if not isinstance(value, Tensor):
                raise TypeError(f"feature[{key!r}] must be a torch.Tensor")
            if value.dim() < 2 or int(value.shape[1]) != train_frames:
                raise ValueError(
                    f"feature[{key!r}] must have temporal length {train_frames}, "
                    f"got {tuple(value.shape)}"
                )

        # The released narrowband_v2m network cannot safely evaluate a
        # partially masked final window: padded query rows can have no finite
        # attention keys and contaminate valid rows in later single-stream
        # blocks. The official checkpoint was trained on a fixed 360-frame
        # canvas, so evaluate the repeated-padded canvas as fully valid and
        # enforce the caller's requested length at the API boundary.
        output = self.model.generate(
            feature=feature,
            seeds=list(seeds),
            length=train_frames,
            camera_is_static=camera_is_static,
            cfg_scale=cfg_scale,
            do_postproc=do_postproc,
            debug=debug,
        )
        return {
            key: (
                value[:, :requested_length].clone()
                if isinstance(value, Tensor)
                and value.dim() >= 2
                and int(value.shape[1]) == train_frames
                else value
            )
            for key, value in output.items()
        }

    def forward(self, *args, **kwargs):
        return self.generate_from_feature(*args, **kwargs)
