"""HyMotion-V2M Bundle: video(feature)-to-motion generation via flow matching.

This bundle is a thin wrapper around the vendored ``MotionGenerationV2M``
pipeline (see ``vendor/hymotion/pipeline/motion_diffusion_v2m.py``).  The
vendored module is a verbatim, self-contained copy of the original
HunyuanMotion V2M inference stack, so the original ``epoch*.ckpt``
(``model_state_dict``) loads with ``strict=True`` and produces numerically
identical results.

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
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

# Dotted import prefix of the vendored, self-contained V2M source package.
_VENDOR_PREFIX = "motius.models.hymotion_v2m.vendor.hymotion"
# Motius repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]


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
        network_module = _rewrite_module_path(network_module)
        network_module_args = deepcopy(network_module_args or {})

        if pipeline_overrides:
            base_pipeline_args = self._merge_nested_dict(
                base_pipeline_args, pipeline_overrides
            )

        # Resolve the mean/std asset (load_mean_std opens it as a file path).
        mean_std = mean_std_path or base_pipeline_args.get("mean_std")
        base_pipeline_args["mean_std"] = _resolve_path(mean_std)
        base_pipeline_args["body_model_path"] = _resolve_path(
            body_model_path
            or "checkpoints/body_models/smplh/neutral/model.npz"
        )

        self._network_module = network_module
        self._motion_rep = base_pipeline_args.get("motion_rep")
        self._pred_type = base_pipeline_args.get("pred_type")

        # Build the vendored pipeline (an nn.Module).  Its ``__init__`` builds
        # the transformer via load_object, SMPL body/mesh models, losses and
        # registers mean/std buffers from the JSON.
        self.model = MotionGenerationV2M(
            network_module=network_module,
            network_module_args=network_module_args,
            **base_pipeline_args,
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

        if root.is_dir():
            config_path = root / config_name
            if not config_path.exists():
                raise FileNotFoundError(
                    f"V2M artifact dir missing config: {config_path}"
                )
            ckpt_path = cls._find_checkpoint(root, ckpt_name)
            if mean_std_path is None:
                mean_std_path = cls._find_mean_std(root)
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
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state = ckpt["model_state_dict"]
        else:
            state = ckpt
        return self.model.load_state_dict(state, strict=strict)

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
