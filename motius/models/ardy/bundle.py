"""Motius model bundle for NVIDIA ARDY checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


ARDY_CHECKPOINTS = {
    "core": "nvidia/ARDY-Core-RP-20FPS-Horizon40",
    "core40": "nvidia/ARDY-Core-RP-20FPS-Horizon40",
    "core8": "nvidia/ARDY-Core-RP-20FPS-Horizon8",
    "g1": "nvidia/ARDY-G1-RP-25FPS-Horizon52",
    "g152": "nvidia/ARDY-G1-RP-25FPS-Horizon52",
    "g18": "nvidia/ARDY-G1-RP-25FPS-Horizon8",
}


def _normalize_model_name(value: str) -> str:
    return ARDY_CHECKPOINTS.get(value.lower(), value)


@MODEL_BUNDLES.register_module()
class ARDYBundle(ModelBundle):
    """Load and own one released ARDY denoiser/tokenizer pair.

    The official text encoder is gated LLM2Vec/Llama 3. Pass
    ``text_encoder=False`` when supplying precomputed ``text_feat`` tensors to
    the pipeline, or pass a compatible callable encoder instance.
    """

    SUPPORTED_TASKS = {
        "text_to_motion": "offline autoregressive text-to-motion generation",
        "streaming_text_to_motion": "stateful one-horizon generation with online prompt changes",
        "kinematic_control": "root path, full-body keyframe, and sparse end-effector constraints",
    }
    CHECKPOINTS = ARDY_CHECKPOINTS

    def __init__(
        self,
        model_name: str = "core",
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        text_encoder: Any = None,
        text_encoder_mode: str = "local",
        text_encoder_url: Optional[str] = None,
        text_encoder_fp32: bool = False,
        load_model: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = _normalize_model_name(str(model_name))
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.checkpoints_dir = checkpoints_dir
        self.text_encoder = text_encoder
        self.text_encoder_mode = text_encoder_mode
        self.text_encoder_url = text_encoder_url
        self.text_encoder_fp32 = bool(text_encoder_fp32)
        self._model = None
        self._model_config = None
        if load_model:
            self.load_model()

    @property
    def model(self):
        return self.load_model()

    @property
    def skeleton(self):
        return self.model.skeleton

    @property
    def motion_rep(self):
        return self.model.motion_rep

    @property
    def fps(self) -> float:
        return float(self.motion_rep.fps)

    @property
    def horizon(self) -> int:
        return int(self.model.gen_horizon_len)

    @property
    def device(self) -> torch.device:
        model = self.model
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device(model.device)

    def load_model(self):
        if self._model is not None:
            return self._model

        try:
            from .network import load_model
        except ImportError as exc:
            if exc.name == "vector_quantize_pytorch":
                raise ImportError(
                    "ARDY requires the optional dependencies. Install with "
                    "`pip install -e '.[ardy]'`."
                ) from exc
            raise

        model_name = self.model_name
        if self.checkpoint_path:
            model_name = Path(self.checkpoint_path).name
        self._model, self._model_config = load_model(
            model_name,
            device=self.device_name,
            text_encoder=self.text_encoder,
            text_encoder_mode=self.text_encoder_mode,
            text_encoder_url=self.text_encoder_url,
            text_encoder_fp32=self.text_encoder_fp32,
            return_config=True,
            checkpoints_dir=self.checkpoints_dir,
            model_path=self.checkpoint_path,
        )
        return self._model

    def to(self, *args, **kwargs):
        if self._model is not None:
            self._model.to(*args, **kwargs)
            if args:
                self._model.device = torch.device(args[0])
            elif kwargs.get("device") is not None:
                self._model.device = torch.device(kwargs["device"])
        return super().to(*args, **kwargs)

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(str(pretrained_model_name_or_path))
        config = dict(kwargs)
        if path.is_dir():
            config.setdefault("checkpoint_path", str(path))
            config.setdefault("model_name", path.name)
        else:
            config.setdefault("model_name", _normalize_model_name(str(pretrained_model_name_or_path)))
        return config

    def forward(self, *args, **kwargs):  # pragma: no cover - pipeline owns sampling
        raise NotImplementedError("Use ARDYPipeline for autoregressive inference")


__all__ = ["ARDYBundle", "ARDY_CHECKPOINTS"]
