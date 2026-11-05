"""Motius-native TMR architecture used by the released evaluator artifacts."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .network import ACTORStyleEncoder, ACTORStyleDecoder, TMRCore


_DEFAULT_ARCH = {
    "latent_dim": 256,
    "ff_size": 1024,
    "num_layers": 6,
    "num_heads": 4,
    "dropout": 0.1,
    "activation": "gelu",
}






@MODEL_BUNDLES.register_module()
class TMRBundle(ModelBundle):
    """Self-contained TMR bundle shared by SMPL-22 and G1 evaluators."""

    def __init__(
        self,
        motion_nfeats: int = 38,
        text_nfeats: int = 768,
        vae: bool = True,
        arch: Optional[Dict[str, Any]] = None,
        sample_mean: bool = True,
        fact: Optional[float] = None,
        **_: Any,
    ) -> None:
        super().__init__()
        misplaced=set(_) & {"lmd","temperature","threshold_selfsim"}
        if misplaced:raise TypeError("Configure training objectives on TMRTrainer: "+", ".join(sorted(misplaced)))
        architecture = deepcopy(_DEFAULT_ARCH)
        if arch:
            architecture.update(deepcopy(arch))
        motion_encoder = ACTORStyleEncoder(motion_nfeats, vae, **architecture)
        text_encoder = ACTORStyleEncoder(text_nfeats, vae, **architecture)
        motion_decoder = ACTORStyleDecoder(motion_nfeats, **architecture)
        self.tmr = TMRCore(
            motion_encoder,
            text_encoder,
            motion_decoder,
            vae=vae,
            sample_mean=sample_mean,
            fact=fact,
        )
        self._save_ckpt_modules.append("tmr")
        self._trainable_modules.append("tmr")
        self._module_checkpoint_formats["tmr"] = "full"

    def reconstruct_pair(self, text_inputs, motion_inputs):
        return self.tmr(text_inputs, motion_inputs)

    @staticmethod
    def _unwrap_tmr(module: nn.Module) -> TMRCore:
        while hasattr(module, "module") and isinstance(module.module, nn.Module):
            module = module.module
        return module


    @torch.inference_mode()
    def encode_text(self, inputs: Dict[str, Tensor], sample_mean: bool = True) -> Tensor:
        return self._unwrap_tmr(self.tmr).encode(
            inputs, modality="text", sample_mean=sample_mean
        )

    @torch.inference_mode()
    def encode_motion(self, inputs: Dict[str, Tensor], sample_mean: bool = True) -> Tensor:
        return self._unwrap_tmr(self.tmr).encode(
            inputs, modality="motion", sample_mean=sample_mean
        )


__all__ = ["TMRBundle"]
