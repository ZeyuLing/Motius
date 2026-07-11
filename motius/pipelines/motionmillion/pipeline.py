"""MotionMillion / "Go to Zero" text-to-motion pipeline.

Drives the Motius-native MotionMillion implementation
(``motius.models.motionmillion.network``): Flan-T5-XL text features -> LLaMA
autoregressive token sampling (greedy, EOS-stopped) -> FSQ de-quantize +
HumanVQVAE decoder -> 272-dim motion -> de-normalise to physical scale.

Matches the released eval generation path
(``utils.eval_trans.evaluation_transformer_motionmillion`` ->
``LLaMAHF.sample`` + ``HumanVQVAE.forward_decoder``) so reproduced metrics align
with the released checkpoints. Generation stops at the EOS token or after
``max_sample_steps`` motion tokens; with the FSQ temporal downsample factor
``stride_t ** down_t = 2`` each token is ~2 frames @ 30 fps. We default to 150
tokens (~300 frames), which covers the full HumanML3D length range — the
released script hard-codes a 50-token (~100-frame) cap that truncates long
motions and is **not** used here. Fully independent of the original repo.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionMillionPipeline(BasePipeline):
    """Inference pipeline for the MotionMillion bundle."""

    BUNDLE_CLS = "motius.models.motionmillion.MotionMillionBundle"

    def __init__(self, bundle, device: Optional[str] = None, max_sample_steps: int = 150, **kwargs):
        super().__init__(bundle, **kwargs)
        self.max_sample_steps = int(max_sample_steps)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
        max_sample_steps: Optional[int] = None,
        if_categorial: bool = False,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate MotionMillion-272 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: optional per-sample max frame budget. If given, sample #i is
                truncated to ``lengths[i]`` frames (matching the released eval,
                which truncates predictions to the GT length). Generation length
                is otherwise determined by the model's EOS token.
            max_sample_steps: override the max number of AR motion tokens.
            if_categorial: categorical (vs greedy top-1) AR sampling.

        Returns:
            List of B arrays, each ``(T_i, 272)`` un-standardized.
        """
        bundle = self.bundle
        steps = self.max_sample_steps if max_sample_steps is None else int(max_sample_steps)
        feat, y_mask = bundle.encode_text(list(captions))

        ar_dtype = next(bundle.ar.parameters()).dtype
        feat = feat.to(ar_dtype)
        use_autocast = self.device.type == "cuda" and ar_dtype != torch.float32
        autocast = (
            torch.autocast(device_type="cuda", dtype=ar_dtype) if use_autocast else nullcontext()
        )

        outputs: List[np.ndarray] = []
        for i in range(len(captions)):
            with autocast:
                idx = bundle.ar.sample_cached(
                    feat[i : i + 1], y_mask[i : i + 1],
                    if_categorial=if_categorial, max_sample_steps=steps,
                )
                pred = bundle.vqvae.forward_decoder(idx)  # (1, T, 272), normalized
            motion = bundle.denormalize(pred[0].float())  # (T, 272), raw scale
            if lengths is not None:
                motion = motion[: int(lengths[i])]
            outputs.append(motion.cpu().numpy().astype(np.float32))
            if progress:
                print(f"[mm] {i + 1}/{len(captions)} tokens={idx.shape[1]} -> {outputs[-1].shape}", flush=True)
        return outputs

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
