"""MoMask text-to-motion pipeline.

Drives the Motius-native MoMask implementation
(``motius.models.momask.network``): CLIP text features ->
``MaskTransformer.generate`` (masked iterative decoding of the base token map,
classifier-free guidance) -> ``ResidualTransformer.generate`` (quantizers
1..5) -> ``RVQVAE.forward_decoder`` -> 263-dim motion -> de-normalise to
physical scale.

Logic matches the parity script ``scripts/eval/momask_infer_h3d_test.py``
(same cond_scale / time_steps / topkr / temperature / unit_length=4 / 20 fps),
so reproduced metrics align with the released checkpoints. Fully independent of
the original repo.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

# MoMask runs at 20 fps with unit_length = 4 (1 token = 4 frames).
MOMASK_UNIT_LENGTH = 4
# MoMask was trained on HumanML3D 60..196 frames; keep at least 10 tokens.
MOMASK_MIN_FRAMES = 40
MOMASK_MAX_FRAMES = 196


@PIPELINES.register_module()
class MoMaskPipeline(BasePipeline):
    """Inference pipeline for the MoMask bundle."""

    BUNDLE_CLS = "motius.models.momask.MoMaskBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @staticmethod
    def clamp_length(n_frames: int) -> int:
        ml = (int(n_frames) // MOMASK_UNIT_LENGTH) * MOMASK_UNIT_LENGTH
        return max(MOMASK_MIN_FRAMES, min(MOMASK_MAX_FRAMES, ml))

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
        cond_scale: float = 4.0,
        time_steps: int = 10,
        topkr: float = 0.9,
        temperature: float = 1.0,
        res_cond_scale: float = 5.0,
        res_temperature: float = 1.0,
        gumbel_sample: bool = False,
        clamp: bool = True,
    ) -> List[np.ndarray]:
        """Generate HumanML3D-263 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: optional per-sample target length in **frames** (20 fps).
                When ``None``, lengths are sampled from the bundle's length
                estimator (requires ``load_length_estimator=True``).
            cond_scale / time_steps / topkr / temperature / gumbel_sample:
                base masked-decoding params (parity defaults cond_scale=4,
                time_steps=10, topkr=0.9, temperature=1.0).
            res_cond_scale / res_temperature: residual-transformer params
                (parity defaults cond_scale=5, temperature=1.0).
            clamp: clamp/round given ``lengths`` to ``[40, 196]`` multiples of 4.

        Returns:
            List of B arrays, each ``(T_i, 263)`` un-standardized.
        """
        from motius.models.momask.network import (
            estimate_token_lengths,
            generate_motion,
        )

        bundle = self.bundle
        device = self.device
        captions = list(captions)
        bs = len(captions)

        if lengths is not None:
            if len(lengths) != bs:
                raise ValueError("captions and lengths must have equal length")
            frame_lens = [self.clamp_length(x) if clamp else int(x) for x in lengths]
            token_lens = torch.tensor(
                [ml // MOMASK_UNIT_LENGTH for ml in frame_lens],
                dtype=torch.long, device=device,
            )
        else:
            if bundle.length_estimator is None:
                raise RuntimeError(
                    "lengths=None requires a length estimator "
                    "(build bundle with load_length_estimator=True)."
                )
            token_lens = estimate_token_lengths(
                bundle.t2m_transformer, bundle.length_estimator, captions
            ).to(device)
            frame_lens = [int(t) * MOMASK_UNIT_LENGTH for t in token_lens.tolist()]

        pred_motions = generate_motion(
            bundle.t2m_transformer,
            bundle.res_transformer,
            bundle.vq_model,
            captions,
            token_lens,
            cond_scale=cond_scale,
            time_steps=time_steps,
            temperature=temperature,
            topk_filter_thres=topkr,
            gsample=gumbel_sample,
            res_cond_scale=res_cond_scale,
            res_temperature=res_temperature,
        )  # (B, T, 263) normalized

        data = bundle.denormalize(pred_motions.float()).cpu().numpy().astype(np.float32)
        return [data[i, : frame_lens[i]] for i in range(bs)]

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
