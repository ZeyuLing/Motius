"""MoGenTS text-to-motion pipeline.

Drives the Motius-native MoGenTS implementation:

CLIP text -> 1D auxiliary MaskTransformer + 2D spatial-temporal
MaskTransformer -> 1D/2D residual transformers -> dual-stream RVQ-VAE decoder
-> HumanML3D-263 -> de-normalized physical scale.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

MOGENTS_UNIT_LENGTH = 4
MOGENTS_MIN_FRAMES = 40
MOGENTS_MAX_FRAMES = 196


@PIPELINES.register_module()
class MoGenTSPipeline(BasePipeline):
    """Inference pipeline for the MoGenTS bundle."""

    BUNDLE_CLS = "motius.models.mogents.MoGenTSBundle"

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
        ml = (int(n_frames) // MOGENTS_UNIT_LENGTH) * MOGENTS_UNIT_LENGTH
        return max(MOGENTS_MIN_FRAMES, min(MOGENTS_MAX_FRAMES, ml))

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
        """Generate HumanML3D-263 motions (physical scale) from text."""
        from motius.models.mogents.network import (
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
                [ml // MOGENTS_UNIT_LENGTH for ml in frame_lens],
                dtype=torch.long,
                device=device,
            )
        else:
            if bundle.length_estimator is None:
                raise RuntimeError(
                    "lengths=None requires a length estimator "
                    "(build bundle with load_length_estimator=True)."
                )
            token_lens = estimate_token_lengths(
                bundle.mask_transformer_ts, bundle.length_estimator, captions
            ).to(device)
            frame_lens = [int(t) * MOGENTS_UNIT_LENGTH for t in token_lens.tolist()]

        pred_motions = generate_motion(
            bundle.mask_transformer_aux,
            bundle.mask_transformer_ts,
            bundle.res_transformer_aux,
            bundle.res_transformer_ts,
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
            n_joint_groups=bundle.n_joint_groups,
        )

        data = bundle.denormalize(pred_motions.float()).cpu().numpy().astype(np.float32)
        return [data[i, : frame_lens[i]] for i in range(bs)]

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
