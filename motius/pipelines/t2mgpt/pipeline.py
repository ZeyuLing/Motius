"""T2M-GPT text-to-motion pipeline.

Drives the Motius-native T2M-GPT implementation
(``motius.models.t2mgpt.network``): CLIP ViT-B/32 text feature ->
cross-conditional GPT autoregressive token sampling -> VQ-VAE
``forward_decoder`` -> 263-dim motion -> de-normalise to physical scale.

The inference path is byte-for-byte aligned with the gold-standard
``scripts/eval/t2mgpt_infer_hml3d263.py``:

* per-sample loop over the batch CLIP features;
* ``gpt.sample(feat[None], if_categorial=True)`` (categorical sampling);
* a ``token_idx = ones(1, 1)`` fallback if ``sample`` raises;
* ``vqvae.forward_decoder(token_idx)`` then ``* std + mean``.

⚠️ Upstream ``GPT_eval_multi`` keeps the GPT in ``train()`` mode at inference
(dropout stays active). We replicate that here — :meth:`infer_t2m` flips the GPT
to ``train()`` — because it is part of the released sampling distribution and is
    required for numerical parity. Fully independent of upstream checkout code.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class T2MGPTPipeline(BasePipeline):
    """Inference pipeline for the T2M-GPT bundle."""

    BUNDLE_CLS = "motius.models.t2mgpt.T2MGPTBundle"

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

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
        if_categorial: bool = True,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate HumanML3D-263 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: optional per-sample frame budget. If given, sample ``i`` is
                truncated to ``lengths[i]`` frames. T2M-GPT determines the length
                itself via the GPT EOS token, so this is *off by default* to stay
                bit-identical with ``t2mgpt_infer_hml3d263.py`` (which saves the
                full generated motion).
            if_categorial: categorical (vs greedy top-1) AR sampling. Defaults to
                ``True`` to match the parity script.
            progress: print per-sample token counts.

        Returns:
            List of B arrays, each ``(T_i, 263)`` un-standardized.
        """
        bundle = self.bundle
        device = self.device

        # Upstream keeps dropout active via train(); the VQ-VAE + CLIP stay eval.
        bundle.gpt.train()

        text_feat = bundle.encode_text(list(captions)).to(device)  # (B, 512)

        outputs: List[np.ndarray] = []
        for i in range(len(captions)):
            feat = text_feat[i]
            try:
                token_idx = bundle.gpt.sample(feat[None], if_categorial)
            except Exception:
                token_idx = torch.ones(1, 1, device=device, dtype=torch.long)
            pred = bundle.vqvae.forward_decoder(token_idx)  # (1, T, 263), normalized
            motion = bundle.denormalize(pred[0].float())     # (T, 263), raw scale
            if lengths is not None and lengths[i] is not None:
                motion = motion[: int(lengths[i])]
            outputs.append(motion.detach().cpu().numpy().astype(np.float32))
            if progress:
                print(
                    f"[t2mgpt] {i + 1}/{len(captions)} tokens={int(token_idx.shape[1])} "
                    f"-> {outputs[-1].shape}",
                    flush=True,
                )
        return outputs

    @torch.no_grad()
    def infer_motion_reconstruction(
        self,
        motions: Sequence[np.ndarray],
        batch_size: int = 64,
    ) -> List[np.ndarray]:
        """Round-trip physical-scale HumanML3D-263 motions through the VQ-VAE."""

        bundle = self.bundle
        groups = {}
        for index, motion in enumerate(motions):
            value = np.asarray(motion, dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != 263:
                raise ValueError(f"expected (T,263), got {value.shape}")
            usable = (len(value) // 4) * 4
            if usable < 4:
                raise ValueError("T2M-GPT reconstruction requires at least 4 frames")
            groups.setdefault(usable, []).append((index, value[:usable]))

        outputs: List[np.ndarray] = [None] * len(motions)
        mean = bundle.mean.to(self.device)
        std = bundle.std.to(self.device).clamp_min(1e-8)
        for length, items in groups.items():
            for start in range(0, len(items), max(1, int(batch_size))):
                chunk = items[start : start + max(1, int(batch_size))]
                batch = torch.from_numpy(np.stack([value for _, value in chunk])).to(
                    self.device
                )
                reconstructed, _, _ = bundle.vqvae((batch - mean) / std)
                reconstructed = bundle.denormalize(reconstructed.float()).cpu().numpy()
                for row, (index, _) in enumerate(chunk):
                    outputs[index] = reconstructed[row, :length].astype(np.float32)
        return outputs

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)

    infer_text_to_motion = infer_t2m
    infer_reconstruction = infer_motion_reconstruction
