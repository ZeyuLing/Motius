"""MotionGPT3 text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionGPT3Pipeline(BasePipeline):
    """Text-to-motion and motion-to-text pipeline for MotionGPT3."""

    BUNDLE_CLS = "motius.models.motiongpt3.MotionGPT3Bundle"

    def __init__(self, bundle, device=None, **kwargs):
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
    def clamp_length(n_frames: int, min_length: int = 40, max_length: int = 196) -> int:
        length = (int(n_frames) // 4) * 4
        return max(min_length, min(max_length, length))

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        stage: str = "test",
        temperature: float = 1.0,
    ) -> List[np.ndarray]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        model = self.bundle.model
        device = self.device
        lengths = [self.clamp_length(x) for x in lengths]

        outputs = model.lm.generate_conditional(
            list(captions),
            lengths=lengths,
            stage=stage,
            tasks=None,
        )
        sampled_token_latents, motion_mask = model.lm.sample_tokens(
            outputs,
            model.lm.device,
            temperature=temperature,
            cfg=model.guidance_scale,
            vae_mean_std_inv=model.vae.mean_std_inv,
        )
        z = sampled_token_latents.reshape(len(lengths), model.vae.latent_size, -1).permute(1, 0, 2)
        feats = model.vae.decode(z, lengths=lengths)
        if motion_mask is not None:
            feats = feats.clone()
            mask = motion_mask.to(device=feats.device, dtype=torch.bool)
            while mask.ndim < feats.ndim:
                mask = mask.unsqueeze(-1)
            feats = torch.where(mask, torch.zeros_like(feats), feats)
        feats = self.bundle.denormalize(feats).detach().cpu().numpy().astype(np.float32)
        return [feats[i, : lengths[i]] for i in range(len(lengths))]

    @torch.no_grad()
    def infer_m2t(
        self,
        motions: Sequence[Union[np.ndarray, torch.Tensor]],
        lengths: Optional[Sequence[int]] = None,
        *,
        stage: str = "test",
        with_len: bool = False,
    ) -> List[str]:
        """Caption denormalized HumanML3D-263 motions."""

        if lengths is None:
            lengths = [int(motion.shape[0]) for motion in motions]
        if len(motions) != len(lengths):
            raise ValueError("motions and lengths must have equal length")
        if not motions:
            return []

        normalized = []
        valid_lengths = []
        for motion, requested_length in zip(motions, lengths):
            if isinstance(motion, torch.Tensor):
                value = motion.detach().to(self.device, dtype=torch.float32)
            else:
                value = torch.from_numpy(
                    np.array(motion, dtype=np.float32, copy=True)
                ).to(self.device)
            if value.ndim != 2 or value.shape[-1] != 263:
                raise ValueError(
                    f"expected HML3D-263 motion shape (T,263), got {tuple(value.shape)}"
                )
            length = max(1, min(int(requested_length), int(value.shape[0]), 196))
            normalized.append(
                (value[:length] - self.bundle.mean.to(value))
                / self.bundle.std.to(value).clamp_min(1e-8)
            )
            valid_lengths.append(length)

        maximum = max(valid_lengths)
        motion_batch = torch.zeros(
            (len(normalized), maximum, 263),
            dtype=torch.float32,
            device=self.device,
        )
        for index, value in enumerate(normalized):
            motion_batch[index, : len(value)] = value

        model = self.bundle.model
        outputs = model.lm.generate_conditional(
            motion_feats=motion_batch,
            motion_encode_net=model.vae,
            lengths=valid_lengths,
            task="m2t",
            stage=stage,
            with_len=with_len,
        )
        return [str(value).strip().strip('"').strip() for value in outputs]

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
