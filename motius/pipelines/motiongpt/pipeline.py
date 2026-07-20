"""MotionGPT text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionGPTPipeline(BasePipeline):
    """Inference pipeline for MotionGPT on HumanML3D-263."""

    BUNDLE_CLS = "motius.models.motiongpt.MotionGPTBundle"

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
        prompt_mode: Optional[str] = None,
        seed: Optional[int] = None,
        do_sample: bool = True,
        progress: bool = False,
    ) -> List[np.ndarray]:
        outputs = self.bundle.infer_hml263(
            captions,
            lengths=lengths,
            prompt_mode=prompt_mode,
            seed=seed,
            do_sample=do_sample,
        )
        if progress:
            for i, arr in enumerate(outputs):
                print(f"[motiongpt] {i + 1}/{len(outputs)} shape={arr.shape}", flush=True)
        return outputs

    @torch.no_grad()
    def infer_m2t(
        self,
        motions: Sequence[Union[np.ndarray, torch.Tensor]],
        lengths: Optional[Sequence[int]] = None,
        with_len: bool = False,
        pad_to_batch_max: bool = False,
        progress: bool = False,
    ) -> List[str]:
        """Generate text descriptions from HML3D-263 motions.

        ``motions`` are expected to be denormalized HumanML3D-263 features, the
        same format saved by the T2M baseline scripts. MotionGPT's VQ-VAE was
        trained on normalized features, so normalization is applied here before
        tokenization. Set ``pad_to_batch_max=True`` only to reproduce the
        released evaluator's batch-dependent zero-padding behavior.
        """
        if lengths is None:
            lengths = [int(m.shape[0]) for m in motions]
        if len(motions) != len(lengths):
            raise ValueError("motions and lengths must have equal length")

        bundle = self.bundle
        normalized = []
        mean = bundle.mean.to(self.device)
        std = bundle.std.to(self.device)
        for motion, length in zip(motions, lengths):
            if isinstance(motion, torch.Tensor):
                motion_t = motion.detach().to(self.device, dtype=torch.float32)
            else:
                motion_t = torch.from_numpy(
                    np.array(motion, dtype=np.float32, copy=True)
                ).to(self.device)
            if motion_t.ndim != 2 or motion_t.shape[-1] != 263:
                raise ValueError(f"expected HML3D-263 motion shape (T,263), got {tuple(motion_t.shape)}")
            length = max(1, min(int(length), int(motion_t.shape[0])))
            motion_t = motion_t[:length]
            motion_norm = (motion_t - mean) / std.clamp_min(1e-8)
            normalized.append(motion_norm)

        if pad_to_batch_max:
            # MotionGPT's released M2T evaluator collates normalized motions
            # into a zero-padded batch and encodes the padded tensor without
            # passing the original lengths to the VQ-VAE. Reproduce that
            # behavior because the resulting motion tokens are batch-dependent.
            maximum = max(int(value.shape[0]) for value in normalized)
            motion_batch = normalized[0].new_zeros(
                (len(normalized), maximum, normalized[0].shape[-1])
            )
            for index, value in enumerate(normalized):
                motion_batch[index, : value.shape[0]] = value
            tokens, _ = bundle.vae.encode(motion_batch)
            motion_tokens = [tokens[index] for index in range(len(normalized))]
            token_lengths = [int(tokens.shape[1])] * len(normalized)
        else:
            motion_tokens = []
            token_lengths = []
            for motion_norm in normalized:
                token, _ = bundle.vae.encode(motion_norm.unsqueeze(0))
                motion_tokens.append(token[0])
                token_lengths.append(int(token.shape[1]))

        texts = bundle.lm.generate_conditional(
            motion_tokens=motion_tokens,
            lengths=token_lengths,
            task="m2t",
            with_len=with_len,
            stage="test",
        )
        texts = [str(x).strip() for x in texts]
        if progress:
            for i, text in enumerate(texts):
                print(f"[motiongpt:m2t] {i + 1}/{len(texts)} {text}", flush=True)
        return texts

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
