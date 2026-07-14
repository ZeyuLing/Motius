"""FlowMDM text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class FlowMDMPipeline(BasePipeline):
    """Inference pipeline for the FlowMDM bundle."""

    BUNDLE_CLS = "motius.models.flowmdm.FlowMDMBundle"

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

    def resolve_length(self, n_frames: int) -> int:
        """Resolve a duration without changing the official BABEL protocol."""

        if getattr(self.bundle, "dataset", "humanml") == "babel":
            length = int(n_frames)
            if length < 30 or length > 200:
                raise ValueError(f"BABEL segment length must be in [30, 200], got {length}")
            return length
        return self.clamp_length(int(n_frames))

    @staticmethod
    def _clear_embedding_cache(sampler) -> None:
        model = getattr(sampler, "model", None)
        if model is None:
            return
        device = next(model.parameters()).device
        for attr in ("emb_hash", "emb_forcemask_hash"):
            if hasattr(model, attr):
                setattr(model, attr, torch.tensor(-1, device=device, dtype=torch.long))

    @torch.no_grad()
    def _sample_one(
        self,
        caption: str,
        length: int,
        inpainting: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        device = self.device
        mask = torch.ones((1, length), device=device, dtype=torch.bool)
        y = {
            "mask": mask,
            "lengths": torch.tensor([length], dtype=torch.long, device=device),
            "text": [caption],
            "tokens": [""],
        }
        if inpainting is not None:
            inpainting_mask, inpainted_motion = inpainting
            y["inpainting_mask"] = inpainting_mask
            y["inpainted_motion"] = inpainted_motion
        model_kwargs = {"y": y}
        self._clear_embedding_cache(self.bundle.sampler)
        sample = self.bundle.sampler.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
        )
        return sample[0, :, 0, :length].permute(1, 0).contiguous()

    @torch.no_grad()
    def _sample_sequence(self, captions: Sequence[str], lengths: Sequence[int]) -> torch.Tensor:
        device = self.device
        lengths_t = torch.tensor(list(lengths), dtype=torch.long, device=device)
        model_kwargs = {
            "y": {
                "mask": torch.ones((len(captions),), device=device, dtype=torch.bool),
                "lengths": lengths_t,
                "text": [str(c) for c in captions],
                "tokens": [""] * len(captions),
                "scale": torch.ones(len(captions), device=device) * self.bundle.guidance_param,
            }
        }
        self._clear_embedding_cache(self.bundle.sampler)
        sample = self.bundle.sampler.p_sample_loop(
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
        )
        total = int(lengths_t.sum().item())
        return sample[0, :, 0, :total].permute(1, 0).contiguous()

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 42,
        shard_index: int = 0,
        sample_offset: int = 0,
    ) -> List[np.ndarray]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        out: List[np.ndarray] = []
        for i, (caption, length) in enumerate(zip(captions, lengths)):
            length = self.resolve_length(int(length))
            torch.manual_seed(int(seed) + int(shard_index) * 100000 + int(sample_offset) + i)
            pred_norm = self._sample_one(str(caption), length)
            pred = self.bundle.denormalize(pred_norm).detach().cpu().numpy().astype(np.float32)
            out.append(pred)
        return out

    def _build_prefix_inpainting(
        self,
        gt_features: np.ndarray,
        length: int,
        condition_num_frames: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if int(condition_num_frames) < 1:
            raise ValueError("condition_num_frames must be >= 1")
        if gt_features.shape[0] < length:
            raise ValueError(f"GT clip too short: {gt_features.shape[0]} < {length}")
        device = self.device
        gt = torch.as_tensor(
            gt_features[:length].astype(np.float32),
            dtype=torch.float32,
            device=device,
        )
        gt_norm = (gt - self.bundle.mean.to(gt)) / self.bundle.std.to(gt)
        inpainted = torch.zeros((1, gt_norm.shape[1], 1, length), device=device)
        mask = torch.zeros_like(inpainted, dtype=torch.bool)
        n_cond = min(int(condition_num_frames), int(length))
        obs = torch.arange(n_cond, device=device)
        inpainted[0, :, 0, obs] = gt_norm[obs].transpose(0, 1)
        mask[0, :, 0, obs] = True
        return mask, inpainted

    @torch.no_grad()
    def infer_tp2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        gt_features: Sequence[np.ndarray],
        condition_num_frames: int,
        seed: int = 42,
        shard_index: int = 0,
        sample_offset: int = 0,
    ) -> List[np.ndarray]:
        """Generate FlowMDM TP2M samples using prefix-frame inpainting."""
        if not (len(captions) == len(lengths) == len(gt_features)):
            raise ValueError("captions, lengths, and gt_features must have equal length")
        out: List[np.ndarray] = []
        for i, (caption, raw_len, gt) in enumerate(zip(captions, lengths, gt_features)):
            length = self.resolve_length(int(raw_len))
            torch.manual_seed(int(seed) + int(shard_index) * 100000 + int(sample_offset) + i)
            inpainting = self._build_prefix_inpainting(gt, length, condition_num_frames)
            pred_norm = self._sample_one(str(caption), length, inpainting=inpainting)
            pred = self.bundle.denormalize(pred_norm).detach().cpu().numpy().astype(np.float32)
            out.append(pred)
        return out

    @torch.no_grad()
    def infer_sequential_t2m(
        self,
        captions_per_sample: Sequence[Sequence[str]],
        lengths_per_sample: Sequence[Sequence[int]],
        seed: int = 42,
        shard_index: int = 0,
        sample_offset: int = 0,
    ) -> List[np.ndarray]:
        if len(captions_per_sample) != len(lengths_per_sample):
            raise ValueError("captions_per_sample and lengths_per_sample must have equal length")
        out: List[np.ndarray] = []
        for i, (captions, lengths) in enumerate(zip(captions_per_sample, lengths_per_sample)):
            if len(captions) != len(lengths):
                raise ValueError(
                    f"sample {i} has {len(captions)} captions but {len(lengths)} lengths"
                )
            if not captions:
                raise ValueError(f"sample {i} has no captions")
            seg_lengths = [self.resolve_length(int(n)) for n in lengths]
            torch.manual_seed(int(seed) + int(shard_index) * 100000 + int(sample_offset) + i)
            pred_norm = self._sample_sequence(captions, seg_lengths)
            pred = self.bundle.denormalize(pred_norm).detach().cpu().numpy().astype(np.float32)
            out.append(pred)
        return out

    infer_multi_prompt_t2m = infer_sequential_t2m

    def __call__(self, captions, lengths, **kwargs):
        if kwargs.pop("tp2m", False):
            return self.infer_tp2m(captions, lengths, **kwargs)
        if kwargs.pop("sequential", False):
            return self.infer_sequential_t2m(captions, lengths, **kwargs)
        if len(captions) > 0 and not isinstance(captions[0], str):
            return self.infer_sequential_t2m(captions, lengths, **kwargs)
        return self.infer_t2m(captions, lengths, **kwargs)
