"""InterMask two-person text-to-motion pipeline."""

from __future__ import annotations

from typing import Optional, Sequence
import random
import numpy as np
import torch
from .sampling import generate as generate_tokens
from motius.motion.representation.intermask import unpack_motion

def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class InterMaskPipeline(BasePipeline):
    """Generate InterHuman-262 or Inter-X interaction motion tokens."""

    BUNDLE_CLS = "motius.models.intermask.InterMaskBundle"

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: str | Sequence[str],
        *,
        motion_len: int = 120,
        seed: Optional[int] = None,
        return_numpy: bool = True,
        cond_scale: Optional[float] = None,
        time_steps: Optional[int] = None,
        topk_filter_thres: Optional[float] = None,
        temperature: Optional[float] = None,
    ):
        """Generate InterMask motion.

        InterHuman returns ``(B, T, 2, 262)``. InterX returns ``(B, T, 56, 12)``.
        """
        prompts = [captions] if isinstance(captions, str) else list(captions)
        if not prompts:
            raise ValueError("InterMaskPipeline needs at least one text prompt")
        if int(motion_len) < 16 or int(motion_len) > 300 or int(motion_len) % 4:
            raise ValueError("InterMask motion_len must be a multiple of four between 16 and 300")
        vq, trans = self.bundle.load_model()
        _seed_everything(seed)
        device = next(trans.parameters()).device
        motion_lens = torch.full((len(prompts),), int(motion_len), dtype=torch.long, device=device)
        ids_length = motion_lens // 4
        ids = generate_tokens(trans,
            prompts,
            ids_length,
            int(time_steps or self.bundle.time_steps),
            float(cond_scale if cond_scale is not None else self.bundle.cond_scale),
            topk_filter_thres=float(topk_filter_thres if topk_filter_thres is not None else self.bundle.topk_filter_thres),
            temperature=float(temperature if temperature is not None else self.bundle.temperature),
        )
        ids1, ids2 = ids[:, : ids.shape[1] // 2], ids[:, ids.shape[1] // 2 :]
        motion1 = unpack_motion(self.bundle.decode_tokens(ids1.unsqueeze(-1).to(device)), self.bundle.dataset_name)
        motion2 = unpack_motion(self.bundle.decode_tokens(ids2.unsqueeze(-1).to(device)), self.bundle.dataset_name)
        motion = torch.cat([motion1, motion2], dim=-1).detach().cpu().numpy()
        if self.bundle.dataset_name == "interhuman":
            motion = motion.reshape(motion.shape[0], motion.shape[1], 2, -1)
            motion = motion * self.bundle._std + self.bundle._mean if self.bundle._mean is not None and self.bundle._std is not None else motion
        else:
            motion = motion.reshape(motion.shape[0], motion.shape[1], 56, 12)
        if return_numpy:
            return motion
        return torch.from_numpy(motion)


    def __call__(self, captions, **kwargs):
        return self.infer_t2m(captions, **kwargs)

    infer_text_to_multi_person_motion = infer_t2m


__all__ = ["InterMaskPipeline"]
