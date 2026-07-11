"""MDM text-to-motion pipeline.

Uses the Motius-native MDM Gaussian-diffusion ancestral sampler
(``motius.models.mdm.network``) to guarantee parity with the released
checkpoint, while exposing the Motius-native ``infer_t2m`` task interface.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

# MDM length limits (HumanML3D training config).
MDM_MIN_FRAMES = 40
MDM_MAX_FRAMES = 196


@PIPELINES.register_module()
class MDMPipeline(BasePipeline):
    """Inference pipeline for the MDM bundle."""

    BUNDLE_CLS = "motius.models.mdm.MDMBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        from motius.models.mdm.network import collate

        self._collate = collate
        if device is not None:
            self.to(device)

    def to(self, device):
        device = torch.device(device)
        self.bundle.net.to(device)
        self.bundle.mean = self.bundle.mean.to(device)
        self.bundle.std = self.bundle.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.bundle.net.parameters()).device

    @staticmethod
    def clamp_length(n_frames: int) -> int:
        ml = (int(n_frames) // 4) * 4
        return max(MDM_MIN_FRAMES, min(MDM_MAX_FRAMES, ml))

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        guidance_param: Optional[float] = None,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate HumanML3D-263 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: list of B target lengths in MDM frames (20 fps native),
                each already clamped/validated by the caller or clamped here.
            guidance_param: classifier-free guidance scale; defaults to the
                bundle's configured value.
            progress: show the per-step denoising progress bar.

        Returns:
            List of B arrays, each ``(length_i, 263)`` un-standardized.
        """
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        bundle = self.bundle
        net = bundle.net
        diffusion = bundle.diffusion
        device = self.device

        scale = bundle.guidance_param if guidance_param is None else float(guidance_param)
        lengths = [self.clamp_length(x) for x in lengths]
        n_frames = max(lengths)
        bs = len(captions)

        collate_args = [
            {"inp": torch.zeros(263, 1, n_frames), "tokens": None, "lengths": ml, "text": cap}
            for cap, ml in zip(captions, lengths)
        ]
        motion, model_kwargs = self._collate(collate_args)
        model_kwargs["y"] = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in model_kwargs["y"].items()
        }
        if scale != 1.0:
            model_kwargs["y"]["scale"] = torch.ones(bs, device=device) * scale
        if "text" in model_kwargs["y"]:
            model_kwargs["y"]["text_embed"] = net.encode_text(model_kwargs["y"]["text"])

        sample = diffusion.p_sample_loop(
            net,
            tuple(motion.shape),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=progress,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )  # (bs, 263, 1, n_frames)

        data_rep = getattr(net, "data_rep", "hml_vec")
        if data_rep != "hml_vec":
            raise RuntimeError(f"expected hml_vec data_rep, got {data_rep}")

        # (bs, 263, 1, T) -> (bs, T, 263), then denormalize.
        sample = sample[:, :, 0, :].permute(0, 2, 1).contiguous()
        sample = bundle.denormalize(sample).cpu().numpy().astype(np.float32)

        return [sample[i, : lengths[i]] for i in range(bs)]

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
