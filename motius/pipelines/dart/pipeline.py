"""DART text-to-motion pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


from .inference import DARTInference
from motius.diffusion.dart import create_gaussian_diffusion


@PIPELINES.register_module()
class DARTPipeline(DARTInference, BasePipeline):
    """Inference pipeline for DART / DartControl HumanML3D rollout."""

    BUNDLE_CLS = "motius.models.dart.DARTBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        options = dict(bundle.inference_options)
        options.update(kwargs)
        self.artifact_dir = bundle.artifact_dir
        self.denoiser_args = bundle.denoiser_args
        self.vae_args = bundle.vae_args
        defaults = dict(
            seed_sequence="data/stand_20fps.pkl",
            guidance_param=5.0,
            respacing="",
            zero_noise=False,
            use_predicted_joints=False,
            fix_floor=False,
            coord_conversion="mbench",
            translation_source="smpl_transl",
            initial_transform="standard",
            rollout_body_type="smplx",
        )
        defaults.update(options)
        for key, value in defaults.items():
            setattr(self, key, value)
        if self.coord_conversion not in {"mbench", "none"}:
            raise ValueError(self.coord_conversion)
        if self.translation_source not in {
            "floor_aligned_smpl_transl",
            "floor_aligned_joints_pelvis",
            "joints_pelvis",
            "smpl_transl",
        }:
            raise ValueError(self.translation_source)
        if self.initial_transform == "official_flowmdm":
            self.initial_transform = "standard"
        if self.initial_transform not in {"standard", "canonical_seed", "identity"}:
            raise ValueError(self.initial_transform)
        from types import SimpleNamespace

        args = SimpleNamespace(**vars(self.denoiser_args.diffusion_args))
        args.respacing = self.respacing
        self.diffusion = create_gaussian_diffusion(args)
        self._dataset_cache = {}
        self._rollout_utility = None
        if device is not None:
            self.to(device)
        if options.get("load_dataset"):
            self._get_dataset(1)

    def to(self, device):
        self.bundle.to(device)
        self._dataset_cache.clear()
        self._rollout_utility = None
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.no_grad()
    def infer_t2m_smpl(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 0,
        sample_offset: int = 0,
        guidance_param: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[dict]:
        return self.generate_smpl_sequences(
            captions,
            lengths,
            seed=seed,
            sample_offset=sample_offset,
            guidance_param=guidance_param,
            show_progress=show_progress,
        )

    @torch.no_grad()
    def infer_t2m_motion135(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 0,
        sample_offset: int = 0,
        guidance_param: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        return self.generate_motion135(
            captions,
            lengths,
            seed=seed,
            sample_offset=sample_offset,
            guidance_param=guidance_param,
            show_progress=show_progress,
        )

    def infer_t2m(self, captions, lengths, **kwargs):
        return self.infer_t2m_motion135(captions, lengths, **kwargs)

    def __call__(self, captions, lengths, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)

    infer_text_to_motion = infer_t2m

    def _guided_denoiser(self, x, timesteps, y=None):
        y["uncond"] = False
        output = self.bundle.predict_latent(x, timesteps, y)
        y["uncond"] = True
        unconditional = self.bundle.predict_latent(x, timesteps, y)
        return unconditional + y["scale"] * (output - unconditional)

    def save_pretrained(self, save_directory, **kwargs):
        self.bundle.inference_options.update(
            {
                key: getattr(self, key)
                for key in [
                    "seed_sequence",
                    "guidance_param",
                    "respacing",
                    "zero_noise",
                    "use_predicted_joints",
                    "fix_floor",
                    "coord_conversion",
                    "translation_source",
                    "initial_transform",
                    "rollout_body_type",
                ]
            }
        )
        return self.bundle.save_pretrained(save_directory, **kwargs)
