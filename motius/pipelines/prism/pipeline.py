"""Public Motius inference API for PRISM 1.0 and PRISM-KT."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import torch

from motius.motion.representation.convert import smpl_to_motion135
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class PRISMPipeline(BasePipeline):
    """T2M, TP2M, and sequential generation with PRISM."""

    BUNDLE_CLS = "motius.models.prism.PRISMBundle"

    @property
    def backend(self):
        return self.bundle.load_model()

    def _format_output(self, result: dict) -> dict:
        smpl = result["smplx_dict"]
        normalized = result["motion_vec"]
        native = normalized.reshape(normalized.shape[0], normalized.shape[1], -1)
        native = self.bundle.processor.denormalize(native.float())[0].cpu().numpy()
        motion135 = smpl_to_motion135(
            smpl["global_orient"], smpl["body_pose"], smpl["transl"]
        )
        return {
            "motion_138": native.astype(np.float32, copy=False),
            "motion_135": np.asarray(motion135, dtype=np.float32),
            "smpl": smpl,
            "fps": float(smpl.get("mocap_framerate", 30.0)),
            "representation": "prism_motion138",
            "variant": self.bundle.variant,
        }

    @torch.inference_mode()
    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        num_frames: Union[int, Sequence[int]] = 129,
        *,
        negative_prompt: str = "",
        prefix_motion_path: Optional[str] = None,
        condition_num_frames: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        kafs_mode: Optional[str] = None,
        canonicalize: bool = False,
        use_blend: bool = True,
        ar_condition_frames: int = 5,
        seed: Optional[int] = None,
        **kwargs,
    ) -> dict:
        if seed is not None:
            torch.manual_seed(int(seed))
        if kafs_mode is not None:
            self.backend.set_kafs_alpha(kafs_mode)
        result = self.backend(
            prompts=list(prompts) if not isinstance(prompts, str) else prompts,
            negative_prompt=negative_prompt,
            first_frame_motion_path=prefix_motion_path,
            condition_num_frames=int(condition_num_frames),
            num_frames_per_segment=(
                list(num_frames) if not isinstance(num_frames, int) else int(num_frames)
            ),
            num_inference_steps=int(num_inference_steps),
            guidance_scale=float(guidance_scale),
            normalize=bool(canonicalize),
            use_blend=bool(use_blend),
            ar_condition_frames=int(ar_condition_frames),
            return_motion_vec=True,
            **kwargs,
        )
        return self._format_output(result)

    def text_to_motion(self, caption: str, num_frames: int = 129, **kwargs) -> dict:
        return self.generate(caption, num_frames, use_blend=False, **kwargs)

    def temporal_condition(
        self,
        caption: str,
        prefix_motion_path: str,
        num_frames: int = 129,
        condition_num_frames: int = 5,
        **kwargs,
    ) -> dict:
        return self.generate(
            caption,
            num_frames,
            prefix_motion_path=prefix_motion_path,
            condition_num_frames=condition_num_frames,
            use_blend=False,
            **kwargs,
        )

    def sequential_generation(
        self,
        prompts: Sequence[str],
        segment_frames: Union[int, Sequence[int]] = 129,
        **kwargs,
    ) -> dict:
        return self.generate(prompts, segment_frames, **kwargs)

    def infer_t2m(self, captions, lengths, **kwargs):
        if isinstance(captions, str):
            return self.text_to_motion(captions, int(lengths), **kwargs)
        if isinstance(lengths, int):
            lengths = [lengths] * len(captions)
        return [
            self.text_to_motion(caption, int(length), **kwargs)
            for caption, length in zip(captions, lengths)
        ]
