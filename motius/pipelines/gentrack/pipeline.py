"""GenTrack G1 inference using the same policy sampler as online training."""

from motius.motion.conditioning.hymotion_text import (
    encode_hymotion_text,
    encode_hymotion_instruction,
)
import torch
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from .sampling import sample_motion, latents_to_qpos


@PIPELINES.register_module()
class GenTrackPipeline(BasePipeline):
    BUNDLE_CLS = "motius.models.gentrack.GenTrackG1Bundle"

    def infer_text_to_motion(
        self,
        captions,
        num_frames=180,
        *,
        num_steps=None,
        guidance=None,
        initial_noise=None,
        **batch_overrides,
    ):
        captions = [captions] if isinstance(captions, str) else list(captions)
        lengths = (
            [int(num_frames)] * len(captions)
            if isinstance(num_frames, int)
            else list(num_frames)
        )
        if len(lengths) != len(captions):
            raise ValueError("num_frames must contain one length per caption")
        return self(
            {"caption": captions, "num_frames": lengths, **batch_overrides},
            num_steps=num_steps,
            guidance=guidance,
            initial_noise=initial_noise,
        )

    @torch.no_grad()
    def __call__(self, batch, *, num_steps=None, guidance=None, initial_noise=None):
        """Generate normalized motion and G1 qpos from cached dual text features."""
        if "text_vec_raw" in batch:
            features = batch
        else:
            features = encode_hymotion_text(self.bundle, batch["caption"])
        context = features["text_ctxt_raw"]
        context_lengths = features["text_ctxt_raw_length"]
        if isinstance(context, torch.Tensor):
            context = [
                value[: int(length)] for value, length in zip(context, context_lengths)
            ]
        lengths = torch.as_tensor(
            batch.get("tgt_length", batch.get("num_frames")), dtype=torch.long
        )
        normalized = sample_motion(
            self.bundle,
            features["text_vec_raw"],
            context,
            context_lengths,
            lengths,
            num_steps=num_steps,
            guidance=guidance,
            initial_noise=initial_noise,
        )
        return {
            "normalized_motion": normalized,
            "qpos": latents_to_qpos(self.bundle, normalized),
            "lengths": lengths,
        }

    def save_pretrained(self, save_directory, **kwargs):
        return self.bundle.save_pretrained(save_directory, **kwargs)
