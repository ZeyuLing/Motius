"""MotionStreamer text-to-motion pipeline.

Drives the Motius-native MotionStreamer implementation
(``motius.models.motionstreamer.network``): SentenceT5-XXL text features ->
LLaMA autoregressive transformer with classifier-free guidance and per-token
diffusion sampling -> latent tokens -> causal TAE decoder -> 272-dim motion.

Matches the upstream eval generation path
(``utils.eval_trans.evaluation_transformer_272_single`` ->
``LLaMAHF.sample_for_eval_CFG`` + ``Causal_HumanTAE.forward_decoder``) so the
reproduced metrics align with the released checkpoints.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

# MotionStreamer token unit length (TAE temporal downsample = stride_t**down_t = 4).
MS_UNIT_LENGTH = 4
# Block size 78 -> at most 77 latent tokens after the text-condition slot.
MS_MAX_TOKENS = 77


def _to_2d_latents(latents: torch.Tensor) -> torch.Tensor:
    """Normalize MotionStreamer sampler outputs to ``(tokens, latent_dim)``."""
    if latents.ndim == 3:
        latents = latents.squeeze(0)
    if latents.ndim == 2 and latents.shape[0] == 16 and latents.shape[1] != 16:
        latents = latents.transpose(0, 1).contiguous()
    if latents.ndim != 2 or latents.shape[-1] != 16:
        raise ValueError(f"expected MotionStreamer latents shaped (*, 16), got {tuple(latents.shape)}")
    return latents.contiguous()


@PIPELINES.register_module()
class MotionStreamerPipeline(BasePipeline):
    """Inference pipeline for the MotionStreamer bundle."""

    BUNDLE_CLS = "motius.models.motionstreamer.MotionStreamerBundle"

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
        """Clamp a target frame count to a valid (token-aligned) motion length."""
        n_tokens = int(n_frames) // MS_UNIT_LENGTH
        n_tokens = max(1, min(MS_MAX_TOKENS, n_tokens))
        return n_tokens * MS_UNIT_LENGTH

    @torch.no_grad()
    def infer_t2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        guidance_param: Optional[float] = None,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate MotionStreamer-272 motions (physical scale) from text.

        Args:
            captions: list of B text prompts.
            lengths: list of B target lengths in frames (30 fps native). Each is
                clamped to a token-aligned length.
            guidance_param: classifier-free guidance scale; defaults to the
                bundle's configured value (4.0).
            progress: optional progress print of per-sample generation.

        Returns:
            List of B arrays, each ``(length_i, 272)`` un-standardized.
        """
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        bundle = self.bundle
        if bundle.text_model is None:
            raise RuntimeError(
                "MotionStreamerBundle was built with load_text_model=False; "
                "the SentenceT5 text encoder is required for generation."
            )
        device = self.device
        scale = bundle.guidance_param if guidance_param is None else float(guidance_param)

        outputs: List[np.ndarray] = []
        for i, (cap, raw_len) in enumerate(zip(captions, lengths)):
            length = self.clamp_length(raw_len)
            # Upstream calls sample_for_eval_CFG with a single-caption list.
            latent = bundle.ar.sample_for_eval_CFG(
                [cap],
                length=length,
                tokenize_model=bundle.text_model,
                device=device,
                unit_length=MS_UNIT_LENGTH,
                cfg=scale,
            )  # (1, length//unit, latent_dim)
            motion = bundle.tae.forward_decoder(latent)  # (1, T, 272)
            motion = bundle.denormalize(motion[0])  # (T, 272)
            motion = motion[:length].cpu().numpy().astype(np.float32)
            outputs.append(motion)
            if progress:
                print(f"[ms] {i + 1}/{len(captions)} len={length} -> {motion.shape}", flush=True)

        return outputs

    def _encode_prefix_latents(
        self,
        motion_272: np.ndarray,
        condition_num_frames: int,
        latent_source: str = "sample",
    ) -> tuple[torch.Tensor, int]:
        """Encode the observed TP2M prefix with the same TAE used by T2M."""
        if int(condition_num_frames) < 1:
            raise ValueError("condition_num_frames must be >= 1")
        n = min(int(condition_num_frames), int(motion_272.shape[0]))
        if n <= 0:
            raise ValueError("empty TP2M prefix")
        prefix = np.asarray(motion_272[:n], dtype=np.float32)
        encoded_frames = n
        if encoded_frames < MS_UNIT_LENGTH:
            pad = np.repeat(prefix[-1:], MS_UNIT_LENGTH - encoded_frames, axis=0)
            prefix = np.concatenate([prefix, pad], axis=0)
            encoded_frames = MS_UNIT_LENGTH

        device = self.device
        x = torch.as_tensor(prefix, dtype=torch.float32, device=device).unsqueeze(0)
        x = (x - self.bundle.mean.to(x)) / self.bundle.std.to(x)
        latents, mu, _ = self.bundle.tae.encode(x)
        if latent_source == "mu":
            latents = mu
        elif latent_source != "sample":
            raise ValueError(f"unsupported latent_source={latent_source!r}")
        latents = _to_2d_latents(latents)
        return latents, encoded_frames

    @torch.no_grad()
    def infer_tp2m(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        gt_motions_272: Sequence[np.ndarray],
        condition_num_frames: int,
        guidance_param: Optional[float] = None,
        temperature: float = 1.0,
        prefix_latent_source: str = "sample",
        sampling_method: str = "new_demo",
        max_motion_length: int = 300,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate TP2M continuations from text plus GT MotionStreamer-272 prefix."""
        if not (len(captions) == len(lengths) == len(gt_motions_272)):
            raise ValueError("captions, lengths, and gt_motions_272 must have equal length")
        if self.bundle.text_model is None:
            raise RuntimeError(
                "MotionStreamerBundle was built with load_text_model=False; "
                "the SentenceT5 text encoder is required for generation."
            )
        if sampling_method not in {"new_demo", "new"}:
            raise ValueError(f"unsupported sampling_method={sampling_method!r}")

        scale = self.bundle.guidance_param if guidance_param is None else float(guidance_param)
        device = self.device
        outputs: List[np.ndarray] = []
        for i, (caption, raw_len, gt_motion) in enumerate(
            zip(captions, lengths, gt_motions_272)
        ):
            prefix_latents, _encoded_frames = self._encode_prefix_latents(
                gt_motion,
                condition_num_frames,
                prefix_latent_source,
            )
            prefix_tokens = int(prefix_latents.shape[0])
            eval_len = (min(int(raw_len), int(max_motion_length)) // MS_UNIT_LENGTH) * MS_UNIT_LENGTH
            sample_total_frames = max(eval_len, (prefix_tokens + 1) * MS_UNIT_LENGTH)
            if sampling_method == "new_demo":
                _xs, new_latents = self.bundle.ar.sample_for_eval_CFG_babel_inference_new_demo(
                    B_text=str(caption),
                    A_motion=prefix_latents,
                    length=sample_total_frames,
                    clip_model=self.bundle.text_model,
                    device=device,
                    tokenizer="t5-xxl",
                    unit_length=MS_UNIT_LENGTH,
                    cfg=scale,
                    temperature=float(temperature),
                )
            else:
                continuation_frames = max(eval_len - prefix_tokens * MS_UNIT_LENGTH, MS_UNIT_LENGTH)
                _xs, new_latents = self.bundle.ar.sample_for_eval_CFG_babel_inference_new(
                    B_text=[str(caption)],
                    A_motion=prefix_latents,
                    length=continuation_frames,
                    clip_model=self.bundle.text_model,
                    device=device,
                    tokenizer="t5-xxl",
                    unit_length=MS_UNIT_LENGTH,
                    cfg=scale,
                )
            full_latents = torch.cat([prefix_latents.unsqueeze(0), new_latents], dim=1)
            motion = self.bundle.tae.forward_decoder(full_latents).squeeze(0)
            motion = self.bundle.denormalize(motion)[:eval_len]
            outputs.append(motion.detach().cpu().numpy().astype(np.float32))
            if progress:
                print(
                    f"[ms-tp2m] {i + 1}/{len(captions)} "
                    f"cond={condition_num_frames} len={eval_len}",
                    flush=True,
                )
        return outputs

    @torch.no_grad()
    def infer_sequential_t2m(
        self,
        captions_per_sample: Sequence[Sequence[str]],
        lengths_per_sample: Sequence[Sequence[int]],
        guidance_param: Optional[float] = None,
        temperature: float = 1.0,
        context_tokens: int = 16,
        block_tokens: int = MS_MAX_TOKENS + 1,
        seed: Optional[int] = None,
        shard_index: int = 0,
        sample_offset: int = 0,
        progress: bool = False,
    ) -> List[np.ndarray]:
        """Generate one continuous MotionStreamer-272 motion per prompt sequence.

        The first segment is sampled from text as normal T2M. Each following
        segment is sampled with MotionStreamer's BABEL continuation sampler,
        conditioned on all previously generated latent tokens. The full latent
        stream is decoded once at the end. Continuation uses a bounded rolling
        latent context, matching MotionStreamer's block-size-limited streaming
        setup instead of feeding the full accumulated history back into the AR
        transformer.
        """
        if len(captions_per_sample) != len(lengths_per_sample):
            raise ValueError("captions_per_sample and lengths_per_sample must have equal length")
        bundle = self.bundle
        if bundle.text_model is None:
            raise RuntimeError(
                "MotionStreamerBundle was built with load_text_model=False; "
                "the SentenceT5 text encoder is required for generation."
            )
        device = self.device
        scale = bundle.guidance_param if guidance_param is None else float(guidance_param)

        outputs: List[np.ndarray] = []
        for i, (captions, raw_lengths) in enumerate(zip(captions_per_sample, lengths_per_sample)):
            if len(captions) != len(raw_lengths):
                raise ValueError(
                    f"sample {i} has {len(captions)} captions but {len(raw_lengths)} lengths"
                )
            if not captions:
                raise ValueError(f"sample {i} has no captions")
            seg_lengths = [self.clamp_length(n) for n in raw_lengths]

            if seed is not None:
                torch.manual_seed(int(seed) + int(shard_index) * 100000 + int(sample_offset) + i)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(
                        int(seed) + int(shard_index) * 100000 + int(sample_offset) + i
                    )

            first_latent = bundle.ar.sample_for_eval_CFG(
                [str(captions[0])],
                length=seg_lengths[0],
                tokenize_model=bundle.text_model,
                device=device,
                unit_length=MS_UNIT_LENGTH,
                cfg=scale,
            )
            acc = _to_2d_latents(first_latent)
            total_len = seg_lengths[0]

            for caption, seg_len in zip(captions[1:], seg_lengths[1:]):
                total_len += seg_len
                remaining = max(1, int(seg_len) // MS_UNIT_LENGTH)
                while remaining > 0:
                    prefix_len = min(int(context_tokens), int(acc.shape[0]))
                    prefix_len = max(1, min(prefix_len, int(block_tokens) - 1))
                    take = min(remaining, max(1, int(block_tokens) - prefix_len))
                    prefix = acc[-prefix_len:].contiguous()
                    _, new_latents = bundle.ar.sample_for_eval_CFG_babel_inference_new_demo(
                        B_text=str(caption),
                        A_motion=prefix,
                        length=(prefix_len + take) * MS_UNIT_LENGTH,
                        clip_model=bundle.text_model,
                        device=device,
                        tokenizer="t5-xxl",
                        unit_length=MS_UNIT_LENGTH,
                        cfg=scale,
                        temperature=float(temperature),
                    )
                    new_latents = _to_2d_latents(new_latents)[:take]
                    if int(new_latents.shape[0]) != take:
                        raise RuntimeError(
                            "MotionStreamer continuation returned "
                            f"{new_latents.shape[0]} tokens, expected {take}"
                        )
                    acc = torch.cat([acc, new_latents], dim=0)
                    remaining -= take

            motion = bundle.tae.forward_decoder(acc.unsqueeze(0))[0]
            motion = bundle.denormalize(motion)[:total_len].cpu().numpy().astype(np.float32)
            outputs.append(motion)
            if progress:
                print(
                    f"[ms-seq] {i + 1}/{len(captions_per_sample)} "
                    f"segments={len(captions)} len={total_len} -> {motion.shape}",
                    flush=True,
                )

        return outputs

    infer_multi_prompt_t2m = infer_sequential_t2m

    def __call__(self, captions, lengths, **kwargs):
        if kwargs.pop("tp2m", False):
            return self.infer_tp2m(captions, lengths, **kwargs)
        if kwargs.pop("sequential", False):
            return self.infer_sequential_t2m(captions, lengths, **kwargs)
        if len(captions) > 0 and not isinstance(captions[0], str):
            return self.infer_sequential_t2m(captions, lengths, **kwargs)
        return self.infer_t2m(captions, lengths, **kwargs)
