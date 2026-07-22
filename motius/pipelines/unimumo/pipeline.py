"""Unified music, motion, and text inference with UniMuMo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from motius.motion.representation.humanml import recover_from_ric
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@dataclass(frozen=True)
class UniMuMoGenerationOutput:
    """Outputs retained from a UniMuMo generation or translation call."""

    waveform: np.ndarray | None = None
    sample_rate: int | None = None
    motion: np.ndarray | None = None
    joints: np.ndarray | None = None
    motion_fps: float | None = None
    music_codes: np.ndarray | None = None
    motion_codes: np.ndarray | None = None
    captions: tuple[str, ...] | None = None


def _load_audio(
    audio: str | Path | np.ndarray | torch.Tensor,
    *,
    sample_rate: int | None,
    target_sample_rate: int,
) -> np.ndarray:
    import librosa

    if isinstance(audio, (str, Path)):
        waveform, _ = librosa.load(
            str(audio), sr=target_sample_rate, mono=True
        )
        return np.asarray(waveform, dtype=np.float32)
    if sample_rate is None:
        raise ValueError("sample_rate is required for array audio input")
    if torch.is_tensor(audio):
        waveform = audio.detach().cpu().numpy()
    else:
        waveform = np.asarray(audio)
    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if int(sample_rate) != target_sample_rate:
        waveform = librosa.resample(
            waveform,
            orig_sr=int(sample_rate),
            target_sr=target_sample_rate,
        )
    return np.asarray(waveform, dtype=np.float32)


@PIPELINES.register_module()
class UniMuMoPipeline(BasePipeline):
    """Run UniMuMo without importing its upstream repository at runtime."""

    BUNDLE_CLS = "motius.models.unimumo.UniMuMoBundle"

    def __init__(self, bundle, device: str | torch.device | None = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device: str | torch.device):
        self.bundle.to(torch.device(device))
        self.bundle.eval()
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    def _random_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device=self.device)
        generator.manual_seed(int(seed))
        return generator

    @staticmethod
    def _description(music: str = "", motion: str = "") -> str:
        return f"{music.strip()} <separation> {motion.strip()}"

    def _decode(
        self, music_codes: torch.Tensor, motion_codes: torch.Tensor
    ) -> UniMuMoGenerationOutput:
        waveform = self.bundle.decode_audio(music_codes)
        motion = self.bundle.decode_motion(music_codes, motion_codes)
        joints = recover_from_ric(motion, joints_num=22)
        return UniMuMoGenerationOutput(
            waveform=waveform[0, 0].float().cpu().numpy(),
            sample_rate=self.bundle.sample_rate,
            motion=motion[0].float().cpu().numpy(),
            joints=joints[0].float().cpu().numpy(),
            motion_fps=self.bundle.motion_fps,
            music_codes=music_codes[0].cpu().numpy(),
            motion_codes=motion_codes[0].cpu().numpy(),
        )

    def _generation_kwargs(
        self,
        *,
        guidance_scale: float,
        temperature: float,
        top_k: int,
        seed: int,
    ) -> dict:
        return {
            "guidance_scale": float(guidance_scale),
            "temperature": float(temperature),
            "top_k": int(top_k),
            "generator": self._random_generator(seed),
        }

    @torch.inference_mode()
    def infer_text_to_music_motion(
        self,
        *,
        music_prompt: str = "",
        motion_prompt: str = "",
        duration_seconds: float | None = None,
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 250,
        seed: int = 0,
    ) -> UniMuMoGenerationOutput:
        duration = (
            float(self.bundle.config["default_duration_seconds"])
            if duration_seconds is None
            else float(duration_seconds)
        )
        maximum = float(self.bundle.config["max_duration_seconds"])
        if not 0 < duration <= maximum:
            raise ValueError(f"duration_seconds must be in (0, {maximum}]")
        timesteps = max(1, int(duration * self.bundle.code_fps))
        music_codes, motion_codes = self.bundle.generate_codes(
            [self._description(music_prompt, motion_prompt)],
            timesteps=timesteps,
            mode="music_motion",
            **self._generation_kwargs(
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            ),
        )
        return self._decode(music_codes, motion_codes)

    @torch.inference_mode()
    def infer_text_to_motion(
        self,
        prompt: str,
        **kwargs,
    ) -> UniMuMoGenerationOutput:
        """Zero-shot text-to-motion through UniMuMo's joint generation path."""

        return self.infer_text_to_music_motion(motion_prompt=prompt, **kwargs)

    @torch.inference_mode()
    def infer_text_to_music(
        self,
        prompt: str,
        **kwargs,
    ) -> UniMuMoGenerationOutput:
        return self.infer_text_to_music_motion(music_prompt=prompt, **kwargs)

    def _audio_codes(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        *,
        sample_rate: int | None,
    ) -> torch.Tensor:
        waveform = _load_audio(
            audio,
            sample_rate=sample_rate,
            target_sample_rate=self.bundle.sample_rate,
        )
        maximum_samples = int(
            self.bundle.config["max_duration_seconds"] * self.bundle.sample_rate
        )
        return self.bundle.encode_audio(waveform[:maximum_samples])

    @torch.inference_mode()
    def infer_music_to_motion(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        *,
        sample_rate: int | None = None,
        motion_prompt: str = "",
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 250,
        seed: int = 0,
    ) -> UniMuMoGenerationOutput:
        music_codes = self._audio_codes(audio, sample_rate=sample_rate)
        generated_music, motion_codes = self.bundle.generate_codes(
            [self._description(motion=motion_prompt)],
            timesteps=music_codes.shape[-1],
            mode="music2motion",
            music_codes=music_codes,
            **self._generation_kwargs(
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            ),
        )
        return self._decode(generated_music, motion_codes)

    @torch.inference_mode()
    def infer_motion_to_music(
        self,
        motion: np.ndarray | torch.Tensor,
        *,
        music_prompt: str = "",
        guidance_scale: float = 4.0,
        temperature: float = 1.0,
        top_k: int = 250,
        seed: int = 0,
    ) -> UniMuMoGenerationOutput:
        motion_codes = self.bundle.encode_motion(motion)
        music_codes, generated_motion = self.bundle.generate_codes(
            [self._description(music=music_prompt)],
            timesteps=motion_codes.shape[-1],
            mode="motion2music",
            motion_codes=motion_codes,
            **self._generation_kwargs(
                guidance_scale=guidance_scale,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
            ),
        )
        return self._decode(music_codes, generated_motion)

    @torch.inference_mode()
    def infer_music_to_text(
        self,
        audio: str | Path | np.ndarray | torch.Tensor,
        *,
        sample_rate: int | None = None,
    ) -> UniMuMoGenerationOutput:
        music_codes = self._audio_codes(audio, sample_rate=sample_rate)
        captions = self.bundle.caption(music_codes, modality="music")
        return UniMuMoGenerationOutput(
            music_codes=music_codes[0].cpu().numpy(),
            captions=tuple(captions),
        )

    @torch.inference_mode()
    def infer_motion_to_text(
        self, motion: np.ndarray | torch.Tensor
    ) -> UniMuMoGenerationOutput:
        motion_codes = self.bundle.encode_motion(motion)
        captions = self.bundle.caption(motion_codes, modality="motion")
        return UniMuMoGenerationOutput(
            motion_codes=motion_codes[0].cpu().numpy(),
            captions=tuple(captions),
        )

    def __call__(self, *args, task: str = "text-to-music-motion", **kwargs):
        methods = {
            "text-to-music-motion": self.infer_text_to_music_motion,
            "t2mm": self.infer_text_to_music_motion,
            "text-to-motion": self.infer_text_to_motion,
            "t2m": self.infer_text_to_motion,
            "text-to-music": self.infer_text_to_music,
            "t2music": self.infer_text_to_music,
            "music-to-motion": self.infer_music_to_motion,
            "m2d": self.infer_music_to_motion,
            "motion-to-music": self.infer_motion_to_music,
            "music-to-text": self.infer_music_to_text,
            "motion-to-text": self.infer_motion_to_text,
            "m2t": self.infer_motion_to_text,
        }
        try:
            method = methods[task]
        except KeyError as exc:
            raise ValueError(f"Unsupported UniMuMo task: {task!r}") from exc
        return method(*args, **kwargs)


__all__ = ["UniMuMoGenerationOutput", "UniMuMoPipeline"]
