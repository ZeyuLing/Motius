"""Sync audio/music waveform to motion after RandomCropPadding."""

from typing import Dict, List, Optional, Union

import numpy as np
import torch
from mmcv.transforms import BaseTransform

from motius.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class CropAudioToMotion(BaseTransform):
    """Crop (or pad) raw audio waveforms to match motion temporal extent.

    Must be placed **after** ``RandomCropPadding`` in the pipeline.
    Uses ``start_frame`` and ``num_frames`` written by RandomCropPadding
    together with the original motion length (``_original_num_frames``)
    and ``fps`` to compute the corresponding audio time window.

    If the sample has no audio, this transform is a no-op.

    Args:
        audio_keys: Audio keys to crop (e.g. ``["audio", "music"]``).
        fps_key: Key for motion FPS.
        sr_key: Key for audio sample rate.
        target_sr: Fallback sample rate when ``sr_key`` is not in results.
    """

    def __init__(
        self,
        audio_keys: Union[str, List[str]] = ("audio", "music"),
        fps_key: str = "fps",
        sr_key: str = "sr",
        target_sr: int = 16000,
    ):
        if isinstance(audio_keys, str):
            audio_keys = [audio_keys]
        self.audio_keys = list(audio_keys)
        self.fps_key = fps_key
        self.sr_key = sr_key
        self.target_sr = target_sr

    def transform(self, results: Dict) -> Dict:
        fps = results.get(self.fps_key)
        if fps is None or fps <= 0:
            return results

        # start_frame set by RandomCropPadding (None if no crop happened)
        # num_frames = real motion frames (before padding)
        start_frame = results.get("start_frame")
        num_frames = results.get("num_frames")
        if num_frames is None:
            return results

        # If no crop happened, audio starts at 0
        if start_frame is None:
            start_frame = 0

        sr = results.get(self.sr_key) or self.target_sr

        # Time window in seconds
        start_sec = start_frame / fps
        duration_sec = num_frames / fps

        audio_start = int(round(start_sec * sr))
        audio_len = int(round(duration_sec * sr))

        for key in self.audio_keys:
            audio = results.get(key)
            if audio is None:
                continue

            # Handle both numpy and torch
            is_tensor = isinstance(audio, torch.Tensor)
            total_samples = audio.shape[-1] if audio.ndim > 1 else len(audio)

            # Crop to window
            end = min(audio_start + audio_len, total_samples)
            start = min(audio_start, total_samples)
            if is_tensor:
                cropped = audio[..., start:end] if audio.ndim > 1 else audio[start:end]
            else:
                cropped = audio[start:end]

            # Pad if needed
            actual_len = cropped.shape[-1] if cropped.ndim > 1 else len(cropped)
            if actual_len < audio_len:
                pad_amount = audio_len - actual_len
                if is_tensor:
                    cropped = torch.nn.functional.pad(cropped, (0, pad_amount))
                else:
                    cropped = np.pad(cropped, (0, pad_amount))

            results[key] = cropped

        return results
