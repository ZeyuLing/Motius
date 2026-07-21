"""Jukebox layer-66 features used by the released EDGE checkpoint."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np


EDGE_AUDIO_FPS = 30.0
EDGE_AUDIO_FEATURE_DIM = 4_800
EDGE_AUDIO_WINDOW_FRAMES = 150
EDGE_AUDIO_WINDOW_SECONDS = 5.0
EDGE_AUDIO_HOP_SECONDS = 2.5
EDGE_JUKEBOX_LAYER = 66
EDGE_JUKEBOX_SAMPLE_RATE = 44_100
EDGE_JUKEBOX_PRIOR_SHA256 = (
    "89a1dd14f5b2f9b16b3e73b53fa2138cc89fd96bb13249b4267fea471de92672"
)
EDGE_JUKEBOX_VQVAE_SHA256 = (
    "69745413a48e887f8a3fe91b972a6f7f434021a1ce911a99187b331eb48c059a"
)


def validate_edge_music_features(features) -> np.ndarray:
    value = np.asarray(features, dtype=np.float32)
    if value.ndim == 2:
        value = value[None]
    expected = (EDGE_AUDIO_WINDOW_FRAMES, EDGE_AUDIO_FEATURE_DIM)
    if value.ndim != 3 or value.shape[1:] != expected:
        raise ValueError(f"EDGE Jukebox features must have shape (N,{expected[0]},{expected[1]}), got {value.shape}")
    if not np.isfinite(value).all():
        raise ValueError("EDGE Jukebox features contain NaN or infinite values")
    return np.ascontiguousarray(value)


def edge_audio_window_count(duration_seconds: float) -> int:
    """Return the overlapping window count needed to cover an audio clip."""

    duration = float(duration_seconds)
    if duration + 1e-6 < EDGE_AUDIO_WINDOW_SECONDS:
        raise ValueError("EDGE requires at least five seconds of input audio")
    remaining = max(0.0, duration - EDGE_AUDIO_WINDOW_SECONDS)
    return int(math.ceil(remaining / EDGE_AUDIO_HOP_SECONDS - 1e-9)) + 1


def extract_edge_jukebox_features(
    audio: str | Path,
    *,
    max_seconds: float | None = None,
    fp16: bool = False,
    cache_dir: str | Path | None = None,
) -> np.ndarray:
    """Extract official overlapping 5-second Jukebox feature windows."""

    import jukemirlib
    import librosa

    if cache_dir is not None:
        from jukemirlib import constants

        constants.CACHE_DIR = str(Path(cache_dir).expanduser().resolve())

    path = Path(audio).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    waveform, _ = librosa.load(
        str(path), sr=EDGE_JUKEBOX_SAMPLE_RATE, mono=True
    )
    duration = float(len(waveform)) / EDGE_JUKEBOX_SAMPLE_RATE
    if max_seconds is not None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        duration = min(duration, float(max_seconds))
        waveform = waveform[: int(round(duration * EDGE_JUKEBOX_SAMPLE_RATE))]
    windows = edge_audio_window_count(duration)
    window_samples = int(round(EDGE_AUDIO_WINDOW_SECONDS * EDGE_JUKEBOX_SAMPLE_RATE))
    hop_samples = int(round(EDGE_AUDIO_HOP_SECONDS * EDGE_JUKEBOX_SAMPLE_RATE))
    rows = []
    for index in range(windows):
        start = index * hop_samples
        audio_window = np.asarray(waveform[start : start + window_samples], dtype=np.float32)
        if len(audio_window) < window_samples:
            audio_window = np.pad(audio_window, (0, window_samples - len(audio_window)))
        peak = float(np.abs(audio_window).max(initial=0.0))
        if peak > 0:
            audio_window /= peak
        representation = jukemirlib.extract(
            audio=audio_window,
            layers=[EDGE_JUKEBOX_LAYER],
            downsample_target_rate=int(EDGE_AUDIO_FPS),
            fp16=bool(fp16),
            fp16_out=False,
        )[EDGE_JUKEBOX_LAYER]
        value = np.asarray(representation, dtype=np.float32)
        if value.shape[0] < EDGE_AUDIO_WINDOW_FRAMES:
            value = np.pad(
                value,
                ((0, EDGE_AUDIO_WINDOW_FRAMES - value.shape[0]), (0, 0)),
                mode="edge",
            )
        rows.append(value[:EDGE_AUDIO_WINDOW_FRAMES])
    return validate_edge_music_features(np.stack(rows))


__all__ = [
    "EDGE_AUDIO_FEATURE_DIM",
    "EDGE_AUDIO_FPS",
    "EDGE_AUDIO_HOP_SECONDS",
    "EDGE_AUDIO_WINDOW_FRAMES",
    "EDGE_AUDIO_WINDOW_SECONDS",
    "EDGE_JUKEBOX_LAYER",
    "EDGE_JUKEBOX_SAMPLE_RATE",
    "EDGE_JUKEBOX_PRIOR_SHA256",
    "EDGE_JUKEBOX_VQVAE_SHA256",
    "extract_edge_jukebox_features",
    "edge_audio_window_count",
    "validate_edge_music_features",
]
