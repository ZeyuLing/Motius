"""Jukebox layer-66 features used by the released EDGE checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np


EDGE_AUDIO_FPS = 30.0
EDGE_AUDIO_FEATURE_DIM = 4_800
EDGE_AUDIO_WINDOW_FRAMES = 150
EDGE_AUDIO_WINDOW_SECONDS = 5.0
EDGE_AUDIO_HOP_SECONDS = 2.5
EDGE_JUKEBOX_LAYER = 66
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
    duration = float(librosa.get_duration(filename=str(path)))
    if max_seconds is not None:
        if max_seconds <= 0:
            raise ValueError("max_seconds must be positive")
        duration = min(duration, float(max_seconds))
    if duration + 1e-6 < EDGE_AUDIO_WINDOW_SECONDS:
        raise ValueError("EDGE requires at least five seconds of input audio")
    windows = int(np.floor((duration - EDGE_AUDIO_WINDOW_SECONDS) / EDGE_AUDIO_HOP_SECONDS)) + 1
    rows = []
    for index in range(windows):
        offset = index * EDGE_AUDIO_HOP_SECONDS
        representation = jukemirlib.extract(
            fpath=str(path),
            offset=offset,
            duration=EDGE_AUDIO_WINDOW_SECONDS,
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
    "EDGE_JUKEBOX_PRIOR_SHA256",
    "EDGE_JUKEBOX_VQVAE_SHA256",
    "extract_edge_jukebox_features",
    "validate_edge_music_features",
]
