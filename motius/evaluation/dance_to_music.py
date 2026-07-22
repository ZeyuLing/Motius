"""Dance-to-music metrics used by the released D2M-GAN protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


D2MGAN_SAMPLE_RATE = 22_050


@dataclass(frozen=True)
class D2MGANBeatScore:
    """Per-clip scores from the D2M-GAN `Beats_Scores` implementation."""

    beat_count_ratio: float
    beat_hit_rate: float
    reference_beat_bins: int
    generated_beat_bins: int
    hit_beat_bins: int


def d2mgan_beat_bins(
    waveform: np.ndarray,
    *,
    sample_rate: int = D2MGAN_SAMPLE_RATE,
) -> np.ndarray:
    """Detect one-second onset bins exactly as D2M-GAN's evaluator does."""

    try:
        import librosa
    except ImportError as exc:
        raise ImportError(
            "Dance-to-music beat evaluation requires `pip install motius[unimumo]`."
        ) from exc

    audio = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if int(sample_rate) != D2MGAN_SAMPLE_RATE:
        audio = librosa.resample(
            audio,
            orig_sr=int(sample_rate),
            target_sr=D2MGAN_SAMPLE_RATE,
        )
    bins = np.zeros(
        int(np.ceil(len(audio) / D2MGAN_SAMPLE_RATE)), dtype=np.int8
    )
    onsets = librosa.onset.onset_detect(
        y=audio,
        sr=D2MGAN_SAMPLE_RATE,
        wait=1,
        delta=0.2,
        pre_avg=1,
        post_avg=1,
        post_max=1,
        units="time",
    )
    for onset in onsets:
        index = int(np.trunc(onset))
        if 0 <= index < len(bins):
            bins[index] = 1
    return bins


def d2mgan_beat_score(
    reference_bins: np.ndarray,
    generated_bins: np.ndarray,
) -> D2MGANBeatScore:
    """Compute upstream beat-count ratio and hit rate for one clip.

    The paper calls the first quantity ``Beats Coverage``. It is generated beat
    bins divided by reference beat bins, so it is intentionally not bounded by
    one. A value above one means that the generated audio contains extra beats.
    """

    reference = np.asarray(reference_bins, dtype=bool).reshape(-1)
    generated = np.asarray(generated_bins, dtype=bool).reshape(-1)
    if reference.shape != generated.shape:
        raise ValueError("reference and generated beat bins must have equal shape")
    reference_count = int(reference.sum())
    if reference_count == 0:
        raise ValueError("reference audio contains no detected beat bins")
    generated_count = int(generated.sum())
    hit_count = int(np.logical_and(reference, generated).sum())
    return D2MGANBeatScore(
        beat_count_ratio=generated_count / reference_count,
        beat_hit_rate=hit_count / reference_count,
        reference_beat_bins=reference_count,
        generated_beat_bins=generated_count,
        hit_beat_bins=hit_count,
    )


def aggregate_d2mgan_beat_scores(
    scores: Iterable[D2MGANBeatScore],
) -> dict[str, float | int]:
    """Macro-average clips, matching the public D2M-GAN evaluation script."""

    rows = tuple(scores)
    if not rows:
        raise ValueError("at least one beat score is required")
    return {
        "n_samples": len(rows),
        "beat_count_ratio": float(
            np.mean([row.beat_count_ratio for row in rows])
        ),
        "beat_hit_rate": float(np.mean([row.beat_hit_rate for row in rows])),
    }


__all__ = [
    "D2MGANBeatScore",
    "D2MGAN_SAMPLE_RATE",
    "aggregate_d2mgan_beat_scores",
    "d2mgan_beat_bins",
    "d2mgan_beat_score",
]
