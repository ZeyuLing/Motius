"""Audio features used by the released Bailando checkpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np


BAILANDO_AUDIO_SAMPLE_RATE = 3_840
BAILANDO_AUDIO_FEATURE_DIM = 438
BAILANDO_AUDIO_FPS = 7.5
BAILANDO_BEAT_CHANNEL = 53


def _load_audio(audio: Union[str, Path, np.ndarray], sample_rate: int | None):
    import librosa

    if isinstance(audio, (str, Path)):
        waveform, _ = librosa.load(
            str(audio), sr=BAILANDO_AUDIO_SAMPLE_RATE, mono=True
        )
        return waveform.astype(np.float32)
    if sample_rate is None:
        raise ValueError("sample_rate is required when audio is a waveform")
    waveform = np.asarray(audio, dtype=np.float32).reshape(-1)
    if int(sample_rate) != BAILANDO_AUDIO_SAMPLE_RATE:
        waveform = librosa.resample(
            waveform,
            orig_sr=int(sample_rate),
            target_sr=BAILANDO_AUDIO_SAMPLE_RATE,
        )
    return np.asarray(waveform, dtype=np.float32)


def extract_bailando_audio_features(
    audio: Union[str, Path, np.ndarray],
    *,
    sample_rate: int | None = None,
) -> np.ndarray:
    """Extract the official 438-D, 7.5 fps Bailando music representation.

    The feature order is 20 MFCC, 20 MFCC-delta, 12 chroma, onset strength,
    beat one-hot, and a 384-bin tempogram. Librosa's default 512-sample hop at
    3,840 Hz gives the 7.5 fps sequence expected by the released GPT.
    """

    import librosa

    waveform = _load_audio(audio, sample_rate)
    if waveform.size < 2_048:
        waveform = np.pad(waveform, (0, 2_048 - waveform.size))

    mel = librosa.feature.melspectrogram(
        y=waveform, sr=BAILANDO_AUDIO_SAMPLE_RATE
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mfcc = librosa.feature.mfcc(S=mel_db, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc, width=3)
    harmonic, percussive = librosa.effects.hpss(waveform)
    chroma = librosa.feature.chroma_cqt(
        y=harmonic,
        sr=BAILANDO_AUDIO_SAMPLE_RATE,
        n_octaves=5,
    )
    onset = librosa.onset.onset_strength(
        y=percussive,
        aggregate=np.median,
        sr=BAILANDO_AUDIO_SAMPLE_RATE,
    )
    tempogram = librosa.feature.tempogram(
        onset_envelope=onset,
        sr=BAILANDO_AUDIO_SAMPLE_RATE,
    )
    _, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset,
        sr=BAILANDO_AUDIO_SAMPLE_RATE,
    )
    beats = np.zeros_like(onset)
    beats[np.asarray(beat_frames, dtype=np.int64)] = 1.0

    frame_count = min(
        mfcc.shape[1],
        mfcc_delta.shape[1],
        chroma.shape[1],
        onset.shape[0],
        beats.shape[0],
        tempogram.shape[1],
    )
    features = np.concatenate(
        [
            mfcc[:, :frame_count],
            mfcc_delta[:, :frame_count],
            chroma[:, :frame_count],
            onset[None, :frame_count],
            beats[None, :frame_count],
            tempogram[:, :frame_count],
        ],
        axis=0,
    ).T.astype(np.float32)
    if features.shape[1] != BAILANDO_AUDIO_FEATURE_DIM:
        raise RuntimeError(
            "Bailando audio extraction produced "
            f"{features.shape[1]} features, expected {BAILANDO_AUDIO_FEATURE_DIM}"
        )
    return features


def beat_frames_from_features(features: np.ndarray) -> np.ndarray:
    """Return official music-beat frame indices from Bailando features."""

    values = np.asarray(features)
    if values.ndim != 2 or values.shape[1] <= BAILANDO_BEAT_CHANNEL:
        raise ValueError(f"Expected (T,438) music features, got {values.shape}")
    return np.flatnonzero(values[:, BAILANDO_BEAT_CHANNEL] > 0.5)


__all__ = [
    "BAILANDO_AUDIO_FEATURE_DIM",
    "BAILANDO_AUDIO_FPS",
    "BAILANDO_AUDIO_SAMPLE_RATE",
    "BAILANDO_BEAT_CHANNEL",
    "beat_frames_from_features",
    "extract_bailando_audio_features",
]
