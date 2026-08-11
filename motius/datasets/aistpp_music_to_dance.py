"""Lazy AIST++ evaluation dataset for music-to-dance pipelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from motius.models.bailando.audio import (
    BAILANDO_AUDIO_FEATURE_DIM,
    BAILANDO_BEAT_CHANNEL,
)
from motius.registry import DATASETS


@dataclass(frozen=True)
class AISTPPMusicDanceRecord:
    """Paths and identifiers needed to load one official evaluation clip."""

    name: str
    music_id: str
    motion_path: Path
    music_feature_path: Path


def aistpp_music_id(sequence_name: str) -> str:
    """Return the AIST++ music id embedded in a sequence filename."""

    fields = Path(sequence_name).stem.split("_")
    if len(fields) < 2 or not fields[-2].startswith("m"):
        raise ValueError(f"Cannot parse AIST++ music id from {sequence_name!r}")
    return fields[-2]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return payload


@DATASETS.register_module()
class AISTPPMusicDanceDataset(Dataset):
    """Official Bailando AIST++ test/validation package.

    The original package stores two music streams: full-rate 60 fps features
    beside each GT dance and 7.5 fps features consumed by Bailando. Keeping
    both explicit prevents the model input stream from silently replacing the
    full-rate beat signal used by the official evaluator.
    """

    def __init__(
        self,
        motion_root: str | Path,
        music_feature_root: str | Path,
        *,
        max_samples: int | None = None,
        validate_paths: bool = True,
    ):
        self.motion_root = Path(motion_root).expanduser()
        self.music_feature_root = Path(music_feature_root).expanduser()
        motion_paths = sorted(self.motion_root.glob("*.json"))
        if max_samples is not None:
            motion_paths = motion_paths[: int(max_samples)]
        if not motion_paths:
            raise FileNotFoundError(f"No AIST++ JSON files found under {self.motion_root}")

        records = []
        for motion_path in motion_paths:
            music_id = aistpp_music_id(motion_path.name)
            feature_path = self.music_feature_root / f"{music_id}.json"
            if validate_paths and not feature_path.is_file():
                raise FileNotFoundError(
                    f"Missing 7.5 fps music feature for {motion_path.name}: {feature_path}"
                )
            records.append(
                AISTPPMusicDanceRecord(
                    name=motion_path.stem,
                    music_id=music_id,
                    motion_path=motion_path,
                    music_feature_path=feature_path,
                )
            )
        self.records = tuple(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        sample = _load_json(record.motion_path)
        feature_sample = _load_json(record.music_feature_path)

        gt_joints = np.asarray(sample["dance_array"], dtype=np.float32)
        if gt_joints.ndim != 2 or gt_joints.shape[1] != 72:
            raise ValueError(
                f"{record.motion_path} dance_array must be (T,72), got {gt_joints.shape}"
            )
        gt_joints = gt_joints.reshape(-1, 24, 3)

        full_music = np.asarray(sample["music_array"], dtype=np.float32)
        model_music = np.asarray(feature_sample["music_array"], dtype=np.float32)
        expected = (BAILANDO_AUDIO_FEATURE_DIM,)
        if full_music.ndim != 2 or full_music.shape[1:] != expected:
            raise ValueError(
                f"{record.motion_path} music_array must be (T,438), got {full_music.shape}"
            )
        if model_music.ndim != 2 or model_music.shape[1:] != expected:
            raise ValueError(
                f"{record.music_feature_path} music_array must be (T,438), got {model_music.shape}"
            )

        return {
            "name": record.name,
            "music_id": record.music_id,
            "gt_joints": gt_joints,
            "music_features": model_music,
            "music_beats": full_music[:, BAILANDO_BEAT_CHANNEL].astype(bool),
            "music_fps": 60.0,
            "motion_fps": 60.0,
        }


__all__ = [
    "AISTPPMusicDanceDataset",
    "AISTPPMusicDanceRecord",
    "aistpp_music_id",
]
