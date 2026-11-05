"""Padded text/music token batches for native joint TM2D training."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module()
class PreparedTM2DDataset(Dataset):
    def __init__(self, manifest):
        self.manifest = Path(manifest)
        self.records = json.loads(self.manifest.read_text())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = Path(self.records[index]["path"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        with np.load(path, allow_pickle=False) as data:
            values = {
                name: torch.from_numpy(data[name].copy())
                for name in [
                    "audio_features",
                    "audio_lengths",
                    "dance_tokens",
                    "motion_tokens",
                    "text_tokens",
                ]
            }
        values["audio_features"] = values["audio_features"].float()
        for name in ["audio_lengths", "dance_tokens", "motion_tokens", "text_tokens"]:
            values[name] = values[name].long()
        features = values["audio_features"]
        if (
            features.ndim != 2
            or features.shape[1] != 438
            or not 0 < int(values["audio_lengths"]) <= len(features)
        ):
            raise ValueError(
                f"TM2D requires padded (F,438) music and a valid length: {path}"
            )
        if any(
            values[name].ndim != 1
            for name in ["dance_tokens", "motion_tokens", "text_tokens"]
        ):
            raise ValueError(
                f"TM2D token records must be padded one-dimensional arrays: {path}"
            )
        return values
