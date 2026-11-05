"""Prepared fixed-window EDGE motion and Jukebox conditioning."""

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module()
class PreparedEDGEDataset(Dataset):
    """JSON/JSONL records with ``path`` to normalized motion/music NPZ files.

    Motion is already transformed by the selected checkpoint's range scaler;
    music is the original prepared 30-fps Jukebox conditioning. Source slicing,
    audio extraction and fitting the training scaler run before this boundary.
    """

    def __init__(self, manifest):
        self.manifest = Path(manifest)
        text = self.manifest.read_text()
        self.records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if self.manifest.suffix == ".jsonl"
            else json.loads(text)
        )

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = Path(self.records[index]["path"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        with np.load(path, allow_pickle=False) as data:
            motion = torch.from_numpy(data["motion"].copy()).float()
            music = torch.from_numpy(data["music_features"].copy()).float()
        if motion.shape != (150, 151) or music.shape != (150, 4800):
            raise ValueError(
                f"EDGE requires [150,151] motion and [150,4800] music in {path}"
            )
        return dict(motion=motion, music_features=music)
