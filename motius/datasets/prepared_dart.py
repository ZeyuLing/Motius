"""Prepared normalized DART primitives at the native training boundary."""

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module()
class PreparedDARTDataset(Dataset):
    def __init__(self, manifest, history_length=2, future_length=8):
        self.manifest = Path(manifest)
        text = self.manifest.read_text()
        self.records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if self.manifest.suffix == ".jsonl"
            else json.loads(text)
        )
        self.length = int(history_length) + int(future_length)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        path = Path(self.records[index]["path"])
        if not path.is_absolute():
            path = self.manifest.parent / path
        with np.load(path, allow_pickle=False) as data:
            motion = torch.from_numpy(data["motion_normalized"].copy()).float()
            text = torch.from_numpy(data["text_embedding"].copy()).float()
            gender = str(data["gender"].item())
            betas = torch.from_numpy(data["betas"].copy()).float()
        if motion.shape != (self.length, 276) or text.shape != (512,):
            raise ValueError(
                f"DART requires ({self.length},276) normalized motion and (512,) text: {path}"
            )
        if betas.shape == (10,):
            betas = betas.expand(self.length, -1).clone()
        if betas.shape != (self.length, 10) or gender not in {"female", "male"}:
            raise ValueError(
                f"DART requires male/female and ten betas per frame: {path}"
            )
        return dict(
            motion=motion,
            text_embedding=text,
            condition=dict(gender=gender, betas=betas),
        )
