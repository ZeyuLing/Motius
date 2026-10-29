"""Neutral prepared-motion records for native diffusion and flow Trainers."""

from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module()
class PreparedMotionDataset(Dataset):
    """Read explicitly normalized, padded NPZ tensors and external captions.

    JSON/JSONL records contain ``path`` and ``text``. Each NPZ contains motion
    [T,D] or [T,J,3] and scalar ``length``. CondMDI additionally requires
    obs_mask [T,D] (and optionally obs_x0); OmniControl requires hint [T,66].
    Original corpus preprocessing and random condition selection run before
    this boundary; this reader does not substitute a new training distribution.
    """

    def __init__(self, manifest, method, data_root=None):
        self.manifest = Path(manifest).expanduser()
        self.root = Path(data_root).expanduser() if data_root else self.manifest.parent
        text = self.manifest.read_text()
        self.records = (
            [json.loads(line) for line in text.splitlines() if line.strip()]
            if self.manifest.suffix == ".jsonl"
            else json.loads(text)
        )
        if method not in {
            "mdm",
            "condmdi",
            "omnicontrol",
            "projflow",
            "motionclr",
            "mld",
            "motionlcm",
        }:
            raise ValueError("Unknown prepared-motion method: " + method)
        self.method = method

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        path = Path(record["path"])
        with np.load(
            path if path.is_absolute() else self.root / path, allow_pickle=False
        ) as source:
            motion = torch.from_numpy(source["motion"].copy()).float()
            length = int(source["length"])
            expected = (22, 3) if self.method == "projflow" else (263,)
            if tuple(motion.shape[1:]) != expected or not 0 < length <= len(motion):
                raise ValueError(f"Invalid {self.method} motion or length in {path}")
            result = dict(
                motion=motion,
                motion_lens=torch.tensor(length),
                text=str(record["text"]),
            )
            if self.method == "condmdi":
                for name in ["obs_mask", "obs_x0"]:
                    if name not in source:
                        if name == "obs_mask":
                            raise ValueError(f"CondMDI requires obs_mask in {path}")
                        continue
                    value = torch.from_numpy(source[name].copy())
                    if value.shape != motion.shape:
                        raise ValueError(f"{name} must match motion in {path}")
                    result[name] = (
                        (value.bool() if name == "obs_mask" else value.float())
                        .T.unsqueeze(1)
                        .contiguous()
                    )
                if result["obs_mask"][..., length:].any():
                    raise ValueError(f"Observed padding frames in {path}")
            elif self.method == "omnicontrol":
                hint = torch.from_numpy(source["hint"].copy()).float()
                if hint.shape != (len(motion), 66):
                    raise ValueError(f"Invalid hint shape in {path}")
                result["hint"] = hint
        return result
