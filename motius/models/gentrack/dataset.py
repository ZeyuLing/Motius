"""GenTrack prompt dataset: yields a text prompt plus its pre-extracted HYMotion
text embedding (no 8B encoder at train time) + the target motion length.

Registered into Motius' global dataset registry.

The embedding cache is stored as ``<feature_dir>/<sha256>.npy`` ([seq, 4096])
with a ``manifest.jsonl`` mapping prompt/id to key. Feature extraction remains
generator-specific; this dataset consumes the resulting immutable cache.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from motius.registry import DATASETS


@DATASETS.register_module()
class PhysFlowPromptDataset(Dataset):
    """Prompts + cached text embeddings for online-adversarial generation.

    Args:
        corpus_file: prompt corpus jsonl (id, prompt, duration_sec, split).
        feature_dir: directory holding ``<key>.npy`` + ``manifest.jsonl``.
        split: keep only rows with this split (None keeps all).
        fps: frames-per-second used to turn duration_sec into num_frames.
        min_frames / max_frames: clamp generated motion length.
        max_samples: optional cap (handy for smoke).
    """

    def __init__(
        self,
        corpus_file: str,
        feature_dir: str,
        split: Optional[str] = "train",
        fps: float = 30.0,
        min_frames: int = 30,
        max_frames: int = 300,
        max_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.feature_dir = feature_dir
        self.fps = float(fps)
        self.min_frames = int(min_frames)
        self.max_frames = int(max_frames)

        manifest = os.path.join(feature_dir, "manifest.jsonl")
        id_to_key: Dict[str, str] = {}
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for _id in rec.get("ids", []):
                    id_to_key[_id] = rec["key"]

        self.samples: List[Dict[str, Any]] = []
        with open(corpus_file, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if split is not None and row.get("split") != split:
                    continue
                key = id_to_key.get(row["id"])
                if key is None:
                    continue
                n_frames = int(round(float(row.get("duration_sec", 4.0)) * self.fps))
                n_frames = max(self.min_frames, min(self.max_frames, n_frames))
                self.samples.append(
                    {"id": row["id"], "prompt": row["prompt"], "key": key, "num_frames": n_frames}
                )
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

        if not self.samples:
            raise RuntimeError(
                f"PhysFlowPromptDataset empty: corpus={corpus_file} split={split} feature_dir={feature_dir}"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.samples[idx]
        arr = np.load(os.path.join(self.feature_dir, f"{rec['key']}.npy"))  # [seq, 4096]
        text_feat = torch.from_numpy(arr).float()
        return {
            "prompt": rec["prompt"],
            "prompt_id": rec["id"],
            "text_feat": text_feat,
            "num_frames": rec["num_frames"],
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        seq = batch[0]["text_feat"].shape[0]
        text_feat = torch.stack([b["text_feat"] for b in batch], dim=0)
        text_pad_mask = torch.ones(len(batch), seq, dtype=torch.bool)
        return {
            "prompt": [b["prompt"] for b in batch],
            "prompt_id": [b["prompt_id"] for b in batch],
            "text_feat": text_feat,
            "text_pad_mask": text_pad_mask,
            "num_frames": [b["num_frames"] for b in batch],
        }
