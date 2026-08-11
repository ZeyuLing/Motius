"""Generic manifest-backed text-motion training data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from motius.registry import DATASETS


def _load_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("samples", payload.get("data"))
    if not isinstance(payload, list):
        raise ValueError("Manifest must be a JSON list or contain a 'samples' list")
    return payload


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.as_tensor(value)


@DATASETS.register_module()
class ManifestTextMotionDataset(Dataset):
    """Read motions and optional text features from a neutral public schema.

    Each manifest record must contain ``motion`` and ``caption``. ``motion`` is
    a path relative to ``data_root`` and points to a ``.npy``, ``.npz``, or
    tensor file. An optional ``text_features`` path may contain either PRISM
    cached features (``t5_text_embeds``, ``t5_text_mask``) or HYMotion T2M
    cached features (``text_vec_raw``, ``text_ctxt_raw``,
    ``text_ctxt_raw_length``).
    """

    def __init__(
        self,
        data_root: str,
        manifest: str,
        motion_dim: Optional[int] = None,
        max_frames: Optional[int] = 360,
        max_text_length: int = 128,
        training: bool = True,
        pad_mode: str = "replicate",
        require_text_features: bool = False,
        motion_array_key: str = "motion",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root).expanduser()
        manifest_path = Path(manifest).expanduser()
        if not manifest_path.is_absolute():
            manifest_path = self.data_root / manifest_path
        self.samples = _load_json_or_jsonl(manifest_path)
        self.motion_dim = None if motion_dim is None else int(motion_dim)
        self.max_frames = None if max_frames is None else int(max_frames)
        self.max_text_length = int(max_text_length)
        self.training = bool(training)
        self.pad_mode = str(pad_mode)
        self.require_text_features = bool(require_text_features)
        self.motion_array_key = str(motion_array_key)
        if self.pad_mode not in {"replicate", "zero"}:
            raise ValueError("pad_mode must be 'replicate' or 'zero'")

    def __len__(self) -> int:
        return len(self.samples)

    def _resolve(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else self.data_root / path

    def _load_mapping(self, path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                return {key: payload[key] for key in payload.files}
        if suffix in {".pt", ".pth"}:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                payload = torch.load(path, map_location="cpu")
            if not isinstance(payload, dict):
                raise TypeError(f"Expected a feature mapping in {path}")
            return payload
        raise ValueError(f"Unsupported feature file: {path}")

    def _load_motion(self, record: Dict[str, Any]) -> torch.Tensor:
        path = self._resolve(str(record["motion"]))
        suffix = path.suffix.lower()
        if suffix == ".npy":
            value = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as payload:
                key = str(record.get("motion_array_key", self.motion_array_key))
                value = payload[key]
        elif suffix in {".pt", ".pth"}:
            try:
                value = torch.load(path, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(path, map_location="cpu")
            if isinstance(value, dict):
                key = str(record.get("motion_array_key", self.motion_array_key))
                value = value[key]
        else:
            raise ValueError(f"Unsupported motion file: {path}")
        motion = _as_tensor(value).to(torch.float32)
        if motion.ndim != 2:
            raise ValueError(f"Motion must have shape [frames, features], got {motion.shape}")
        if self.motion_dim is not None and motion.shape[-1] != self.motion_dim:
            raise ValueError(
                f"Expected motion_dim={self.motion_dim}, got {motion.shape[-1]} in {path}"
            )
        start = max(0, int(record.get("start_frame", 0)))
        end = min(len(motion), int(record.get("end_frame", len(motion))))
        return motion[start:end]

    def _crop_and_pad(self, motion: torch.Tensor) -> tuple[torch.Tensor, int]:
        if len(motion) == 0:
            raise ValueError("Empty motions are not valid training samples")
        if self.max_frames is None:
            return motion, len(motion)
        if len(motion) > self.max_frames:
            max_start = len(motion) - self.max_frames
            start = int(torch.randint(max_start + 1, ()).item()) if self.training else 0
            motion = motion[start : start + self.max_frames]
        valid_frames = len(motion)
        if valid_frames < self.max_frames:
            pad_frames = self.max_frames - valid_frames
            if self.pad_mode == "replicate":
                padding = motion[-1:].expand(pad_frames, -1).clone()
            else:
                padding = motion.new_zeros(pad_frames, motion.shape[-1])
            motion = torch.cat((motion, padding), dim=0)
        return motion, valid_frames

    def _caption(self, record: Dict[str, Any]) -> str:
        caption = record.get("caption", "")
        if isinstance(caption, list):
            if not caption:
                return ""
            index = int(torch.randint(len(caption), ()).item()) if self.training else 0
            caption = caption[index]
        return str(caption)

    def _text_features(self, record: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        feature_path = record.get("text_features")
        if feature_path is None:
            if self.require_text_features:
                raise KeyError("Sample is missing required 'text_features'")
            return {}
        payload = self._load_mapping(self._resolve(str(feature_path)))
        output: Dict[str, torch.Tensor] = {}
        for key in (
            "t5_text_embeds",
            "t5_text_mask",
            "text_vec_raw",
            "text_ctxt_raw",
            "text_ctxt_raw_length",
        ):
            if key in payload:
                output[key] = _as_tensor(payload[key])

        if "t5_text_embeds" in output:
            embeds = output["t5_text_embeds"][: self.max_text_length]
            mask = output.get("t5_text_mask")
            if mask is None:
                mask = torch.ones(len(embeds), dtype=torch.long)
            else:
                mask = mask[: self.max_text_length].to(torch.long)
            if len(embeds) < self.max_text_length:
                amount = self.max_text_length - len(embeds)
                embeds = torch.cat(
                    (embeds, embeds.new_zeros(amount, embeds.shape[-1])), dim=0
                )
                mask = torch.cat((mask, mask.new_zeros(amount)), dim=0)
            output["t5_text_embeds"] = embeds
            output["t5_text_mask"] = mask

        if "text_ctxt_raw" in output:
            context = output["text_ctxt_raw"][: self.max_text_length]
            length = min(len(context), self.max_text_length)
            if len(context) < self.max_text_length:
                context = torch.cat(
                    (
                        context,
                        context.new_zeros(
                            self.max_text_length - len(context), context.shape[-1]
                        ),
                    ),
                    dim=0,
                )
            output["text_ctxt_raw"] = context
            output["text_ctxt_raw_length"] = torch.tensor(length, dtype=torch.long)
        return output

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.samples[index]
        motion, valid_frames = self._crop_and_pad(self._load_motion(record))
        output: Dict[str, Any] = {
            "motion": motion,
            "caption": self._caption(record),
            "num_frames": torch.tensor(valid_frames, dtype=torch.long),
            "tgt_length": torch.tensor(valid_frames, dtype=torch.long),
        }
        output.update(self._text_features(record))
        return output


__all__ = ["ManifestTextMotionDataset"]
