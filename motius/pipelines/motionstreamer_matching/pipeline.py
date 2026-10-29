"""Preprocessing and batching for the released MotionStreamer matching encoders."""

import os
from pathlib import Path
from typing import Sequence
import numpy as np
import torch
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionStreamerMatchingPipeline(BasePipeline):
    BUNDLE_CLS = (
        "motius.models.motionstreamer.evaluator_bundle.MotionStreamer272Evaluator"
    )

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            bundle.artifact_dir
            / bundle.preprocessor_config.get("tokenizer", "tokenizer"),
            local_files_only=True,
        )
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str], *, batch_size: int = 32) -> np.ndarray:
        rows = []
        for start in range(0, len(texts), batch_size):
            rows.append(
                self.bundle.encode_text_tokens(
                    self.tokenizer(
                        list(texts[start : start + batch_size]),
                        return_tensors="pt",
                        padding=True,
                    ).to(self.bundle.device)
                ).cpu()
            )
        return torch.cat(rows).numpy() if rows else np.empty((0, 256), np.float32)

    def save_pretrained(self, save_directory, **kwargs):
        result = self.bundle.save_pretrained(save_directory, **kwargs)
        self.tokenizer.save_pretrained(Path(save_directory) / "tokenizer")
        return result

    @torch.no_grad()
    def encode_motions(
        self,
        motions: Sequence[np.ndarray],
        *,
        lengths: Sequence[int] | None = None,
        batch_size: int = 32,
    ) -> np.ndarray:
        if lengths is None:
            lengths = [len(motion) for motion in motions]
        rows = np.zeros(
            (len(motions), self.bundle.max_motion_length, len(self.bundle.mean)),
            dtype=np.float32,
        )
        clipped_lengths = []
        mean = self.bundle.mean.cpu().numpy()
        std = self.bundle.std.cpu().numpy()
        for index, (motion, requested_length) in enumerate(zip(motions, lengths)):
            value = np.asarray(motion)
            if value.ndim != 2 or value.shape[1] != len(mean):
                raise ValueError(f"Expected MotionStreamer-272, got {value.shape}.")
            length = min(
                len(value), int(requested_length), self.bundle.max_motion_length
            )
            length = length // self.bundle.unit_length * self.bundle.unit_length
            if length < self.bundle.unit_length:
                raise ValueError(
                    "Motion is too short for the MotionStreamer evaluator."
                )
            rows[index, :length] = (value[:length] - mean) / std
            clipped_lengths.append(length)
        embeddings = []
        for start in range(0, len(rows), batch_size):
            maximum = max(clipped_lengths[start : start + batch_size])
            batch = torch.from_numpy(rows[start : start + batch_size, :maximum]).to(
                self.bundle.device
            )
            batch_lengths = torch.tensor(
                clipped_lengths[start : start + batch_size],
                device=self.bundle.device,
                dtype=torch.long,
            )
            embeddings.append(
                self.bundle.encode_motion_features(batch, batch_lengths).cpu()
            )
        return (
            torch.cat(embeddings).numpy()
            if embeddings
            else np.empty((0, 256), np.float32)
        )
