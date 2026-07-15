"""Inference and scoring API for Motius TMR evaluator artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from safetensors.torch import load_file

from motius.evaluation.metrics import aggregate_t2m_metrics, diversity, r_precision
from motius.models.tmr import TMRBundle
from motius.registry import EVALUATORS


def _device(value: str | torch.device) -> torch.device:
    requested = torch.device(value)
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return requested


def _normalize_tmr_state_dict(
    state: dict[str, torch.Tensor], bundle: TMRBundle
) -> dict[str, torch.Tensor]:
    """Accept both bundle-prefixed and official TMR-core artifact layouts."""

    expected = set(bundle.state_dict())
    if set(state) == expected:
        return state
    prefixed = {f"tmr.{key}": value for key, value in state.items()}
    if set(prefixed) == expected:
        return prefixed
    return state


def _resolve_text_model_source(artifact_dir: Path, preprocessor: dict) -> str | Path:
    bundled = artifact_dir / preprocessor.get("text_encoder_dir", "text_encoder")
    if bundled.is_dir():
        return bundled
    return preprocessor.get("token_model", "distilbert-base-uncased")


def _pad(features: Sequence[np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    lengths = torch.tensor([len(item) for item in features], dtype=torch.long, device=device)
    width = int(features[0].shape[-1])
    maximum = int(lengths.max())
    values = torch.zeros((len(features), maximum, width), dtype=torch.float32, device=device)
    mask = torch.arange(maximum, device=device)[None] < lengths[:, None]
    for index, item in enumerate(features):
        tensor = torch.as_tensor(item, dtype=torch.float32, device=device)
        values[index, : len(tensor)] = tensor
    return {"x": values, "mask": mask, "length": lengths}


@EVALUATORS.register_module()
class TMRTextMotionEvaluator:
    """Load a Motius TMR artifact and score paired captions and motions."""

    def __init__(
        self,
        bundle: TMRBundle,
        mean: np.ndarray,
        std: np.ndarray,
        tokenizer,
        text_model,
        *,
        device: str | torch.device = "cuda",
        batch_size: int = 128,
        artifact_dir: Optional[str | Path] = None,
        preprocessor: Optional[dict] = None,
    ) -> None:
        self.device = _device(device)
        self.bundle = bundle.to(self.device).eval()
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        if self.mean.shape != self.std.shape:
            raise ValueError(f"Mean/std shapes differ: {self.mean.shape} and {self.std.shape}.")
        if np.any(self.std <= 0):
            raise ValueError("Every motion standard deviation must be positive.")
        self.tokenizer = tokenizer
        self.text_model = text_model.to(self.device).eval()
        for parameter in self.text_model.parameters():
            parameter.requires_grad = False
        self.batch_size = int(batch_size)
        self.artifact_dir = Path(artifact_dir) if artifact_dir else None
        self.preprocessor = dict(preprocessor or {})

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        device: str | torch.device = "cuda",
        batch_size: int = 128,
        local_files_only: bool = False,
        revision: Optional[str] = None,
    ) -> "TMRTextMotionEvaluator":
        source = Path(pretrained_model_name_or_path)
        if source.is_dir():
            artifact_dir = source
        else:
            from huggingface_hub import snapshot_download

            artifact_dir = Path(
                snapshot_download(
                    str(pretrained_model_name_or_path),
                    revision=revision,
                    local_files_only=local_files_only,
                )
            )
        config = json.loads((artifact_dir / "config.json").read_text())
        preprocessor = json.loads((artifact_dir / "preprocessor_config.json").read_text())
        bundle = TMRBundle(
            motion_nfeats=int(config["motion_nfeats"]),
            text_nfeats=int(config.get("text_nfeats", 768)),
            vae=bool(config.get("vae", True)),
            arch=config.get("arch"),
            sample_mean=True,
        )
        state = load_file(str(artifact_dir / config.get("weights_file", "model.safetensors")))
        state = _normalize_tmr_state_dict(state, bundle)
        missing, unexpected = bundle.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Evaluator artifact does not match TMRBundle; missing={missing}, "
                f"unexpected={unexpected}"
            )
        from transformers import AutoModel, AutoTokenizer

        text_model_source = _resolve_text_model_source(artifact_dir, preprocessor)
        tokenizer = AutoTokenizer.from_pretrained(
            text_model_source, local_files_only=local_files_only
        )
        text_model = AutoModel.from_pretrained(
            text_model_source, local_files_only=local_files_only
        )
        return cls(
            bundle,
            np.load(artifact_dir / preprocessor.get("mean", "stats/mean.npy")),
            np.load(artifact_dir / preprocessor.get("std", "stats/std.npy")),
            tokenizer,
            text_model,
            device=device,
            batch_size=batch_size,
            artifact_dir=artifact_dir,
            preprocessor=preprocessor,
        )

    def _motion_batches(self, motions: Sequence[np.ndarray]) -> Iterable[dict[str, torch.Tensor]]:
        for start in range(0, len(motions), self.batch_size):
            normalized = []
            for motion in motions[start : start + self.batch_size]:
                array = np.asarray(motion, dtype=np.float32)
                if array.ndim != 2 or array.shape[1] != len(self.mean):
                    raise ValueError(
                        f"Expected motion shape (T,{len(self.mean)}), got {array.shape}."
                    )
                normalized.append((array - self.mean) / self.std)
            yield _pad(normalized, self.device)

    @torch.inference_mode()
    def encode_motions(self, motions: Sequence[np.ndarray]) -> np.ndarray:
        if not motions:
            raise ValueError("At least one motion is required.")
        encoded = [
            self.bundle.encode_motion(batch, sample_mean=True).cpu().numpy()
            for batch in self._motion_batches(motions)
        ]
        return np.concatenate(encoded, axis=0)

    @torch.inference_mode()
    def encode_texts(self, captions: Sequence[str]) -> np.ndarray:
        if not captions:
            raise ValueError("At least one caption is required.")
        encoded_batches = []
        for start in range(0, len(captions), self.batch_size):
            batch_captions = [str(value) for value in captions[start : start + self.batch_size]]
            tokens = self.tokenizer(batch_captions, return_tensors="pt", padding=True)
            tokens = {key: value.to(self.device) for key, value in tokens.items()}
            output = self.text_model(**tokens).last_hidden_state
            mask = tokens["attention_mask"].bool()
            inputs = {"x": output, "mask": mask, "length": mask.sum(dim=1)}
            encoded_batches.append(
                self.bundle.encode_text(inputs, sample_mean=True).cpu().numpy()
            )
        return np.concatenate(encoded_batches, axis=0)

    def evaluate(
        self,
        captions: Sequence[str],
        predicted_motions: Sequence[np.ndarray],
        reference_motions: Optional[Sequence[np.ndarray]] = None,
        *,
        chunk_size: int = 32,
        n_repeats: int = 1,
        seed: int = 0,
        positive_group_ids: Optional[Sequence[object]] = None,
    ) -> dict[str, object]:
        if len(captions) != len(predicted_motions):
            raise ValueError("Captions and predicted motions must be paired one-to-one.")
        if positive_group_ids is not None and len(positive_group_ids) != len(captions):
            raise ValueError("positive_group_ids must match the caption count.")
        text_embeddings = self.encode_texts(captions)
        predicted_embeddings = self.encode_motions(predicted_motions)
        if reference_motions is not None:
            if len(reference_motions) != len(captions):
                raise ValueError("Reference motions must match the caption count.")
            return aggregate_t2m_metrics(
                text_embeddings,
                self.encode_motions(reference_motions),
                predicted_embeddings,
                n_repeats=n_repeats,
                chunk=chunk_size,
                seed=seed,
                positive_group_ids=positive_group_ids,
            )

        count = len(captions)
        if count < 3:
            raise ValueError(f"At least three paired samples are required, got {count}.")
        chunk = max(3, min(int(chunk_size), count))
        used = count // chunk * chunk
        rng = np.random.default_rng(seed)
        group_ids = (
            np.asarray(positive_group_ids, dtype=object)
            if positive_group_ids is not None
            else None
        )
        precision, matching, div = [], [], []
        for _ in range(int(n_repeats)):
            order = rng.permutation(count)
            counts = np.zeros(3, dtype=np.float64)
            distance = 0.0
            for start in range(0, used, chunk):
                indices = order[start : start + chunk]
                groups = group_ids[indices] if group_ids is not None else None
                value, score = r_precision(
                    text_embeddings[indices],
                    predicted_embeddings[indices],
                    top_k=3,
                    positive_group_ids=groups,
                )
                counts += value
                distance += score
            precision.append(counts / used)
            matching.append(distance / used)
            div.append(diversity(predicted_embeddings, rng=rng))
        precision_array = np.stack(precision)
        return {
            "n_samples_used": int(used),
            "n_repeats": int(n_repeats),
            "r_precision_policy": (
                "caption_group_multi_positive"
                if positive_group_ids is not None
                else "paired_diagonal_single_positive"
            ),
            "n_positive_groups": (
                len(set(positive_group_ids))
                if positive_group_ids is not None
                else int(count)
            ),
            "r_precision": precision_array.mean(0).tolist(),
            "r_precision_std": precision_array.std(0).tolist(),
            "matching_score": float(np.mean(matching)),
            "diversity_predicted": float(np.mean(div)),
        }


@EVALUATORS.register_module()
class TMRG1Evaluator(TMRTextMotionEvaluator):
    """TMR evaluator specialized by artifact metadata for G1-38D motion."""


__all__ = ["TMRTextMotionEvaluator", "TMRG1Evaluator"]
