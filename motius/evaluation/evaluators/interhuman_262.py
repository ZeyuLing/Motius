"""InterHuman / InterGen native-262 evaluator.

This is the evaluator used by InterGen and by InterMask on InterHuman:
InterCLIP embeds paired two-person motions (`m1`, `m2`) and captions, then
reports R-Precision, multimodal distance (MM-D), diversity, and FID.

Inputs are native InterHuman-262 packs:

```
np.savez(path, m1=(N,T,262), m2=(N,T,262), lens=(N,), texts=(N,))
```

The code is independent of the official InterGen repository and loads a
self-contained Motius evaluator artifact from the Hugging Face Hub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from motius.models.interclip import load_interclip_checkpoint
from motius.registry import EVALUATORS

from motius.evaluation.metrics.t2m import euclidean_distance_matrix


_EMB_SCALE = 6.0


def _calculate_top_k(argmax: np.ndarray, top_k: int = 3) -> np.ndarray:
    size = argmax.shape[0]
    gt = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    correct = np.zeros(size, dtype=bool)
    out = []
    for i in range(top_k):
        correct = correct | (argmax[:, i] == gt[:, i])
        out.append(correct[:, None])
    return np.concatenate(out, axis=1)


def _activation_stats(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    activations = activations * _EMB_SCALE
    return np.mean(activations, axis=0), np.cov(activations, rowvar=False)


def _sqrtm_psd(matrix: np.ndarray) -> np.ndarray:
    matrix = (matrix + matrix.T) * 0.5
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def _stable_frechet(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    eps: float = 1e-6,
) -> float:
    from scipy import linalg

    diff = mu1 - mu2
    try:
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                raise ValueError(f"Imaginary component {np.max(np.abs(covmean.imag))}")
            covmean = covmean.real
        return float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2.0 * np.trace(covmean))
    except ValueError:
        eye = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        s1 = (sigma1 + eye).astype(np.float64)
        s2 = (sigma2 + eye).astype(np.float64)
        sqrt_s1 = _sqrtm_psd(s1)
        covmean = _sqrtm_psd(sqrt_s1 @ s2 @ sqrt_s1)
        return float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))


def _diversity(emb: np.ndarray, diversity_times: int) -> float:
    assert emb.ndim == 2
    if emb.shape[0] <= 1:
        return 0.0
    times = min(diversity_times, emb.shape[0] - 1)
    emb = emb * _EMB_SCALE
    first = np.random.choice(emb.shape[0], times, replace=False)
    second = np.random.choice(emb.shape[0], times, replace=False)
    return float(np.linalg.norm((emb[first] - emb[second]) / 2.0, axis=1).mean())


def load_native262_pack(path: str | Path, max_len: int = 300) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    m1 = data["m1"].astype(np.float32)
    m2 = data["m2"].astype(np.float32)
    lens = np.minimum(data["lens"].astype(np.int64), min(max_len, m1.shape[1], m2.shape[1]))
    texts = np.asarray(data["texts"])
    return m1, m2, lens, texts


@EVALUATORS.register_module()
class InterHuman262Evaluator:
    """InterGen/InterMask InterCLIP evaluator for native 2P-262 packs."""

    def __init__(
        self,
        ckpt_path: str,
        device: str = "cuda",
        batch_size: int = 96,
        retrieval_batch_size: int = 96,
        retrieval_repeats: int = 20,
        diversity_times: int = 300,
        max_len: int = 300,
    ):
        self.ckpt_path = Path(ckpt_path)
        self.device = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        self.batch_size = int(batch_size)
        self.retrieval_batch_size = int(retrieval_batch_size)
        self.retrieval_repeats = int(retrieval_repeats)
        self.diversity_times = int(diversity_times)
        self.max_len = int(max_len)
        self._model = None

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        **kwargs,
    ) -> "InterHuman262Evaluator":
        source = Path(pretrained_model_name_or_path)
        if not source.is_dir():
            from huggingface_hub import snapshot_download

            source = Path(snapshot_download(repo_id=str(pretrained_model_name_or_path)))
        checkpoint = source / "model.safetensors"
        if not checkpoint.exists():
            checkpoint = source / "interclip.ckpt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"InterCLIP checkpoint missing from {source}")
        return cls(ckpt_path=str(checkpoint), **kwargs)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.ckpt_path.exists():
            raise FileNotFoundError(
                "InterHuman262 evaluator checkpoint missing: "
                f"{self.ckpt_path}. Expected official InterGen interclip.ckpt."
            )
        self._model = load_interclip_checkpoint(str(self.ckpt_path), device=self.device, input_dim=258)

    @torch.no_grad()
    def embed_pack(self, pack: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        self._ensure_loaded()
        assert self._model is not None
        m1, m2, lens, texts = pack
        text_chunks = []
        motion_chunks = []
        for start in range(0, len(lens), self.batch_size):
            end = min(start + self.batch_size, len(lens))
            idx = np.arange(start, end)
            cur_len = int(lens[idx].max())
            motion_lens = torch.from_numpy(lens[idx]).long().to(self.device)
            batch = {
                "text": [str(texts[j]) for j in idx],
                "motions": torch.cat(
                    [
                        torch.from_numpy(m1[idx, :cur_len]).float(),
                        torch.from_numpy(m2[idx, :cur_len]).float(),
                    ],
                    dim=-1,
                ).to(self.device),
                "motion_lens": motion_lens,
            }
            align_idx = np.argsort(motion_lens.detach().cpu().tolist())[::-1].copy()
            batch["motions"] = batch["motions"][align_idx]
            batch["motion_lens"] = batch["motion_lens"][align_idx]
            text_emb = self._model.encode_text(dict(batch))["text_emb"][align_idx]
            motion_emb = self._model.encode_motion(dict(batch))["motion_emb"]
            text_chunks.append(text_emb.detach().cpu().numpy())
            motion_chunks.append(motion_emb.detach().cpu().numpy())
        return np.concatenate(text_chunks, axis=0), np.concatenate(motion_chunks, axis=0)

    def _retrieval(self, text_emb: np.ndarray, motion_emb: np.ndarray, seed: int) -> Tuple[np.ndarray, float]:
        if self.retrieval_batch_size < 3:
            raise ValueError(
                "InterHuman262 retrieval_batch_size must be >= 3 because the "
                "official protocol reports R@1/R@2/R@3."
            )
        rng = np.random.RandomState(seed)
        n = text_emb.shape[0]
        usable = (n // self.retrieval_batch_size) * self.retrieval_batch_size
        if usable <= 0:
            raise ValueError(f"Not enough samples ({n}) for retrieval batch size {self.retrieval_batch_size}")
        top_sum = np.zeros(3, dtype=np.float64)
        mm_sum = 0.0
        count = 0
        for _ in range(self.retrieval_repeats):
            order = rng.permutation(n)[:usable]
            for start in range(0, usable, self.retrieval_batch_size):
                idx = order[start : start + self.retrieval_batch_size]
                dist = euclidean_distance_matrix(text_emb[idx], motion_emb[idx])
                mm_sum += float(np.trace(dist))
                top_sum += _calculate_top_k(np.argsort(dist, axis=1), top_k=3).sum(axis=0)
                count += len(idx)
        return top_sum / count, mm_sum / count

    def evaluate_packs(
        self,
        gt_pack: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
        pred_packs: Mapping[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
        seed: int = 42,
    ) -> Dict[str, Dict[str, object]]:
        packs = {"Real": gt_pack, **dict(pred_packs)}
        results: Dict[str, Dict[str, object]] = {}
        motion_embeddings: Dict[str, np.ndarray] = {}
        for name, pack in packs.items():
            text_emb, mot_emb = self.embed_pack(pack)
            rp, mm = self._retrieval(text_emb, mot_emb, seed)
            div = _diversity(mot_emb, self.diversity_times)
            results[name] = {
                "n": int(mot_emb.shape[0]),
                "rp_top1": float(rp[0]),
                "rp_top2": float(rp[1]),
                "rp_top3": float(rp[2]),
                "mm_dist": float(mm),
                "diversity": float(div),
            }
            motion_embeddings[name] = mot_emb

        gt_mu, gt_cov = _activation_stats(motion_embeddings["Real"])
        for name, mot_emb in motion_embeddings.items():
            if name == "Real":
                results[name]["fid"] = 0.0
                continue
            mu, cov = _activation_stats(mot_emb)
            results[name]["fid"] = _stable_frechet(gt_mu, gt_cov, mu, cov)
        return results

    def evaluate_npz(
        self,
        gt_path: str | Path,
        pred_paths: Mapping[str, str | Path],
        seed: int = 42,
    ) -> Dict[str, Dict[str, object]]:
        gt = load_native262_pack(gt_path, self.max_len)
        preds = {name: load_native262_pack(path, self.max_len) for name, path in pred_paths.items()}
        return self.evaluate_packs(gt, preds, seed=seed)

    def write_json(self, results: Mapping[str, object], out_json: str | Path) -> None:
        out = Path(out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(out)


__all__ = ["InterHuman262Evaluator", "load_native262_pack"]
