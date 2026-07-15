"""Canonical T2M retrieval, FID, matching-distance, and diversity metrics."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def euclidean_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    squared = -2 * a @ b.T
    squared += (a**2).sum(axis=1, keepdims=True)
    squared += (b**2).sum(axis=1)
    return np.sqrt(np.maximum(squared, 0))


def r_precision(
    text: np.ndarray,
    motion: np.ndarray,
    top_k: int = 3,
    positive_group_ids: Optional[Sequence[object]] = None,
):
    """Compute retrieval recall with optional multi-positive caption groups.

    Standard T2M benchmarks pair every text with exactly one motion, so the
    diagonal is the positive set. BABEL contains many repeated action captions;
    when ``positive_group_ids`` is provided, every motion sharing the query's
    group id is accepted as a positive instead of becoming a false negative.
    """

    distances = euclidean_distance_matrix(text, motion)
    ranking = np.argsort(distances, axis=1)
    if positive_group_ids is None:
        positive = np.eye(len(text), dtype=bool)
    else:
        groups = np.asarray(positive_group_ids)
        if groups.ndim != 1 or len(groups) != len(text):
            raise ValueError(
                "positive_group_ids must be a one-dimensional sequence matching "
                f"the embedding count, got shape {groups.shape}."
            )
        positive = groups[:, None] == groups[None, :]
    correct = np.zeros(len(text), dtype=bool)
    counts = np.zeros(top_k, dtype=np.float64)
    for rank in range(top_k):
        correct |= positive[np.arange(len(text)), ranking[:, rank]]
        counts[rank] = correct.sum()
    matching = distances.copy()
    matching[~positive] = np.inf
    return counts, float(matching.min(axis=1).sum())


def diversity(
    embeddings: np.ndarray,
    n: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> float:
    n = min(int(n), len(embeddings))
    if n == 0:
        return 0.0
    rng = rng or np.random.default_rng()
    first = embeddings[rng.choice(len(embeddings), n, replace=False)]
    second = embeddings[rng.choice(len(embeddings), n, replace=False)]
    return float(np.linalg.norm(first - second, axis=1).mean())


def _activation_stats(values: np.ndarray):
    return values.mean(axis=0), np.cov(values, rowvar=False)


def _frechet(mean_a, cov_a, mean_b, cov_b, eps: float = 1e-6) -> float:
    from scipy import linalg

    difference = mean_a - mean_b
    covariance, _ = linalg.sqrtm(cov_a.dot(cov_b), disp=False)
    if not np.isfinite(covariance).all():
        offset = np.eye(cov_a.shape[0]) * eps
        covariance = linalg.sqrtm((cov_a + offset).dot(cov_b + offset))
    if np.iscomplexobj(covariance):
        covariance = covariance.real
    return float(
        difference.dot(difference)
        + np.trace(cov_a)
        + np.trace(cov_b)
        - 2 * np.trace(covariance)
    )


def aggregate_t2m_metrics(
    text_embeddings: np.ndarray,
    real_embeddings: np.ndarray,
    predicted_embeddings: np.ndarray,
    n_repeats: int = 1,
    chunk: int = 32,
    seed: int = 0,
    positive_group_ids: Optional[Sequence[object]] = None,
) -> Dict[str, object]:
    n = min(len(text_embeddings), len(real_embeddings), len(predicted_embeddings))
    if n < 3:
        raise ValueError(f"At least three paired samples are required, got {n}.")
    chunk = max(3, min(int(chunk), n))
    text_embeddings = text_embeddings[:n]
    real_embeddings = real_embeddings[:n]
    predicted_embeddings = predicted_embeddings[:n]
    group_ids = None
    if positive_group_ids is not None:
        if len(positive_group_ids) < n:
            raise ValueError("positive_group_ids must cover every evaluated sample.")
        group_ids = np.asarray(positive_group_ids, dtype=object)[:n]
    rng = np.random.default_rng(seed)
    r_values, matching_values, fids, real_div, pred_div = [], [], [], [], []
    used = n // chunk * chunk
    for _ in range(int(n_repeats)):
        order = rng.permutation(n)
        counts = np.zeros(3, dtype=np.float64)
        matching = 0.0
        for start in range(0, used, chunk):
            indices = order[start : start + chunk]
            value, distance = r_precision(
                text_embeddings[indices],
                predicted_embeddings[indices],
                top_k=3,
                positive_group_ids=(group_ids[indices] if group_ids is not None else None),
            )
            counts += value
            matching += distance
        r_values.append(counts / used)
        matching_values.append(matching / used)
        mean_real, cov_real = _activation_stats(real_embeddings[order])
        mean_pred, cov_pred = _activation_stats(predicted_embeddings[order])
        fids.append(_frechet(mean_real, cov_real, mean_pred, cov_pred))
        real_div.append(diversity(real_embeddings, rng=rng))
        pred_div.append(diversity(predicted_embeddings, rng=rng))
    r_array = np.stack(r_values)
    return {
        "n_samples_used": int(used),
        "n_repeats": int(n_repeats),
        "r_precision_policy": (
            "caption_group_multi_positive"
            if group_ids is not None
            else "paired_diagonal_single_positive"
        ),
        "r_precision": r_array.mean(0).tolist(),
        "r_precision_std": r_array.std(0).tolist(),
        "matching_score": float(np.mean(matching_values)),
        "fid": float(np.mean(fids)),
        "diversity_reference": float(np.mean(real_div)),
        "diversity_predicted": float(np.mean(pred_div)),
    }


__all__ = ["aggregate_t2m_metrics", "diversity", "r_precision"]
