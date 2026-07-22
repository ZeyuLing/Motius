"""Canonical T2M retrieval, FID, matching-distance, and diversity metrics."""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence

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
    """Compute retrieval recall with optional multi-positive semantic groups.

    Standard T2M benchmarks pair every text with exactly one motion, so the
    diagonal is the positive set. When ``positive_group_ids`` is provided,
    every motion sharing the query's group id is accepted as a positive instead
    of becoming a false negative.
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


def retrieval_audit(
    text_embeddings: np.ndarray,
    motion_embeddings: np.ndarray,
    *,
    chunk: int = 32,
    seed: int = 0,
    top_k: int = 3,
    positive_group_ids: Optional[Sequence[object]] = None,
    query_indices: Optional[Iterable[int]] = None,
) -> dict[int, dict[str, object]]:
    """Expose the exact candidate batches and rankings used by R-Precision.

    ``text_to_motion`` follows the leaderboard metric. ``motion_to_text`` is
    also reported because it gives the more intuitive answer to which captions
    a motion subclip resembles. Samples in the incomplete final batch are
    marked as unevaluated, matching :func:`aggregate_t2m_metrics`.
    """

    text = np.asarray(text_embeddings)
    motion = np.asarray(motion_embeddings)
    if text.ndim != 2 or motion.ndim != 2 or text.shape != motion.shape:
        raise ValueError(
            "text_embeddings and motion_embeddings must have the same two-dimensional "
            f"shape, got {text.shape} and {motion.shape}."
        )
    count = len(text)
    if count < 3:
        raise ValueError(f"At least three paired samples are required, got {count}.")
    chunk = max(3, min(int(chunk), count))
    top_k = max(1, min(int(top_k), chunk))
    used = count // chunk * chunk
    groups = (
        np.arange(count, dtype=object)
        if positive_group_ids is None
        else np.asarray(positive_group_ids, dtype=object)
    )
    if groups.ndim != 1 or len(groups) != count:
        raise ValueError("positive_group_ids must match the embedding count.")

    selected = set(range(count) if query_indices is None else map(int, query_indices))
    if any(index < 0 or index >= count for index in selected):
        raise ValueError("query_indices contains an out-of-range sample index.")
    records: dict[int, dict[str, object]] = {
        index: {"sample_index": index, "evaluated": False} for index in selected
    }
    order = np.random.default_rng(seed).permutation(count)

    def ranked_items(
        ranking: np.ndarray,
        distances: np.ndarray,
        batch_indices: np.ndarray,
        query_group: object,
    ) -> tuple[list[dict[str, object]], int | None]:
        positives = groups[batch_indices[ranking]] == query_group
        positive_locations = np.flatnonzero(positives)
        positive_rank = int(positive_locations[0] + 1) if len(positive_locations) else None
        items = [
            {
                "sample_index": int(batch_indices[local_index]),
                "distance": float(distances[local_index]),
                "is_positive": bool(groups[batch_indices[local_index]] == query_group),
            }
            for local_index in ranking[:top_k]
        ]
        return items, positive_rank

    for batch_index, start in enumerate(range(0, used, chunk)):
        batch_indices = order[start : start + chunk]
        requested = [local for local, index in enumerate(batch_indices) if int(index) in selected]
        if not requested:
            continue
        distances = euclidean_distance_matrix(text[batch_indices], motion[batch_indices])
        for local_index in requested:
            sample_index = int(batch_indices[local_index])
            query_group = groups[sample_index]
            t2m_order = np.argsort(distances[local_index])
            m2t_order = np.argsort(distances[:, local_index])
            t2m_top, t2m_rank = ranked_items(
                t2m_order, distances[local_index], batch_indices, query_group
            )
            m2t_top, m2t_rank = ranked_items(
                m2t_order, distances[:, local_index], batch_indices, query_group
            )
            records[sample_index] = {
                "sample_index": sample_index,
                "evaluated": True,
                "batch_index": int(batch_index),
                "batch_size": int(chunk),
                "text_to_motion": {
                    "positive_rank": t2m_rank,
                    "top": t2m_top,
                },
                "motion_to_text": {
                    "positive_rank": m2t_rank,
                    "top": m2t_top,
                },
            }
    return records


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


def l2_normalize_embeddings(
    embeddings: np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Return one unit-length feature vector per sample.

    uTMR FID is defined in this normalized latent space so that encoder feature
    scale cannot dominate the distribution distance. Retrieval, MM-Dist, and
    Diversity continue to use the evaluator's native embeddings.
    """

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"Expected batched embeddings, got shape {values.shape}.")
    values = values.reshape(len(values), -1)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= eps):
        indices = np.flatnonzero(norms[:, 0] <= eps).tolist()
        raise ValueError(f"Cannot L2-normalize zero embeddings at indices {indices}.")
    return values / norms


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
    normalize_fid: bool = True,
) -> Dict[str, object]:
    n = min(len(text_embeddings), len(real_embeddings), len(predicted_embeddings))
    if n < 3:
        raise ValueError(f"At least three paired samples are required, got {n}.")
    chunk = max(3, min(int(chunk), n))
    text_embeddings = text_embeddings[:n]
    real_embeddings = real_embeddings[:n]
    predicted_embeddings = predicted_embeddings[:n]
    if normalize_fid:
        fid_real_embeddings = l2_normalize_embeddings(real_embeddings)
        fid_predicted_embeddings = l2_normalize_embeddings(predicted_embeddings)
        fid_embedding_space = "l2_normalized"
    else:
        fid_real_embeddings = np.asarray(
            real_embeddings, dtype=np.float64
        ).reshape(n, -1)
        fid_predicted_embeddings = np.asarray(
            predicted_embeddings, dtype=np.float64
        ).reshape(n, -1)
        fid_embedding_space = "native_raw"
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
        mean_real, cov_real = _activation_stats(fid_real_embeddings[order])
        mean_pred, cov_pred = _activation_stats(fid_predicted_embeddings[order])
        fids.append(_frechet(mean_real, cov_real, mean_pred, cov_pred))
        real_div.append(diversity(real_embeddings, rng=rng))
        pred_div.append(diversity(predicted_embeddings, rng=rng))
    r_array = np.stack(r_values)
    return {
        "n_samples_used": int(used),
        "n_repeats": int(n_repeats),
        "r_precision_policy": (
            "group_multi_positive"
            if group_ids is not None
            else "paired_diagonal_single_positive"
        ),
        "r_precision": r_array.mean(0).tolist(),
        "r_precision_std": r_array.std(0).tolist(),
        "matching_score": float(np.mean(matching_values)),
        "fid": float(np.mean(fids)),
        "fid_embedding_space": fid_embedding_space,
        "diversity_reference": float(np.mean(real_div)),
        "diversity_predicted": float(np.mean(pred_div)),
    }


__all__ = [
    "aggregate_t2m_metrics",
    "diversity",
    "l2_normalize_embeddings",
    "r_precision",
    "retrieval_audit",
]
