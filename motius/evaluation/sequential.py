"""Evaluation helpers for temporally composed BABEL motion sequences.

The public protocol operates on canonical SMPL-22 joint positions. Semantic
metrics are delegated to the Motius joint-position TMR evaluator while the
transition metrics in this module remain checkpoint-free.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import linalg

from motius.evaluation.babel import normalize_action_text
from motius.motion.skeleton.canonical import canonicalize_smpl22_joints


def caption_group_id(caption: str) -> str:
    """Normalize punctuation and spacing without merging semantic synonyms."""

    return normalize_action_text(caption)


@dataclass(frozen=True)
class SequentialSegment:
    """One captioned, half-open interval in a sequential motion."""

    caption: str
    start_frame: int
    end_frame: int
    action_group_id: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "SequentialSegment":
        segment = cls(
            caption=str(value["caption"]).strip(),
            start_frame=int(value["start_frame"]),
            end_frame=int(value["end_frame"]),
            action_group_id=(
                str(value["action_group_id"]).strip()
                if value.get("action_group_id")
                else None
            ),
        )
        if not segment.caption:
            raise ValueError("Sequential segment captions must not be empty.")
        if segment.start_frame < 0 or segment.end_frame <= segment.start_frame:
            raise ValueError(
                "Sequential segments require 0 <= start_frame < end_frame, got "
                f"[{segment.start_frame}, {segment.end_frame})."
            )
        return segment


@dataclass(frozen=True)
class SequentialCase:
    """A BABEL sequence and its ordered caption intervals."""

    case_id: str
    reference_path: Path | None
    prediction_path: Path
    segments: tuple[SequentialSegment, ...]

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object],
        *,
        base_dir: Path,
        prediction_dir: Path | None = None,
    ) -> "SequentialCase":
        case_id = str(value.get("case_id", "")).strip()
        if not case_id:
            raise ValueError("Every sequential case requires a non-empty case_id.")
        reference_value = value.get("reference_path")
        reference_path = (
            _resolve_path(base_dir, reference_value)
            if reference_value is not None
            else None
        )
        prediction_value = value.get("prediction_path")
        if prediction_value is not None:
            prediction_path = _resolve_path(base_dir, prediction_value)
        elif prediction_dir is not None:
            prediction_path = prediction_dir / f"{case_id}.npy"
        else:
            raise ValueError(
                f"Case {case_id!r} has no prediction_path and no prediction_dir was provided."
            )
        segments = tuple(
            SequentialSegment.from_mapping(item) for item in value.get("segments", [])
        )
        if not segments:
            raise ValueError(f"Case {case_id!r} has no captioned segments.")
        previous_end = 0
        for segment in segments:
            if segment.start_frame < previous_end:
                raise ValueError(f"Case {case_id!r} contains overlapping or unordered segments.")
            previous_end = segment.end_frame
        return cls(case_id, reference_path, prediction_path, segments)


def _resolve_path(base_dir: Path, value: object) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else base_dir / path


def load_joints66(path: str | Path) -> np.ndarray:
    """Load canonical SMPL-22 joints as a finite ``(T, 66)`` float32 array."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    motion = _as_joints66(np.load(path), source=str(path))
    if len(motion) < 4 or not np.isfinite(motion).all():
        raise ValueError(f"Motion at {path} must contain >=4 finite frames.")
    return motion


def load_joints66_pool(path: str | Path) -> list[np.ndarray]:
    """Load a padded ``motions``/``lengths`` joints66 reference archive."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path) as archive:
        if "motions" not in archive or "lengths" not in archive:
            raise ValueError(f"Reference archive {path} needs motions and lengths arrays.")
        motions = np.asarray(archive["motions"], dtype=np.float32)
        lengths = np.asarray(archive["lengths"], dtype=np.int64)
    if motions.ndim != 3 or motions.shape[2] != 66 or lengths.shape != (len(motions),):
        raise ValueError(
            f"Invalid joints66 pool at {path}: motions={motions.shape}, lengths={lengths.shape}."
        )
    result = []
    for index, length in enumerate(lengths):
        if length < 4 or length > motions.shape[1]:
            raise ValueError(f"Invalid motion length {length} at {path}[{index}].")
        result.append(_as_joints66(motions[index, : int(length)], source=f"{path}[{index}]"))
    return result


def _as_joints66(value: np.ndarray, *, source: str = "motion") -> np.ndarray:
    motion = np.asarray(value, dtype=np.float32)
    if motion.ndim == 3 and motion.shape[1:] == (22, 3):
        motion = motion.reshape(len(motion), 66)
    if motion.ndim != 2 or motion.shape[1] != 66:
        raise ValueError(f"Expected joints66 {source}, got shape {motion.shape}.")
    return motion


def _frechet_distance(first: np.ndarray, second: np.ndarray) -> float:
    first = _embedding_matrix(first)
    second = _embedding_matrix(second)
    if len(first) < 2 or len(second) < 2:
        raise ValueError("Transition FID requires at least two reference and predicted windows.")
    mean_first, mean_second = first.mean(0), second.mean(0)
    cov_first = np.cov(first, rowvar=False)
    cov_second = np.cov(second, rowvar=False)
    covariance, _ = linalg.sqrtm(cov_first.dot(cov_second), disp=False)
    if not np.isfinite(covariance).all():
        offset = np.eye(cov_first.shape[0]) * 1e-6
        covariance = linalg.sqrtm((cov_first + offset).dot(cov_second + offset))
    if np.iscomplexobj(covariance):
        covariance = covariance.real
    delta = mean_first - mean_second
    return float(
        delta.dot(delta)
        + np.trace(cov_first)
        + np.trace(cov_second)
        - 2.0 * np.trace(covariance)
    )


def _embedding_matrix(values: np.ndarray) -> np.ndarray:
    """Normalize evaluator outputs to one flat feature vector per motion."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError(f"Expected batched motion embeddings, got shape {values.shape}.")
    return values.reshape(len(values), -1)


def _diversity(values: np.ndarray, *, seed: int, samples: int = 300) -> float:
    values = _embedding_matrix(values)
    if not len(values):
        return 0.0
    rng = np.random.default_rng(seed)
    count = min(int(samples), len(values))
    left = values[rng.choice(len(values), count, replace=False)]
    right = values[rng.choice(len(values), count, replace=False)]
    return float(np.linalg.norm(left - right, axis=1).mean())


def _jerk_curve(windows: Sequence[np.ndarray], fps: float) -> np.ndarray:
    """Mean joint jerk magnitude in m/s^3 at each transition-relative frame."""

    if fps <= 0:
        raise ValueError("fps must be positive.")
    if not windows:
        raise ValueError("At least one transition window is required.")
    shaped = np.stack([item.reshape(len(item), 22, 3) for item in windows])
    jerk = np.diff(shaped, n=3, axis=1) * float(fps) ** 3
    return np.linalg.norm(jerk, axis=-1).mean(axis=(0, 2))


def _transition_physics(
    reference: Sequence[np.ndarray],
    predicted: Sequence[np.ndarray],
    *,
    fps: float,
) -> dict[str, float]:
    reference_curve = _jerk_curve(reference, fps)
    predicted_curve = _jerk_curve(predicted, fps)
    if reference_curve.shape != predicted_curve.shape:
        raise ValueError("Reference and predicted transition jerk curves must align.")
    delta = np.abs(predicted_curve - reference_curve)
    return {
        "peak_jerk_reference": float(reference_curve.max()),
        "peak_jerk_predicted": float(predicted_curve.max()),
        "auj_gap": float(np.trapz(delta, dx=1.0 / float(fps))),
    }


def _transition_slice(boundary: int, length: int, window_frames: int) -> slice | None:
    if window_frames < 4:
        raise ValueError("transition_frames must be at least 4.")
    left = int(window_frames) // 2
    start = boundary - left
    end = start + int(window_frames)
    if start < 0 or end > length:
        return None
    return slice(start, end)


def evaluate_sequential_cases(
    cases: Sequence[SequentialCase],
    evaluator,
    *,
    reference_segment_pool: Sequence[np.ndarray] | None = None,
    reference_transition_pool: Sequence[np.ndarray] | None = None,
    fps: float = 30.0,
    transition_frames: int = 30,
    chunk_size: int = 32,
    n_repeats: int = 1,
    seed: int = 0,
    protocol: str = "babel-sequential-joints66-v1",
) -> dict[str, object]:
    """Evaluate captioned subsequences and their transition neighborhoods."""

    if not cases:
        raise ValueError("At least one sequential case is required.")
    captions: list[str] = []
    paired_reference_segments: list[np.ndarray] = []
    predicted_segments: list[np.ndarray] = []
    paired_reference_transitions: list[np.ndarray] = []
    predicted_transitions: list[np.ndarray] = []
    action_group_ids: list[str | None] = []

    for case in cases:
        reference = load_joints66(case.reference_path) if case.reference_path else None
        predicted = load_joints66(case.prediction_path)
        required = max(segment.end_frame for segment in case.segments)
        if len(predicted) < required or (reference is not None and len(reference) < required):
            raise ValueError(
                f"Case {case.case_id!r} requires {required} frames, got "
                f"reference={len(reference) if reference is not None else 'none'} "
                f"and prediction={len(predicted)}."
            )
        for segment in case.segments:
            region = slice(segment.start_frame, segment.end_frame)
            captions.append(segment.caption)
            action_group_ids.append(segment.action_group_id)
            if reference is not None:
                paired_reference_segments.append(reference[region])
            predicted_segments.append(canonicalize_smpl22_joints(predicted[region]))
        for segment in case.segments[:-1]:
            region = _transition_slice(segment.end_frame, required, transition_frames)
            if region is not None:
                if reference is not None:
                    paired_reference_transitions.append(reference[region])
                predicted_transitions.append(canonicalize_smpl22_joints(predicted[region]))

    if len(captions) < 3:
        raise ValueError("Sequential semantic evaluation requires at least three segments.")
    has_action_groups = [value is not None for value in action_group_ids]
    if any(has_action_groups) and not all(has_action_groups):
        raise ValueError("Sequential manifests must define action_group_id for every segment.")
    if all(has_action_groups):
        positive_group_ids = [str(value) for value in action_group_ids]
        group_kind = "official_babel_act_cat"
        group_policy = "action_group_multi_positive"
    else:
        positive_group_ids = [caption_group_id(caption) for caption in captions]
        group_kind = "normalized_caption_fallback"
        group_policy = "caption_group_multi_positive"
    group_counts = Counter(positive_group_ids)
    retrieval = evaluator.evaluate(
        captions,
        predicted_segments,
        None,
        chunk_size=chunk_size,
        n_repeats=n_repeats,
        seed=seed,
        positive_group_ids=positive_group_ids,
    )
    subsequence = dict(retrieval)
    if "matching_score" in subsequence:
        subsequence["mm_dist"] = subsequence.pop("matching_score")

    semantic_reference = (
        [
            canonicalize_smpl22_joints(
                _as_joints66(item, source="reference segment")
            )
            for item in paired_reference_segments
        ]
        if reference_segment_pool is None
        else [
            canonicalize_smpl22_joints(
                _as_joints66(item, source="reference segment")
            )
            for item in reference_segment_pool
        ]
    )
    if not semantic_reference:
        raise ValueError(
            "Sequential FID requires a BABEL val reference segment pool or paired references."
        )
    reference_subsequence = None
    if reference_segment_pool is None:
        if len(semantic_reference) != len(captions):
            raise ValueError("Paired reference segments must match the caption count.")
        reference_subsequence = dict(
            evaluator.evaluate(
                captions,
                semantic_reference,
                None,
                chunk_size=chunk_size,
                n_repeats=n_repeats,
                seed=seed,
                positive_group_ids=positive_group_ids,
            )
        )
        if "matching_score" in reference_subsequence:
            reference_subsequence["mm_dist"] = reference_subsequence.pop(
                "matching_score"
            )
    reference_embeddings = evaluator.encode_motions(semantic_reference)
    predicted_embeddings = evaluator.encode_motions(predicted_segments)
    reference_diversity = _diversity(reference_embeddings, seed=seed)
    subsequence.update(
        {
            "fid": _frechet_distance(reference_embeddings, predicted_embeddings),
            "diversity_reference": reference_diversity,
            "diversity_predicted": _diversity(predicted_embeddings, seed=seed),
        }
    )
    if reference_subsequence is not None:
        reference_subsequence.update(
            {
                "fid": 0.0,
                "diversity_reference": reference_diversity,
                "diversity_predicted": reference_diversity,
            }
        )

    transition_reference = (
        [
            canonicalize_smpl22_joints(
                _as_joints66(item, source="reference transition")
            )
            for item in paired_reference_transitions
        ]
        if reference_transition_pool is None
        else [
            canonicalize_smpl22_joints(
                _as_joints66(item, source="reference transition")
            )
            for item in reference_transition_pool
        ]
    )
    if len(transition_reference) < 2:
        raise ValueError(
            "Sequential transition evaluation requires at least two reference windows."
        )
    if len(predicted_transitions) < 2:
        raise ValueError("Sequential transition evaluation requires at least two boundaries.")
    reference_embeddings = evaluator.encode_motions(transition_reference)
    predicted_embeddings = evaluator.encode_motions(predicted_transitions)
    transition = {
        "fid": _frechet_distance(reference_embeddings, predicted_embeddings),
        "diversity_reference": _diversity(reference_embeddings, seed=seed),
        "diversity_predicted": _diversity(predicted_embeddings, seed=seed),
        **_transition_physics(
            transition_reference,
            predicted_transitions,
            fps=fps,
        ),
    }
    group_summary = {
        "kind": group_kind,
        "normalization": (
            "babel-act-cat-v1" if all(has_action_groups) else "unicode_word_casefold"
        ),
        "r_precision_policy": group_policy,
        "unique": len(group_counts),
        "duplicate_groups": sum(count > 1 for count in group_counts.values()),
        "segments_in_duplicate_groups": sum(
            count for count in group_counts.values() if count > 1
        ),
    }
    result = {
        "protocol": str(protocol),
        "motion_representation": "SMPL-22 joints66",
        "evaluator": "Motius Joint-Position Evaluator",
        "fps": float(fps),
        "transition_frames": int(transition_frames),
        "n_cases": len(cases),
        "n_segments": len(captions),
        "retrieval_groups": group_summary,
        "n_reference_segments": len(semantic_reference),
        "n_transitions": len(predicted_transitions),
        "n_reference_transitions": len(transition_reference),
        "reference_subsequence": reference_subsequence,
        "reference_transition": {
            "fid": 0.0,
            "diversity": transition["diversity_reference"],
            "peak_jerk": transition["peak_jerk_reference"],
            "auj_gap": 0.0,
        },
        "subsequence": subsequence,
        "transition": transition,
    }
    if not all(has_action_groups):
        result["caption_groups"] = group_summary
    return result


__all__ = [
    "SequentialCase",
    "SequentialSegment",
    "evaluate_sequential_cases",
    "load_joints66",
    "load_joints66_pool",
]
