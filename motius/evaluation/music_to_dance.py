"""AIST++ Music-to-Dance evaluation protocol used by Bailando."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import linalg
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist
from scipy.signal import argrelextrema

from motius.evaluation.base_evaluator import BaseEvaluator
from motius.evaluation.metrics.dance_features import (
    extract_geometric_features,
    extract_kinetic_features,
)
from motius.evaluation.metrics.physical import (
    PHYSICAL_METRIC_KEYS,
    aggregate_physical_metrics,
    compute_physical_metrics,
)
from motius.registry import EVALUATORS


MUSIC_TO_DANCE_METRIC_KEYS = (
    "FID_k",
    "FID_g",
    "Diversity_k",
    "Diversity_g",
    "BeatAlign",
)
AISTPP_MUSIC_DANCE_EVALUATOR_REPO_ID = (
    "ZeyuLing/Motius-Evaluator-AISTPP-Music-to-Dance"
)
AISTPP_MUSIC_DANCE_EVALUATOR_FORMAT = "motius-aistpp-music-dance-evaluator-v1"


@dataclass(frozen=True)
class MusicDanceSample:
    """One generated/reference dance pair aligned to one music sequence."""

    pred_joints: np.ndarray
    gt_joints: np.ndarray
    music_beats: np.ndarray
    music_fps: float = 60.0
    motion_fps: float = 60.0
    name: str = ""


def _validate_joints(joints: np.ndarray, name: str) -> np.ndarray:
    values = np.asarray(joints, dtype=np.float32)
    if values.ndim == 2 and values.shape[1] == 72:
        values = values.reshape(-1, 24, 3)
    if values.ndim != 3 or values.shape[1:] != (24, 3):
        raise ValueError(f"{name} must have shape (T,24,3) or (T,72), got {values.shape}")
    if values.shape[0] < 5:
        raise ValueError(f"{name} needs at least five frames")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    return values


def root_anchor_motion(joints: np.ndarray) -> np.ndarray:
    """Apply Bailando's official first-root translation normalization."""

    values = _validate_joints(joints, "joints").copy()
    return values - values[:1, :1]


def _normalize_from_gt(gt: np.ndarray, pred: np.ndarray):
    mean = gt.mean(axis=0)
    std = gt.std(axis=0)
    return (gt - mean) / (std + 1e-10), (pred - mean) / (std + 1e-10)


def frechet_distance(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.ndim != 2 or gt.ndim != 2 or pred.shape[1] != gt.shape[1]:
        raise ValueError(f"FID expects (N,D) arrays, got {pred.shape} and {gt.shape}")
    if len(pred) < 2 or len(gt) < 2:
        raise ValueError("FID requires at least two generated and two GT clips")
    mu_pred, mu_gt = pred.mean(axis=0), gt.mean(axis=0)
    cov_pred = np.cov(pred, rowvar=False)
    cov_gt = np.cov(gt, rowvar=False)
    covariance_mean, _ = linalg.sqrtm(cov_pred.dot(cov_gt), disp=False)
    if not np.isfinite(covariance_mean).all():
        offset = np.eye(cov_pred.shape[0]) * 1e-5
        covariance_mean = linalg.sqrtm(
            (cov_pred + offset).dot(cov_gt + offset)
        )
    if np.iscomplexobj(covariance_mean):
        covariance_mean = covariance_mean.real
    difference = mu_pred - mu_gt
    value = (
        difference.dot(difference)
        + np.trace(cov_pred)
        + np.trace(cov_gt)
        - 2.0 * np.trace(covariance_mean)
    )
    return float(max(value, 0.0))


def average_pairwise_distance(features: np.ndarray) -> float:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Diversity requires at least two feature vectors")
    return float(pdist(values, metric="euclidean").mean())


def motion_beat_frames(joints: np.ndarray) -> np.ndarray:
    """Detect motion beats as local minima of smoothed mean joint speed."""

    values = _validate_joints(joints, "joints")
    velocity = np.linalg.norm(values[1:] - values[:-1], axis=2).mean(axis=1)
    smoothed = gaussian_filter(velocity, 5)
    return argrelextrema(smoothed, np.less)[0]


def beat_alignment_score(
    music_beats: np.ndarray,
    motion_beats: np.ndarray,
    *,
    music_fps: float = 60.0,
    motion_fps: float = 60.0,
) -> float:
    """Compute Bailando BeatAlign with explicit music/motion frame rates."""

    music = np.asarray(music_beats)
    if music.ndim != 1:
        raise ValueError(f"music_beats must be one-dimensional, got {music.shape}")
    if music.dtype == np.bool_ or np.array_equal(music, music.astype(bool)):
        music = np.flatnonzero(music)
    music = music.astype(np.float64) * float(motion_fps) / float(music_fps)
    motion = np.asarray(motion_beats, dtype=np.float64).reshape(-1)
    if not len(music) or not len(motion):
        return 0.0
    nearest_squared = np.min((music[:, None] - motion[None]) ** 2, axis=1)
    return float(np.exp(-nearest_squared / (2.0 * 9.0)).mean())


def _truncate_music_beats(
    music_beats: np.ndarray,
    *,
    motion_velocity_frames: int,
    music_fps: float,
    motion_fps: float,
) -> np.ndarray:
    """Match the official beat stream to the generated velocity sequence."""

    music = np.asarray(music_beats)
    music_limit = int(
        np.ceil(float(motion_velocity_frames) * float(music_fps) / float(motion_fps))
    )
    if music.dtype == np.bool_ or np.array_equal(music, music.astype(bool)):
        return music[:music_limit]
    return music[np.asarray(music) < music_limit]


def _rows_from_output(output: Mapping) -> list[MusicDanceSample]:
    pred = np.asarray(output["pred_joints"])
    gt = np.asarray(output["gt_joints"])
    beats = np.asarray(output["music_beats"])
    if pred.ndim == 3:
        pred = pred[None]
        gt = gt[None]
        beats = beats[None]
    if pred.shape[0] != gt.shape[0] or beats.shape[0] != pred.shape[0]:
        raise ValueError("pred_joints, gt_joints, and music_beats batch sizes differ")
    names = output.get("names") or [output.get("name", "")] * pred.shape[0]
    music_fps = float(output.get("music_fps", 60.0))
    motion_fps = float(output.get("motion_fps", 60.0))
    return [
        MusicDanceSample(
            pred_joints=pred[index],
            gt_joints=gt[index],
            music_beats=beats[index],
            music_fps=music_fps,
            motion_fps=motion_fps,
            name=str(names[index]),
        )
        for index in range(pred.shape[0])
    ]


@EVALUATORS.register_module()
class AISTPPMusicDanceEvaluator(BaseEvaluator):
    """Official Bailando quality/diversity/alignment plus physical metrics."""

    def __init__(
        self,
        *,
        max_frames: int = 1_200,
        physical: bool = True,
        reference_features: Mapping[str, np.ndarray] | None = None,
        reference_feature_path: str | Path | None = None,
    ):
        super().__init__()
        self.max_frames = int(max_frames)
        self.physical = bool(physical)
        if reference_features is not None and reference_feature_path is not None:
            raise ValueError(
                "Provide reference_features or reference_feature_path, not both"
            )
        if reference_feature_path is not None:
            with np.load(reference_feature_path, allow_pickle=False) as payload:
                reference_features = {
                    "kinetic": payload["kinetic"],
                    "geometric": payload["geometric"],
                    **(
                        {"names": payload["names"]}
                        if "names" in payload.files
                        else {}
                    ),
                    **(
                        {"skipped": payload["skipped"]}
                        if "skipped" in payload.files
                        else {}
                    ),
                }
        self.reference_audit = {
            key: np.asarray(reference_features[key]).copy()
            for key in ("names", "skipped")
            if reference_features is not None and key in reference_features
        }
        self.reference_features = (
            {
                "kinetic": np.asarray(reference_features["kinetic"], dtype=np.float64),
                "geometric": np.asarray(
                    reference_features["geometric"], dtype=np.float64
                ),
            }
            if reference_features is not None
            else None
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path = AISTPP_MUSIC_DANCE_EVALUATOR_REPO_ID,
        *,
        physical: bool = True,
        local_files_only: bool = False,
        revision: str | None = None,
    ) -> "AISTPPMusicDanceEvaluator":
        """Load the fixed AIST++ reference feature pool from a local/HF artifact."""

        source = Path(pretrained_model_name_or_path).expanduser()
        if source.is_dir():
            artifact = source
        else:
            from huggingface_hub import snapshot_download

            artifact = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=revision,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        "evaluator_config.json",
                        "aistpp_reference_features.npz",
                        "README.md",
                        "LICENSE*",
                        "ATTRIBUTIONS.md",
                    ],
                )
            )
        config = json.loads((artifact / "evaluator_config.json").read_text())
        if config.get("artifact_format") != AISTPP_MUSIC_DANCE_EVALUATOR_FORMAT:
            raise ValueError(
                "Unsupported AIST++ evaluator artifact format: "
                f"{config.get('artifact_format')!r}"
            )
        return cls(
            max_frames=int(config.get("max_frames", 1_200)),
            physical=physical,
            reference_feature_path=artifact / config.get(
                "reference_features", "aistpp_reference_features.npz"
            ),
        )

    def save_pretrained(self, save_directory: str | Path) -> str:
        """Save a self-contained AIST++ metric protocol artifact."""

        if self.reference_features is None:
            raise ValueError("A reference feature pool is required for export")
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        feature_payload = {
            "kinetic": self.reference_features["kinetic"].astype(np.float32),
            "geometric": self.reference_features["geometric"].astype(np.float32),
            **self.reference_audit,
        }
        np.savez_compressed(output / "aistpp_reference_features.npz", **feature_payload)
        config = {
            "artifact_format": AISTPP_MUSIC_DANCE_EVALUATOR_FORMAT,
            "evaluator_class": "motius.evaluation.AISTPPMusicDanceEvaluator",
            "dataset": "AIST++",
            "protocol": "Bailando CVPR 2022",
            "max_frames": self.max_frames,
            "reference_features": "aistpp_reference_features.npz",
            "num_reference_samples": int(len(self.reference_features["kinetic"])),
            "feature_dimensions": {
                "kinetic": int(self.reference_features["kinetic"].shape[1]),
                "geometric": int(self.reference_features["geometric"].shape[1]),
            },
            "skipped_source_entries": [
                Path(str(value)).name
                for value in self.reference_audit.get("skipped", [])
            ],
            "source_repository": "https://github.com/lisiyao21/Bailando",
            "source_revision": "cc90b98bff81c9709570db413c9610c2562e27ca",
        }
        (output / "evaluator_config.json").write_text(
            json.dumps(config, indent=2) + "\n", encoding="utf-8"
        )
        return str(output)

    def process(self, output) -> None:
        if isinstance(output, MusicDanceSample):
            self._results.append(output)
            return
        self._results.extend(_rows_from_output(output))

    def compute(self) -> dict[str, float | int]:
        if len(self._results) < 2:
            raise ValueError("Music-to-Dance evaluation requires at least two clips")

        pred_kinetic = []
        pred_geometric = []
        gt_kinetic = []
        gt_geometric = []
        alignments = []
        pred_physical = []
        gt_physical = []
        for sample in self._results:
            pred_full = root_anchor_motion(sample.pred_joints)
            gt_full = root_anchor_motion(sample.gt_joints)
            pred = pred_full[: self.max_frames]
            gt = gt_full[: self.max_frames]
            pred_kinetic.append(extract_kinetic_features(pred))
            pred_geometric.append(extract_geometric_features(pred))
            if self.reference_features is None:
                gt_kinetic.append(extract_kinetic_features(gt))
                gt_geometric.append(extract_geometric_features(gt))
            pred_motion_beats = motion_beat_frames(pred_full)
            alignments.append(
                beat_alignment_score(
                    _truncate_music_beats(
                        sample.music_beats,
                        motion_velocity_frames=len(pred_full) - 1,
                        music_fps=sample.music_fps,
                        motion_fps=sample.motion_fps,
                    ),
                    pred_motion_beats,
                    music_fps=sample.music_fps,
                    motion_fps=sample.motion_fps,
                )
            )
            if self.physical:
                pred_physical.append(compute_physical_metrics(pred[:, :22]))
                gt_physical.append(compute_physical_metrics(gt[:, :22]))

        pred_k = np.stack(pred_kinetic)
        pred_g = np.stack(pred_geometric)
        if self.reference_features is None:
            gt_k = np.stack(gt_kinetic)
            gt_g = np.stack(gt_geometric)
            reference_source = "paired_evaluation_gt"
        else:
            gt_k = self.reference_features["kinetic"]
            gt_g = self.reference_features["geometric"]
            reference_source = "aistpp_reference_feature_pool"
        gt_k, pred_k = _normalize_from_gt(gt_k, pred_k)
        gt_g, pred_g = _normalize_from_gt(gt_g, pred_g)

        metrics: dict[str, float | int | str] = {
            "FID_k": frechet_distance(pred_k, gt_k),
            "FID_g": frechet_distance(pred_g, gt_g),
            "Diversity_k": average_pairwise_distance(pred_k),
            "Diversity_g": average_pairwise_distance(pred_g),
            "BeatAlign": float(np.mean(alignments)),
            "GT_Diversity_k": average_pairwise_distance(gt_k),
            "GT_Diversity_g": average_pairwise_distance(gt_g),
            "num_reference_samples": len(gt_k),
            "num_samples": len(self._results),
            "reference_source": reference_source,
        }
        if self.physical:
            pred_values = aggregate_physical_metrics(pred_physical)
            gt_values = aggregate_physical_metrics(gt_physical)
            for key in PHYSICAL_METRIC_KEYS:
                metrics[f"Physical/{key}"] = float(pred_values[key])
                metrics[f"GT_Physical/{key}"] = float(gt_values[key])
        return metrics


BailandoEvaluator = AISTPPMusicDanceEvaluator


__all__ = [
    "AISTPP_MUSIC_DANCE_EVALUATOR_FORMAT",
    "AISTPP_MUSIC_DANCE_EVALUATOR_REPO_ID",
    "AISTPPMusicDanceEvaluator",
    "BailandoEvaluator",
    "MUSIC_TO_DANCE_METRIC_KEYS",
    "MusicDanceSample",
    "average_pairwise_distance",
    "beat_alignment_score",
    "frechet_distance",
    "motion_beat_frames",
    "root_anchor_motion",
]
