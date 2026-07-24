#!/usr/bin/env python3
"""Associate per-video predictions to 3DPW people and aggregate metrics."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.representation.monocular_capture import (
    MonocularCaptureResult,
    MonocularTrack,
    load_monocular_capture_result,
)
from tools.eval_monocular_capture_results import evaluate_results


def _target_bbox(track: MonocularTrack) -> np.ndarray:
    keypoints = np.asarray(track.native_parameters["poses2d_xyc"])
    boxes = np.zeros((track.num_frames, 4), dtype=np.float64)
    for frame, points in enumerate(keypoints):
        visible = points[:, 2] > 0
        if visible.any():
            xy = points[visible, :2]
            boxes[frame] = [
                xy[:, 0].min(),
                xy[:, 1].min(),
                xy[:, 0].max(),
                xy[:, 1].max(),
            ]
    return boxes


def _prediction_bbox(track: MonocularTrack) -> np.ndarray | None:
    value = track.native_parameters.get("tracking_bboxes")
    if value is None:
        value = track.native_parameters.get("bbox_xyxy")
    if value is None:
        return None
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[0] != track.num_frames or boxes.shape[1] < 4:
        raise ValueError(
            f"Track {track.track_id!r} has invalid tracking_bboxes {boxes.shape}."
        )
    return boxes[:, :4]


def _representative_intrinsics(
    result: MonocularCaptureResult,
) -> np.ndarray | None:
    if result.camera_intrinsics is None:
        return None
    intrinsics = np.asarray(result.camera_intrinsics, dtype=np.float64)
    if intrinsics.shape == (3, 3):
        return intrinsics
    if intrinsics.ndim == 3 and intrinsics.shape[1:] == (3, 3):
        finite = np.isfinite(intrinsics).all(axis=(1, 2))
        if finite.any():
            return np.median(intrinsics[finite], axis=0)
    return None


def _prediction_bbox_scale(
    prediction: MonocularCaptureResult,
    target: MonocularCaptureResult,
) -> tuple[float, float]:
    """Map PromptHMR's resized-video pixels into benchmark image pixels."""
    if prediction.source_model != "PromptHMR-Video":
        return 1.0, 1.0
    prediction_k = _representative_intrinsics(prediction)
    target_k = _representative_intrinsics(target)
    if prediction_k is None or target_k is None:
        return 1.0, 1.0
    prediction_center = prediction_k[:2, 2]
    target_center = target_k[:2, 2]
    if (
        not np.isfinite(prediction_center).all()
        or not np.isfinite(target_center).all()
        or (prediction_center <= 0).any()
        or (target_center <= 0).any()
    ):
        return 1.0, 1.0
    scale = target_center / prediction_center
    if (scale < 0.1).any() or (scale > 10.0).any():
        raise ValueError(
            "Implausible PromptHMR bbox pixel scale from camera intrinsics: "
            f"{scale.tolist()}"
        )
    return float(scale[0]), float(scale[1])


def _box_area(boxes: np.ndarray) -> np.ndarray:
    return np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(
        boxes[:, 3] - boxes[:, 1],
        0,
    )


def _mean_iou(
    prediction_track: MonocularTrack,
    prediction_boxes: np.ndarray,
    target_track: MonocularTrack,
    target_boxes: np.ndarray,
) -> float:
    frame_ids, prediction_indices, target_indices = np.intersect1d(
        prediction_track.frame_ids,
        target_track.frame_ids,
        assume_unique=True,
        return_indices=True,
    )
    if not len(frame_ids):
        return 0.0
    pred = prediction_boxes[prediction_indices]
    target = target_boxes[target_indices]
    valid = (
        prediction_track.valid[prediction_indices]
        & target_track.valid[target_indices]
        & (_box_area(pred) > 0)
        & (_box_area(target) > 0)
    )
    if not valid.any():
        return 0.0
    pred = pred[valid]
    target = target[valid]
    xy1 = np.maximum(pred[:, :2], target[:, :2])
    xy2 = np.minimum(pred[:, 2:4], target[:, 2:4])
    intersection = np.maximum(xy2 - xy1, 0).prod(axis=1)
    union = _box_area(pred) + _box_area(target) - intersection
    return float(np.divide(intersection, union, out=np.zeros_like(union), where=union > 0).mean())


def _associate(
    prediction: MonocularCaptureResult,
    target: MonocularCaptureResult,
) -> tuple[MonocularCaptureResult, dict]:
    target_boxes = [_target_bbox(track) for track in target.tracks]
    prediction_boxes = [_prediction_bbox(track) for track in prediction.tracks]
    bbox_scale = _prediction_bbox_scale(prediction, target)
    if bbox_scale != (1.0, 1.0):
        pixel_scale = np.asarray(
            [bbox_scale[0], bbox_scale[1], bbox_scale[0], bbox_scale[1]],
            dtype=np.float64,
        )
        prediction_boxes = [
            None if boxes is None else boxes * pixel_scale
            for boxes in prediction_boxes
        ]
    assignments: list[tuple[int, int, float]] = []
    mode = "largest_gt_track"

    if prediction.tracks and all(box is not None for box in prediction_boxes):
        scores = np.asarray(
            [
                [
                    _mean_iou(pred_track, pred_box, target_track, target_box)
                    for target_track, target_box in zip(target.tracks, target_boxes)
                ]
                for pred_track, pred_box in zip(prediction.tracks, prediction_boxes)
            ],
            dtype=np.float64,
        )
        pred_indices, target_indices = linear_sum_assignment(-scores)
        assignments = [
            (int(pred_index), int(target_index), float(scores[pred_index, target_index]))
            for pred_index, target_index in zip(pred_indices, target_indices)
            if scores[pred_index, target_index] > 0
        ]
        mode = "hungarian_mean_2d_bbox_iou"
    elif prediction.tracks and target.tracks:
        mean_areas = [
            float(_box_area(box)[_box_area(box) > 0].mean())
            if (_box_area(box) > 0).any()
            else 0.0
            for box in target_boxes
        ]
        assignments = [(0, int(np.argmax(mean_areas)), 0.0)]

    matched_tracks = tuple(
        replace(
            prediction.tracks[pred_index],
            track_id=target.tracks[target_index].track_id,
            metadata={
                **dict(prediction.tracks[pred_index].metadata),
                "3dpw_association": {
                    "mode": mode,
                    "target_track_id": target.tracks[target_index].track_id,
                    "score": score,
                },
            },
        )
        for pred_index, target_index, score in assignments
    )
    report = {
        "mode": mode,
        "prediction_bbox_scale_to_target": list(bbox_scale),
        "assignments": [
            {
                "prediction_track_id": prediction.tracks[pred_index].track_id,
                "target_track_id": target.tracks[target_index].track_id,
                "score": score,
            }
            for pred_index, target_index, score in assignments
        ],
        "unmatched_target_tracks": sorted(
            {track.track_id for track in target.tracks}
            - {track.track_id for track in matched_tracks}
        ),
    }
    return replace(prediction, tracks=matched_tracks), report


def _load_targets(index_path: Path) -> dict[str, MonocularCaptureResult]:
    index = json.loads(index_path.read_text())
    grouped: dict[str, list[MonocularCaptureResult]] = {}
    for record in index["artifacts"]:
        _, sequence, _track = record["sample_id"].split(":", 2)
        result = load_monocular_capture_result(index_path.parent / record["artifact"])
        grouped.setdefault(sequence, []).append(result)
    combined = {}
    for sequence, results in grouped.items():
        ordered = sorted(
            (result.tracks[0] for result in results),
            key=lambda track: track.track_id,
        )
        combined[sequence] = replace(
            results[0],
            tracks=tuple(ordered),
            metadata={**dict(results[0].metadata), "sequence_id": sequence},
        )
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--ground-truth-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    if "outputs" not in output.parts:
        raise ValueError("--output must live under outputs/.")

    targets = _load_targets(args.ground_truth_index)
    sequences = []
    weighted_values: dict[str, float] = {}
    weighted_counts: dict[str, int] = {}
    for sequence, target in sorted(targets.items()):
        prediction_path = args.prediction_dir / f"{sequence}.motius.npz"
        if prediction_path.is_file():
            prediction = load_monocular_capture_result(
                prediction_path,
                include_vertices=False,
            )
            associated, association = _associate(prediction, target)
        else:
            associated = replace(
                target,
                source_model="missing-prediction",
                source_revision="none",
                checkpoint_sha256="0" * 64,
                tracks=(),
                # Empty tracks imply zero frames; drop frame-aligned fields.
                camera_intrinsics=None,
                camera_to_world=None,
                frame_timestamps=None,
                world_coordinate_system=None,
            )
            association = {
                "mode": "missing_sequence_prediction",
                "assignments": [],
                "unmatched_target_tracks": [
                    track.track_id for track in target.tracks
                ],
            }
        result = evaluate_results(
            associated,
            target,
            protocol="3dpw_test_camera_v1",
            min_coverage=0.0,
            require_all_tracks=False,
        )
        result["sequence_id"] = sequence
        result["association"] = association
        sequences.append(result)
        for name, value in result["metrics"].items():
            count = result["metric_samples"][name]
            weighted_values[name] = weighted_values.get(name, 0.0) + value * count
            weighted_counts[name] = weighted_counts.get(name, 0) + count

    target_valid = sum(item["target_valid_frames"] for item in sequences)
    evaluated = sum(item["evaluated_frames"] for item in sequences)
    payload = {
        "schema_version": 1,
        "protocol": "3dpw_test_camera_v1",
        "ground_truth_index": args.ground_truth_index.name,
        "population_sequences": len(sequences),
        "population_tracks": sum(len(item["tracks"]) for item in sequences),
        "target_valid_frames": target_valid,
        "evaluated_frames": evaluated,
        "coverage_percent": 100.0 * evaluated / max(target_valid, 1),
        "complete_track_coverage": all(
            item["complete_track_coverage"] for item in sequences
        ),
        "metrics": {
            name: weighted_values[name] / weighted_counts[name]
            for name in weighted_values
        },
        "metric_samples": weighted_counts,
        "sequences": sequences,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({key: payload[key] for key in ("coverage_percent", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
