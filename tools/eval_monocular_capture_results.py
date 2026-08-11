#!/usr/bin/env python3
"""Evaluate canonical Motius monocular-capture NPZ artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motius.evaluation.evaluators.monocular_capture import (
    evaluate_camera_coordinates,
    evaluate_common_joint_coordinates,
    evaluate_global_coordinates,
)
from motius.evaluation.monocular_capture import SUPPORTED_PROTOCOLS
from motius.motion.representation.monocular_capture import (
    MonocularCaptureResult,
    MonocularTrack,
    load_monocular_capture_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument(
        "--protocol",
        required=True,
        choices=sorted(SUPPORTED_PROTOCOLS),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-coverage", type=float, default=1.0)
    parser.add_argument("--allow-missing-tracks", action="store_true")
    return parser.parse_args()


def _indices_by_frame(
    prediction: MonocularTrack,
    target: MonocularTrack,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    frame_ids, prediction_indices, target_indices = np.intersect1d(
        prediction.frame_ids,
        target.frame_ids,
        assume_unique=True,
        return_indices=True,
    )
    valid = prediction.valid[prediction_indices] & target.valid[target_indices]
    target_valid_count = int(target.valid.sum())
    coverage = float(valid.sum() / max(target_valid_count, 1))
    return prediction_indices, target_indices, valid, coverage


def _subset(value: np.ndarray | None, indices: np.ndarray) -> np.ndarray | None:
    return None if value is None else value[indices]


def _same_joint_protocol(
    prediction: MonocularTrack,
    target: MonocularTrack,
) -> bool:
    return (
        prediction.body_model.lower() == target.body_model.lower()
        and prediction.joint_names
        and prediction.joint_names == target.joint_names
    )


def _evaluate_track(
    prediction: MonocularTrack,
    target: MonocularTrack,
    *,
    protocol: str,
) -> tuple[dict, dict[str, np.ndarray]]:
    prediction_indices, target_indices, valid, coverage = _indices_by_frame(
        prediction,
        target,
    )
    space = "world" if protocol == "emdb_2_global_v1" else "camera"
    prediction_joints = _subset(
        prediction.joints_world if space == "world" else prediction.joints_camera,
        prediction_indices,
    )
    target_joints = _subset(
        target.joints_world if space == "world" else target.joints_camera,
        target_indices,
    )
    if prediction_joints is None or target_joints is None:
        raise ValueError(
            f"Track {target.track_id} is missing required {space}-space joints."
        )
    if _same_joint_protocol(prediction, target):
        prediction_vertices = _subset(
            (
                prediction.vertices_world
                if space == "world"
                else prediction.vertices_camera
            ),
            prediction_indices,
        )
        target_vertices = _subset(
            target.vertices_world if space == "world" else target.vertices_camera,
            target_indices,
        )
        mesh_metrics_available = (
            prediction_vertices is not None and target_vertices is not None
        )
        if not mesh_metrics_available:
            prediction_vertices = None
            target_vertices = None
        if space == "camera":
            pelvis_indices = tuple(
                target.metadata.get("evaluation_pelvis_indices", (1, 2))
            )
            result = evaluate_camera_coordinates(
                prediction_joints,
                target_joints,
                prediction_vertices=prediction_vertices,
                target_vertices=target_vertices,
                valid=valid,
                pelvis_indices=pelvis_indices,
                fps=float(target.metadata.get("evaluation_fps", 30.0)),
            )
        else:
            result = evaluate_global_coordinates(
                prediction_joints,
                target_joints,
                prediction_vertices=prediction_vertices,
                target_vertices=target_vertices,
                valid=valid,
                fps=float(target.metadata.get("evaluation_fps", 30.0)),
            )
    else:
        mesh_metrics_available = False
        result = evaluate_common_joint_coordinates(
            prediction_joints,
            prediction.joint_names,
            prediction.body_model,
            target_joints,
            target.joint_names,
            target.body_model,
            space=space,
            valid=valid,
            fps=float(target.metadata.get("evaluation_fps", 30.0)),
        )
    record = {
        "track_id": target.track_id,
        "target_valid_frames": int(target.valid.sum()),
        "evaluated_frames": int(valid.sum()),
        "coverage_percent": coverage * 100.0,
        "prediction_body_model": prediction.body_model,
        "target_body_model": target.body_model,
        "mesh_metrics_available": mesh_metrics_available,
        "metrics": result.means,
        "protocol": result.protocol,
    }
    return record, result.per_frame


def evaluate_results(
    prediction: MonocularCaptureResult,
    target: MonocularCaptureResult,
    *,
    protocol: str,
    min_coverage: float,
    require_all_tracks: bool = True,
) -> dict:
    prediction_by_id = {track.track_id: track for track in prediction.tracks}
    records = []
    values: dict[str, list[np.ndarray]] = {}
    for target_track in target.tracks:
        try:
            prediction_track = prediction_by_id[target_track.track_id]
        except KeyError as exc:
            if require_all_tracks:
                raise ValueError(
                    f"Missing prediction track {target_track.track_id!r}."
                ) from exc
            records.append(
                {
                    "track_id": target_track.track_id,
                    "status": "missing_prediction",
                    "target_valid_frames": int(target_track.valid.sum()),
                    "evaluated_frames": 0,
                    "coverage_percent": 0.0,
                    "prediction_body_model": None,
                    "target_body_model": target_track.body_model,
                    "mesh_metrics_available": False,
                    "metrics": {},
                    "protocol": {"missing_prediction": True},
                }
            )
            continue
        record, per_frame = _evaluate_track(
            prediction_track,
            target_track,
            protocol=protocol,
        )
        if record["coverage_percent"] + 1e-9 < min_coverage * 100.0:
            raise ValueError(
                f"Track {target_track.track_id} coverage "
                f"{record['coverage_percent']:.2f}% is below "
                f"{min_coverage * 100.0:.2f}%."
            )
        records.append(record)
        for name, array in per_frame.items():
            if len(array):
                values.setdefault(name, []).append(array)
    summary_metrics = {
        name: float(np.concatenate(arrays).mean())
        for name, arrays in values.items()
    }
    metric_samples = {
        name: int(sum(len(array) for array in arrays))
        for name, arrays in values.items()
    }
    target_valid = sum(record["target_valid_frames"] for record in records)
    evaluated = sum(record["evaluated_frames"] for record in records)
    return {
        "schema_version": 1,
        "protocol": protocol,
        "prediction": prediction.public_manifest(),
        "target": target.public_manifest(),
        "population": len(records),
        "target_valid_frames": target_valid,
        "evaluated_frames": evaluated,
        "coverage_percent": 100.0 * evaluated / max(target_valid, 1),
        "complete_track_coverage": all(
            record.get("status") != "missing_prediction" for record in records
        ),
        "metrics": summary_metrics,
        "metric_samples": metric_samples,
        "tracks": records,
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.min_coverage <= 1.0:
        raise SystemExit("--min-coverage must be in [0, 1].")
    if "outputs" not in args.output.resolve().parts:
        raise SystemExit("--output must live under the repository outputs tree.")
    prediction = load_monocular_capture_result(args.prediction)
    target = load_monocular_capture_result(args.target)
    payload = evaluate_results(
        prediction,
        target,
        protocol=args.protocol,
        min_coverage=args.min_coverage,
        require_all_tracks=not args.allow_missing_tracks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["metrics"], indent=2))


if __name__ == "__main__":
    main()
