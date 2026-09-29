#!/usr/bin/env python3
"""Run one HYMotion-V2M shard and export canonical 3DPW predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.representation.monocular_capture import (
    MonocularCaptureResult,
    load_monocular_capture_result,
    save_monocular_capture_result,
)
from motius.models.hymotion_v2m import HyMotionV2MBundle
from motius.models.hymotion_v2m.preprocess import V2MVideoPreprocessor
from motius.pipelines.hymotion_v2m import HyMotionV2MPipeline


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--mean-std-path", type=Path)
    parser.add_argument("--body-model-path", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam3d-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3d-mhr", type=Path, required=True)
    parser.add_argument("--yolox-checkpoint", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--output-method", default="hymotion_v2m")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--assignment-plan", type=Path)
    parser.add_argument(
        "--target-crop-index",
        type=Path,
        default=ROOT
        / (
            "outputs/evaluation/monocular_capture/ground_truth/"
            "3dpw_joint_only/ground_truth_index.json"
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _target_tracks(index_path: Path, sequence: str):
    index_path = index_path.expanduser().resolve()
    index = json.loads(index_path.read_text())
    tracks = []
    for artifact in index["artifacts"]:
        metadata = artifact["public_manifest"]["metadata"]
        if metadata["sequence_id"] != sequence:
            continue
        result = load_monocular_capture_result(
            index_path.parent / artifact["artifact"]
        )
        if len(result.tracks) != 1:
            raise ValueError(f"{artifact['artifact']}: expected one GT track.")
        tracks.append(result.tracks[0])
    if not tracks:
        raise ValueError(f"{sequence}: no target tracks in {index_path}.")
    return sorted(tracks, key=lambda track: track.track_id)


def _target_bbox_xyxy(track) -> np.ndarray:
    poses2d = np.asarray(track.native_parameters.get("poses2d_xyc"))
    if poses2d.shape != (track.num_frames, 18, 3):
        raise ValueError(
            f"{track.track_id}: invalid official poses2d {poses2d.shape}."
        )
    boxes = np.full((track.num_frames, 4), np.nan, dtype=np.float32)
    for local_index, frame_id in enumerate(track.frame_ids):
        visible = poses2d[local_index, :, 2] > 0
        if not np.any(visible):
            continue
        points = poses2d[local_index, visible, :2]
        boxes[int(frame_id), :2] = points.min(axis=0)
        boxes[int(frame_id), 2:] = points.max(axis=0)
    known = np.flatnonzero(np.isfinite(boxes).all(axis=1))
    if not len(known):
        raise ValueError(f"{track.track_id}: no visible 2D target frames.")
    timeline = np.arange(track.num_frames, dtype=np.float32)
    dense = np.stack(
        [
            np.interp(timeline, known, boxes[known, coordinate])
            for coordinate in range(4)
        ],
        axis=1,
    )
    for _ in range(2):
        padded = np.pad(dense, ((2, 2), (0, 0)), mode="edge")
        dense = np.stack(
            [
                padded[index : index + 5].mean(axis=0)
                for index in range(track.num_frames)
            ]
        )
    return dense.astype(np.float32)


def _convert(
    output: dict,
    *,
    sequence: str,
    checkpoint_sha256: str,
    shard_id: int,
    num_shards: int,
    track_id: str = "person_0",
) -> MonocularCaptureResult:
    requested_frames = int(np.asarray(output["keypoints3d"]).shape[1])
    return HyMotionV2MPipeline.to_monocular_capture_result(
        output,
        checkpoint_sha256=checkpoint_sha256,
        track_id=track_id,
        track_metadata={
            "crop_protocol": "caller_supplied_dense_target_bbox",
            "requested_frames": requested_frames,
        },
        metadata={
            "benchmark": "3DPW Test",
            "protocol": "3dpw_test_camera_v1",
            "sequence_id": sequence,
            "shard_id": shard_id,
            "num_shards": num_shards,
            "camera_motion": "static_identity_fallback",
            "requested_frames": requested_frames,
            "generated_frames": requested_frames,
            "ranking_eligible": False,
            "ranking_exclusion": (
                "static-camera fallback; camera-space metrics remain diagnostic "
                "until a video-only camera estimator is integrated"
            ),
        },
    )


def main() -> None:
    args = _parser().parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards.")
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to((ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("--output-root must live under repository outputs/.") from exc

    manifest = json.loads(args.video_manifest.read_text())
    records = sorted(manifest["videos"], key=lambda item: item["sequence_id"])
    if args.max_sequences is not None:
        if args.max_sequences < 1:
            raise ValueError("--max-sequences must be positive.")
        records = records[: args.max_sequences]
    if args.assignment_plan is None:
        assigned = records[args.shard_id :: args.num_shards]
    else:
        plan = json.loads(args.assignment_plan.read_text())
        if plan.get("num_shards") != args.num_shards:
            raise ValueError(
                "Assignment plan shard count does not match --num-shards."
            )
        record_by_id = {record["sequence_id"]: record for record in records}
        assigned_ids = plan.get("assignments", {}).get(str(args.shard_id), [])
        unknown = sorted(set(assigned_ids) - record_by_id.keys())
        if unknown:
            raise ValueError(
                "Assignment plan contains unknown sequences: "
                + ", ".join(unknown)
            )
        assigned = [record_by_id[sequence_id] for sequence_id in assigned_ids]
    prediction_dir = output_root / args.output_method / "predictions"
    status_dir = output_root / args.output_method / "status"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    if not args.overwrite:
        assigned = [
            record
            for record in assigned
            if not (
                prediction_dir / f"{record['sequence_id']}.motius.npz"
            ).exists()
        ]
    if not assigned:
        return

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    checkpoint = checkpoint_dir / "epoch100.ckpt"
    checkpoint_sha256 = (
        _sha256(checkpoint)
        if args.checkpoint_sha256 is None
        else args.checkpoint_sha256.lower()
    )
    if len(checkpoint_sha256) != 64:
        raise ValueError("--checkpoint-sha256 must be a SHA-256 digest.")
    bundle = HyMotionV2MBundle.from_pretrained(
        str(checkpoint_dir),
        mean_std_path=(
            str(args.mean_std_path.expanduser().resolve())
            if args.mean_std_path is not None
            else None
        ),
        body_model_path=(
            str(args.body_model_path.expanduser().resolve())
            if args.body_model_path is not None
            else None
        ),
        device="cuda",
    )
    pipeline = HyMotionV2MPipeline(bundle)
    preprocessor = V2MVideoPreprocessor(
        device="cuda",
        sam3d_ckpt=str(args.sam3d_checkpoint),
        sam3d_mhr=str(args.sam3d_mhr),
        yolox_ckpt=str(args.yolox_checkpoint),
        ffmpeg=args.ffmpeg,
    )
    failures = []

    for record in assigned:
        sequence = record["sequence_id"]
        prediction_path = prediction_dir / f"{sequence}.motius.npz"
        status_path = status_dir / f"{sequence}.json"
        try:
            target_results = []
            for target in _target_tracks(args.target_crop_index, sequence):
                generated = pipeline.infer_monocular_motion_capture(
                    str(args.video_dir / record["video"]),
                    work_dir=(
                        f"/tmp/motius_hymotion_v2m_{args.shard_id}/"
                        f"{sequence}/{target.track_id}"
                    ),
                    seeds=[0],
                    cfg_scale=1.0,
                    overlap_frames=30,
                    bbox_xyxy=_target_bbox_xyxy(target),
                    preprocessor=preprocessor,
                    checkpoint_sha256=checkpoint_sha256,
                    track_id=target.track_id,
                    track_metadata={
                        "crop_protocol": "caller_supplied_dense_target_bbox",
                    },
                    metadata={
                        "benchmark": "3DPW Test",
                        "protocol": "3dpw_test_camera_v1",
                        "sequence_id": sequence,
                        "shard_id": args.shard_id,
                        "num_shards": args.num_shards,
                    },
                )
                target_results.append(generated)
            base = max(
                target_results,
                key=lambda item: item.tracks[0].num_frames,
            )
            result = replace(
                base,
                tracks=tuple(item.tracks[0] for item in target_results),
                metadata={
                    **dict(base.metadata),
                    "inference_protocol": "per_gt_track_crop_v1",
                    "target_tracks": len(target_results),
                },
            )
            save_monocular_capture_result(result, prediction_path)
            status = {
                "status": "complete",
                "sequence_id": sequence,
                "frames": result.tracks[0].num_frames,
                "checkpoint_sha256": checkpoint_sha256,
            }
        except Exception as exc:
            failures.append(sequence)
            status = {
                "status": "failed",
                "sequence_id": sequence,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        status_path.write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps({key: value for key, value in status.items() if key != "traceback"}))

    if failures:
        raise RuntimeError(f"HYMotion-V2M shard failures: {', '.join(failures)}")


if __name__ == "__main__":
    main()
