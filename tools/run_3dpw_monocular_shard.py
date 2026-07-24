#!/usr/bin/env python3
"""Run one sharded monocular-capture method over staged 3DPW Test videos."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.gem_smpl import GemSmplBundle
from motius.models.gem_x import GemXBundle
from motius.models.gvhmr import GVHMRBundle
from motius.models.prompthmr import PromptHMRBundle
from motius.motion.representation.monocular_capture import (
    load_monocular_capture_result,
    save_monocular_capture_result,
)
from motius.pipelines.gem_smpl import GemSmplPipeline
from motius.pipelines.gem_x import GemXPipeline
from motius.pipelines.gvhmr import GVHMRPipeline
from motius.pipelines.prompthmr import PromptHMRPipeline


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        required=True,
        choices=("prompthmr", "gem_smpl", "gem_x", "gvhmr"),
    )
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-method")
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--max-frames", type=int)
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
    parser.add_argument(
        "--prompthmr-root",
        type=Path,
        default=ROOT / "outputs/tmp/prompthmr/upstream",
    )
    parser.add_argument(
        "--prompthmr-python",
        type=Path,
        default=ROOT / "outputs/tmp/conda-envs/phmr_pt2.4/bin/python",
    )
    parser.add_argument(
        "--gem-smpl-root",
        type=Path,
        default=ROOT / "outputs/tmp/gem_smpl/upstream",
    )
    parser.add_argument(
        "--gem-x-root",
        type=Path,
        default=ROOT / "outputs/tmp/gem_x/upstream",
    )
    parser.add_argument(
        "--gvhmr-root",
        type=Path,
        default=ROOT / "outputs/tmp/gvhmr/upstream",
    )
    parser.add_argument(
        "--gvhmr-python",
        type=Path,
        default=ROOT / "outputs/tmp/gvhmr/conda-env/bin/python",
    )
    parser.add_argument(
        "--smplx-model",
        type=Path,
        default=ROOT / "checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz",
    )
    parser.add_argument("--smplx-version", default="2020")
    parser.add_argument(
        "--smplx-sha256",
        default="bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74",
    )
    return parser


def _pipeline(args: argparse.Namespace):
    if args.method == "prompthmr":
        bundle = PromptHMRBundle(
            upstream_dir=str(args.prompthmr_root),
            video_checkpoint="bedlam1+2",
            python_command=(str(args.prompthmr_python),),
        )
        return PromptHMRPipeline(bundle)
    if args.method == "gem_smpl":
        root = args.gem_smpl_root.resolve()
        return GemSmplPipeline(
            GemSmplBundle(
                runtime_root=str(root),
                checkpoint=str(root / "inputs/pretrained/gem_smpl.ckpt"),
            )
        )
    if args.method == "gvhmr":
        return GVHMRPipeline(
            GVHMRBundle(
                runtime_root=args.gvhmr_root.resolve(),
                python_executable=args.gvhmr_python.resolve(),
            )
        )
    root = args.gem_x_root.resolve()
    return GemXPipeline(
        GemXBundle(
            runtime_root=str(root),
            checkpoint=str(root / "inputs/pretrained/gem_soma.ckpt"),
            soma_assets=str(root / "inputs/soma_assets"),
        )
    )


def _target_tracks(index_path: Path, sequence: str):
    index_path = index_path.expanduser().resolve()
    index = json.loads(index_path.read_text())
    tracks = []
    for artifact in index["artifacts"]:
        metadata = artifact["public_manifest"]["metadata"]
        if metadata["sequence_id"] != sequence:
            continue
        result = load_monocular_capture_result(index_path.parent / artifact["artifact"])
        if len(result.tracks) != 1:
            raise ValueError(f"{artifact['artifact']}: expected one GT track.")
        tracks.append(result.tracks[0])
    if not tracks:
        raise ValueError(f"{sequence}: no target tracks in {index_path}.")
    return sorted(tracks, key=lambda track: track.track_id)


def _target_bbox_xys(track, frames: int) -> np.ndarray:
    poses2d = np.asarray(track.native_parameters.get("poses2d_xyc"))
    if poses2d.shape != (track.num_frames, 18, 3):
        raise ValueError(f"{track.track_id}: invalid official poses2d {poses2d.shape}.")
    xyxy = np.full((frames, 4), np.nan, dtype=np.float32)
    for local_index, frame_id in enumerate(track.frame_ids):
        visible = poses2d[local_index, :, 2] > 0
        if not np.any(visible):
            continue
        points = poses2d[local_index, visible, :2]
        xyxy[int(frame_id), :2] = points.min(axis=0)
        xyxy[int(frame_id), 2:] = points.max(axis=0)
    known = np.flatnonzero(np.isfinite(xyxy).all(axis=1))
    if not len(known):
        raise ValueError(f"{track.track_id}: no visible 2D target frames.")
    timeline = np.arange(frames, dtype=np.float32)
    dense_xyxy = np.stack(
        [
            np.interp(timeline, known, xyxy[known, coordinate])
            for coordinate in range(4)
        ],
        axis=1,
    )
    center = (dense_xyxy[:, :2] + dense_xyxy[:, 2:]) * 0.5
    size = np.max(dense_xyxy[:, 2:] - dense_xyxy[:, :2], axis=1) * 1.2
    bbox_xys = np.concatenate([center, size[:, None]], axis=1).astype(np.float32)
    padded = np.pad(bbox_xys, ((2, 2), (0, 0)), mode="edge")
    return np.stack(
        [padded[index : index + 5].mean(axis=0) for index in range(frames)]
    ).astype(np.float32)


def _target_bbox_xyxy(track, frames: int) -> np.ndarray:
    poses2d = np.asarray(track.native_parameters.get("poses2d_xyc"))
    if poses2d.shape != (track.num_frames, 18, 3):
        raise ValueError(f"{track.track_id}: invalid official poses2d {poses2d.shape}.")
    boxes = np.full((frames, 4), np.nan, dtype=np.float32)
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
    timeline = np.arange(frames, dtype=np.float32)
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
            [padded[index : index + 5].mean(axis=0) for index in range(frames)]
        )
    return dense.astype(np.float32)


def _tracking_bboxes(bbox_xys: np.ndarray) -> np.ndarray:
    half = bbox_xys[:, 2:3] * 0.5
    return np.concatenate(
        [bbox_xys[:, :2] - half, bbox_xys[:, :2] + half],
        axis=1,
    )


def _run_gem_smpl_targets(
    args: argparse.Namespace,
    pipeline,
    *,
    sequence: str,
    video: Path,
    work_dir: Path,
):
    import torch

    targets = _target_tracks(args.target_crop_index, sequence)
    predictions = []
    base_result = None
    for target in targets:
        target_work_dir = work_dir / target.track_id
        bbox_xys = _target_bbox_xys(target, target.num_frames)
        preprocess = target_work_dir / video.stem / "preprocess"
        preprocess.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(bbox_xys), preprocess / "bbx.pt")
        result = pipeline.run(
            video,
            target_work_dir,
            original_fps=30.0,
        )
        if len(result.tracks) != 1:
            raise ValueError(f"{sequence}/{target.track_id}: expected one prediction.")
        prediction = replace(
            result.tracks[0],
            track_id=target.track_id,
            native_parameters={
                **dict(result.tracks[0].native_parameters),
                "tracking_bboxes": _tracking_bboxes(bbox_xys),
            },
            metadata={
                **dict(result.tracks[0].metadata),
                "target_crop_source": "official_3dpw_poses2d_xyc",
                "target_crop_track_id": target.track_id,
                "target_crop_enlarge": 1.2,
                "target_crop_smoothing_window": 5,
            },
        )
        predictions.append(prediction)
        base_result = result
    assert base_result is not None
    return replace(
        base_result,
        tracks=tuple(predictions),
        metadata={
            **dict(base_result.metadata),
            "inference_protocol": "per_gt_track_crop_v1",
            "target_tracks": len(predictions),
        },
    )


def _run_gvhmr_targets(
    args: argparse.Namespace,
    pipeline,
    *,
    sequence: str,
    video: Path,
    work_dir: Path,
):
    targets = _target_tracks(args.target_crop_index, sequence)
    predictions = []
    base_result = None
    for target in targets:
        target_work_dir = work_dir / target.track_id
        bbox_xyxy = _target_bbox_xyxy(target, target.num_frames)
        result = pipeline.infer_video(
            video,
            target_work_dir,
            original_fps=30.0,
            use_dpvo=False,
            materialize_geometry=True,
            bbox_xyxy=bbox_xyxy,
        )
        if len(result.tracks) != 1:
            raise ValueError(f"{sequence}/{target.track_id}: expected one prediction.")
        prediction = replace(
            result.tracks[0],
            track_id=target.track_id,
            metadata={
                **dict(result.tracks[0].metadata),
                "target_crop_source": "official_3dpw_poses2d_xyc",
                "target_crop_track_id": target.track_id,
                "target_crop_enlarge": 1.2,
                "target_crop_aspect_ratio": "192:256",
                "target_crop_smoothing_window": 5,
                "target_crop_smoothing_passes": 2,
            },
        )
        predictions.append(prediction)
        if base_result is not None:
            np.testing.assert_allclose(
                result.camera_intrinsics,
                base_result.camera_intrinsics,
            )
            np.testing.assert_array_equal(
                result.frame_timestamps,
                base_result.frame_timestamps,
            )
        base_result = result
    assert base_result is not None
    return replace(
        base_result,
        tracks=tuple(predictions),
        metadata={
            **dict(base_result.metadata),
            "inference_protocol": "per_gt_track_crop_v1",
            "target_tracks": len(predictions),
        },
    )


def _run_gem_x_targets(
    args: argparse.Namespace,
    pipeline,
    *,
    sequence: str,
    video: Path,
    work_dir: Path,
):
    import torch

    targets = _target_tracks(args.target_crop_index, sequence)
    predictions = []
    base_result = None
    for target in targets:
        target_work_dir = work_dir / target.track_id
        bbox_xyxy = _target_bbox_xyxy(target, target.num_frames)
        center = (bbox_xyxy[:, :2] + bbox_xyxy[:, 2:]) * 0.5
        size = np.max(bbox_xyxy[:, 2:] - bbox_xyxy[:, :2], axis=1) * 1.2
        bbox_xys = np.concatenate([center, size[:, None]], axis=1).astype(
            np.float32
        )
        preprocess = target_work_dir / video.stem / "preprocess"
        preprocess.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "bbx_xyxy": torch.from_numpy(bbox_xyxy),
                "bbx_xys": torch.from_numpy(bbox_xys),
            },
            preprocess / "bbx.pt",
        )
        result = pipeline.run(
            video,
            target_work_dir,
            original_fps=30.0,
        )
        if len(result.tracks) != 1:
            raise ValueError(
                f"{sequence}/{target.track_id}: expected one prediction."
            )
        prediction = replace(
            result.tracks[0],
            track_id=target.track_id,
            native_parameters={
                **dict(result.tracks[0].native_parameters),
                "tracking_bboxes": bbox_xyxy,
            },
            metadata={
                **dict(result.tracks[0].metadata),
                "target_crop_source": "official_3dpw_poses2d_xyc",
                "target_crop_track_id": target.track_id,
                "target_crop_enlarge": 1.2,
                "target_crop_smoothing_window": 5,
                "target_crop_smoothing_passes": 2,
                "camera_trajectory_source": (
                    "official_demo_identity_fallback_without_external_vo"
                ),
            },
        )
        predictions.append(prediction)
        base_result = result
    assert base_result is not None
    return replace(
        base_result,
        tracks=tuple(predictions),
        metadata={
            **dict(base_result.metadata),
            "inference_protocol": "per_gt_track_crop_v1",
            "target_tracks": len(predictions),
            "camera_trajectory_source": (
                "official_demo_identity_fallback_without_external_vo"
            ),
            "world_metrics_eligible": False,
        },
    )


def _run_one(
    args: argparse.Namespace,
    pipeline,
    *,
    sequence: str,
    video: Path,
    work_dir: Path,
):
    if args.method == "prompthmr":
        result = pipeline(
            video,
            original_fps=30.0,
            output_fps=30.0,
            reuse_existing=True,
        )
        result = pipeline.replay_licensed_geometry(
            result,
            args.smplx_model,
            gender="neutral",
            model_version=args.smplx_version,
            expected_sha256=args.smplx_sha256,
            spaces=("camera",),
            include_vertices=False,
            device="cuda",
        )
    elif args.method == "gem_smpl":
        result = _run_gem_smpl_targets(
            args,
            pipeline,
            sequence=sequence,
            video=video,
            work_dir=work_dir,
        )
    elif args.method == "gvhmr":
        result = _run_gvhmr_targets(
            args,
            pipeline,
            sequence=sequence,
            video=video,
            work_dir=work_dir,
        )
    elif args.method == "gem_x":
        result = _run_gem_x_targets(
            args,
            pipeline,
            sequence=sequence,
            video=video,
            work_dir=work_dir,
        )
    else:
        result = pipeline.run(
            video,
            work_dir,
            original_fps=30.0,
        )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "benchmark": "3DPW Test",
            "protocol": "3dpw_test_camera_v1",
            "sequence_id": sequence,
            "video_sha256_source": "motius_3dpw_test_videos_v1",
            "shard_id": args.shard_id,
            "num_shards": args.num_shards,
        }
    )
    return replace(result, metadata=metadata)


def main() -> None:
    args = _parser().parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_id < args.num_shards:
        raise ValueError("Require 0 <= shard-id < num-shards.")
    output_root = args.output_root.expanduser().resolve()
    try:
        output_root.relative_to((ROOT / "outputs").resolve())
    except ValueError as exc:
        raise ValueError("--output-root must live under repository outputs/.") from exc
    video_dir = args.video_dir.expanduser().resolve()
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
    if not assigned:
        return

    pipeline = _pipeline(args)
    output_method = args.output_method or args.method
    prediction_dir = output_root / output_method / "predictions"
    work_root = output_root / output_method / "_official_runs"
    status_dir = output_root / output_method / "status"
    for path in (prediction_dir, work_root, status_dir):
        path.mkdir(parents=True, exist_ok=True)

    failures = []
    for record in assigned:
        sequence = record["sequence_id"]
        video = video_dir / record["video"]
        inference_video = video
        if args.max_frames is not None:
            if args.max_frames < 1:
                raise ValueError("--max-frames must be positive.")
            clip_dir = work_root / "_smoke_inputs"
            clip_dir.mkdir(parents=True, exist_ok=True)
            inference_video = clip_dir / f"{sequence}_first{args.max_frames}.avi"
            if not inference_video.is_file():
                subprocess.run(
                    [
                        "ffmpeg",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-y",
                        "-i",
                        str(video),
                        "-frames:v",
                        str(args.max_frames),
                        "-c:v",
                        "mpeg4",
                        "-q:v",
                        "2",
                        str(inference_video),
                    ],
                    check=True,
                )
        output = prediction_dir / f"{sequence}.motius.npz"
        status_path = status_dir / f"{sequence}.json"
        if output.exists() and not args.overwrite:
            continue
        try:
            result = _run_one(
                args,
                pipeline,
                sequence=sequence,
                video=inference_video,
                work_dir=work_root / sequence,
            )
            save_monocular_capture_result(result, output)
            payload = {
                "status": "complete",
                "method": args.method,
                "output_method": output_method,
                "sequence_id": sequence,
                "prediction": output.relative_to(output_root).as_posix(),
                "tracks": result.num_tracks,
                "frames": max(track.num_frames for track in result.tracks),
                "source_revision": result.source_revision,
                "checkpoint_sha256": result.checkpoint_sha256,
            }
        except Exception as exc:
            payload = {
                "status": "failed",
                "method": args.method,
                "output_method": output_method,
                "sequence_id": sequence,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            failures.append(sequence)
        status_path.write_text(json.dumps(payload, indent=2) + "\n")
        print(json.dumps({key: payload[key] for key in payload if key != "traceback"}))

    if failures:
        raise RuntimeError(
            f"{args.method} shard {args.shard_id} failed: {', '.join(failures)}"
        )


if __name__ == "__main__":
    main()
