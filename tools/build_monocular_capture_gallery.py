#!/usr/bin/env python3
"""Build a local-only Three.js prediction/GT viewer for monocular capture."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.motion.representation.monocular_capture import (  # noqa: E402
    MonocularTrack,
    load_monocular_capture_result,
)
from motius.motion.representation.rotation import (  # noqa: E402
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
)
from smpl_gallery_assets import encode_motion135  # noqa: E402


SMPL24_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8,
    9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prediction-dir",
        type=Path,
        default=ROOT
        / "outputs/evaluation/monocular_capture/3dpw_test/gem_smpl/predictions",
    )
    parser.add_argument(
        "--ground-truth-index",
        type=Path,
        default=ROOT
        / (
            "outputs/evaluation/monocular_capture/ground_truth/"
            "3dpw_joint_only/ground_truth_index.json"
        ),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=ROOT
        / "outputs/evaluation/monocular_capture/3dpw_test/gem_smpl/metrics.json",
    )
    parser.add_argument(
        "--official-run-dir",
        type=Path,
        default=ROOT
        / (
            "outputs/evaluation/monocular_capture/3dpw_test/"
            "gem_smpl/_official_runs"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / (
            "outputs/visualization/monocular_capture/3dpw_test/"
            "gem_smpl/interactive"
        ),
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Optional sequence ID to include; repeat for a focused preview.",
    )
    return parser.parse_args()


def require_output_path(path: Path) -> Path:
    output = path.expanduser().resolve()
    output.relative_to((ROOT / "outputs").resolve())
    return output


def track_by_id(tracks: tuple[MonocularTrack, ...], track_id: str) -> MonocularTrack:
    for track in tracks:
        if track.track_id == track_id:
            return track
    raise KeyError(f"Track {track_id!r} not found.")


def dense_joints(track: MonocularTrack, frames: int) -> tuple[np.ndarray, np.ndarray]:
    if track.joints_camera is None:
        raise ValueError(f"{track.track_id} has no camera joints.")
    if tuple(track.joint_names) != (
        "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee",
        "Spine2", "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head", "L_Shoulder",
        "R_Shoulder", "L_Elbow", "R_Elbow", "L_Wrist", "R_Wrist",
        "L_Hand", "R_Hand",
    ):
        raise ValueError(f"Expected audited SMPL24 joints, got {track.joint_names}.")

    values = np.asarray(track.joints_camera, dtype=np.float32)
    frame_ids = np.asarray(track.frame_ids, dtype=np.int64)
    source_valid = np.asarray(track.valid, dtype=np.bool_)
    finite = np.isfinite(values).all(axis=(1, 2))
    source_valid = source_valid & finite
    in_range = (frame_ids >= 0) & (frame_ids < frames)

    dense = np.zeros((frames, 24, 3), dtype=np.float32)
    valid = np.zeros(frames, dtype=np.bool_)
    selected_ids = frame_ids[in_range]
    dense[selected_ids] = values[in_range]
    valid[selected_ids] = source_valid[in_range]
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        raise ValueError(f"{track.track_id} has no finite valid frames.")

    # Invalid frames are hidden in the viewer, but nearest-frame filling keeps
    # their placeholder coordinates inside the quantization range.
    timeline = np.arange(frames)
    for joint in range(24):
        for axis in range(3):
            dense[:, joint, axis] = np.interp(
                timeline,
                valid_indices,
                dense[valid_indices, joint, axis],
            )
    return dense, valid


def encode_positions(values: np.ndarray) -> tuple[bytes, dict]:
    flat = values.reshape(-1, 3)
    minimum = flat.min(axis=0).astype(np.float32)
    maximum = flat.max(axis=0).astype(np.float32)
    scale = np.maximum((maximum - minimum) / 65535.0, 1e-8).astype(np.float32)
    quantized = np.rint((values - minimum) / scale).clip(0, 65535).astype("<u2")
    return quantized.tobytes(), {
        "position_count": int(quantized.size),
        "position_minimum": minimum.tolist(),
        "position_scale": scale.tolist(),
    }


def metric_error(
    prediction: np.ndarray,
    target: np.ndarray,
    prediction_valid: np.ndarray,
    target_valid: np.ndarray,
) -> np.ndarray:
    error = np.full(len(prediction), np.nan, dtype="<f4")
    valid = prediction_valid & target_valid
    pred = prediction[valid] - prediction[valid, :1]
    gt = target[valid] - target[valid, :1]
    error[valid] = np.linalg.norm(pred - gt, axis=-1).mean(axis=-1) * 1000.0
    return error


def camera_motion135(track: MonocularTrack) -> np.ndarray:
    if track.poses_axis_angle is None:
        raise ValueError(f"{track.track_id} has no SMPL pose parameters.")
    pose = np.asarray(track.poses_axis_angle, dtype=np.float32)
    if pose.ndim != 3 or pose.shape[1] < 22 or pose.shape[2] != 3:
        raise ValueError(f"Unexpected SMPL pose shape for {track.track_id}: {pose.shape}")
    pose = pose[:, :22].copy()
    if track.metadata.get("poses_axis_angle_space") == "3dpw_sequence_world":
        world_to_camera = np.asarray(
            track.native_parameters["cam_poses_world_to_camera"],
            dtype=np.float32,
        )
        translation_world = np.asarray(
            track.native_parameters["smpl_trans_world"],
            dtype=np.float32,
        )
        global_world = axis_angle_to_matrix(pose[:, 0])
        global_camera = world_to_camera[:, :3, :3] @ global_world
        # The first two columns are sufficient for Three.js rot6d replay.
        rotation6d = matrix_to_rotation_6d(
            np.concatenate(
                [
                    global_camera[:, None],
                    axis_angle_to_matrix(pose[:, 1:].reshape(-1, 3)).reshape(
                        len(pose), 21, 3, 3
                    ),
                ],
                axis=1,
            ),
            convention="row",
        ).reshape(len(pose), 132)
        translation = (
            np.einsum(
                "tij,tj->ti",
                world_to_camera[:, :3, :3],
                translation_world,
            )
            + world_to_camera[:, :3, 3]
        )
    else:
        rotation6d = matrix_to_rotation_6d(
            axis_angle_to_matrix(pose.reshape(-1, 3)).reshape(
                len(pose), 22, 3, 3
            ),
            convention="row",
        ).reshape(len(pose), 132)
        if track.root_translation_camera is None:
            raise ValueError(f"{track.track_id} has no camera translation.")
        translation = np.asarray(track.root_translation_camera, dtype=np.float32)
    motion = np.concatenate([translation, rotation6d], axis=1).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(motion).all():
        raise ValueError(f"{track.track_id} camera SMPL replay is non-finite.")
    return motion


def world_motion135(track: MonocularTrack) -> np.ndarray:
    if "body_params_global.global_orient" in track.native_parameters:
        orient = np.asarray(
            track.native_parameters["body_params_global.global_orient"],
            dtype=np.float32,
        ).reshape(-1, 1, 3)
        body_pose = np.asarray(
            track.native_parameters["body_params_global.body_pose"],
            dtype=np.float32,
        ).reshape(len(orient), 21, 3)
        pose = np.concatenate([orient, body_pose], axis=1)
        translation = np.asarray(
            track.native_parameters["body_params_global.transl"],
            dtype=np.float32,
        )
    else:
        if track.poses_axis_angle is None:
            raise ValueError(f"{track.track_id} has no world SMPL pose.")
        pose = np.asarray(track.poses_axis_angle, dtype=np.float32)[:, :22]
        translation = np.asarray(
            track.native_parameters["smpl_trans_world"],
            dtype=np.float32,
        )
    rotation6d = matrix_to_rotation_6d(
        axis_angle_to_matrix(pose.reshape(-1, 3)).reshape(
            len(pose), 22, 3, 3
        ),
        convention="row",
    ).reshape(len(pose), 132)
    motion = np.concatenate([translation, rotation6d], axis=1).astype(
        np.float32,
        copy=False,
    )
    if not np.isfinite(motion).all():
        raise ValueError(f"{track.track_id} world SMPL replay is non-finite.")
    return motion


def target_lookup(index: dict, root: Path) -> dict[tuple[str, str], Path]:
    lookup = {}
    for item in index["artifacts"]:
        public = item["public_manifest"]
        metadata = public["metadata"]
        lookup[(metadata["sequence_id"], metadata["track_id"])] = (
            root / item["artifact"]
        )
    return lookup


def load_tracking_bbox(
    run_root: Path,
    sequence: str,
    track_id: str,
    frames: int,
) -> np.ndarray:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Loading the trusted GEM bbox cache requires torch.") from exc
    target_root = run_root / sequence / track_id
    candidates = sorted(target_root.rglob("bbx.pt"))
    if not candidates:
        candidates = sorted((run_root / sequence).rglob("bbx.pt"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one GEM bbox cache for {sequence}/{track_id}, "
            f"found {candidates}."
        )
    value = torch.load(candidates[0], map_location="cpu", weights_only=False)
    if isinstance(value, dict):
        value = value.get("bbx_xys", value.get("bbox_xys"))
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    boxes = np.asarray(value, dtype=np.float32)
    if boxes.shape != (frames, 3) or not np.isfinite(boxes).all():
        raise ValueError(f"{sequence}: expected finite bbox_xys ({frames}, 3).")
    return boxes


def target_poses2d_bbox(track, frames: int) -> np.ndarray:
    output = np.full((frames, 4), np.nan, dtype=np.float32)
    poses2d = np.asarray(track.native_parameters.get("poses2d_xyc"))
    if poses2d.shape[:2] != (track.num_frames, 18) or poses2d.shape[2] < 3:
        raise ValueError(f"{track.track_id}: missing official 3DPW poses2d.")
    for local_index, frame_id in enumerate(track.frame_ids):
        visible = poses2d[local_index, :, 2] > 0
        if not track.valid[local_index] or not np.any(visible):
            continue
        points = poses2d[local_index, visible, :2]
        output[int(frame_id), :2] = points.min(axis=0)
        output[int(frame_id), 2:] = points.max(axis=0)
    return output


def main() -> None:
    args = parse_args()
    prediction_dir = args.prediction_dir.expanduser().resolve()
    index_path = args.ground_truth_index.expanduser().resolve()
    metrics_path = args.metrics.expanduser().resolve()
    official_run_dir = args.official_run_dir.expanduser().resolve()
    output = require_output_path(args.output_dir)
    assets = output / "assets"
    existing_cases = {}
    existing_manifest_path = output / "manifest.json"
    if existing_manifest_path.is_file():
        existing_manifest = json.loads(existing_manifest_path.read_text())
        existing_cases = {
            case["case_id"]: case for case in existing_manifest.get("cases", [])
        }
    output.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    for stale in assets.glob("*.joints"):
        stale.unlink()
    shutil.copy2(
        Path(__file__).with_name("monocular_capture_gallery.html"),
        output / "index.html",
    )

    metrics = json.loads(metrics_path.read_text())
    gt_index = json.loads(index_path.read_text())
    gt_lookup = target_lookup(gt_index, index_path.parent)
    cases = []
    selected_cases = set(args.case)
    sequence_records = [
        record
        for record in metrics["sequences"]
        if not selected_cases or record["sequence_id"] in selected_cases
    ]
    if selected_cases != {record["sequence_id"] for record in sequence_records}:
        missing = selected_cases - {
            record["sequence_id"] for record in sequence_records
        }
        raise ValueError(f"Unknown sequence IDs: {sorted(missing)}")
    for sequence_record in sequence_records:
        sequence = sequence_record["sequence_id"]
        assignments = sequence_record["association"]["assignments"]
        if not assignments:
            raise ValueError(f"Expected at least one GEM-SMPL assignment for {sequence}.")
        assignment = assignments[0]

        prediction = load_monocular_capture_result(
            prediction_dir / f"{sequence}.motius.npz"
        )
        pred_track = track_by_id(
            prediction.tracks,
            assignment["prediction_track_id"],
        )
        target_tracks = []
        for record in sequence_record["tracks"]:
            track_id = record["track_id"]
            if track_id != assignment["target_track_id"]:
                continue
            target = load_monocular_capture_result(gt_lookup[(sequence, track_id)])
            target_tracks.append(
                (
                    track_by_id(target.tracks, track_id),
                    record,
                )
            )
        frames = max(
            [int(np.max(pred_track.frame_ids)) + 1]
            + [
                int(np.max(target_track.frame_ids)) + 1
                for target_track, _ in target_tracks
            ]
        )
        pred_joints, pred_valid = dense_joints(pred_track, frames)
        pred_motion = camera_motion135(pred_track)
        pred_world_motion = world_motion135(pred_track)
        tracking_bbox = load_tracking_bbox(
            official_run_dir,
            sequence,
            assignment["prediction_track_id"],
            frames,
        )
        dense_targets = [
            (
                target_track,
                record,
                *dense_joints(target_track, frames),
                camera_motion135(target_track),
                world_motion135(target_track),
            )
            for target_track, record in target_tracks
        ]
        matched = next(
            item
            for item in dense_targets
            if item[0].track_id == assignment["target_track_id"]
        )
        target_bbox = target_poses2d_bbox(matched[0], frames)
        errors = metric_error(pred_joints, matched[2], pred_valid, matched[3])

        pred_bytes, pred_descriptor = encode_positions(pred_joints)
        payload = bytearray()
        pred_descriptor["position_offset"] = len(payload)
        payload.extend(pred_bytes)
        target_descriptors = []
        for target_track, record, target_joints, _, _, _ in dense_targets:
            target_bytes, descriptor = encode_positions(target_joints)
            descriptor.update(
                {
                    "track_id": target_track.track_id,
                    "status": record.get("status", "matched_prediction"),
                    "matched": target_track.track_id
                    == assignment["target_track_id"],
                }
            )
            descriptor["position_offset"] = len(payload)
            payload.extend(target_bytes)
            target_descriptors.append(descriptor)
        pred_mask_offset = len(payload)
        payload.extend(pred_valid.astype(np.uint8).tobytes())
        for descriptor, (_, _, _, target_valid, _, _) in zip(
            target_descriptors,
            dense_targets,
        ):
            descriptor["valid_offset"] = len(payload)
            payload.extend(target_valid.astype(np.uint8).tobytes())
        while len(payload) % 4:
            payload.append(0)
        error_offset = len(payload)
        payload.extend(errors.tobytes())
        pred_motion_bytes, pred_motion_descriptor = encode_motion135(
            pred_motion,
            stride=1,
        )
        pred_motion_descriptor["translation_offset"] = len(payload)
        pred_motion_descriptor["rotation_offset"] = (
            len(payload) + pred_motion_descriptor["translation_count"] * 2
        )
        payload.extend(pred_motion_bytes)
        for descriptor, (_, _, _, _, target_motion, target_world_motion) in zip(
            target_descriptors,
            dense_targets,
        ):
            motion_bytes, motion_descriptor = encode_motion135(
                target_motion,
                stride=1,
            )
            motion_descriptor["translation_offset"] = len(payload)
            motion_descriptor["rotation_offset"] = (
                len(payload) + motion_descriptor["translation_count"] * 2
            )
            descriptor["motion"] = motion_descriptor
            payload.extend(motion_bytes)
            world_bytes, world_descriptor = encode_motion135(
                target_world_motion,
                stride=1,
            )
            world_descriptor["translation_offset"] = len(payload)
            world_descriptor["rotation_offset"] = (
                len(payload) + world_descriptor["translation_count"] * 2
            )
            descriptor["world_motion"] = world_descriptor
            payload.extend(world_bytes)
        pred_world_bytes, pred_world_descriptor = encode_motion135(
            pred_world_motion,
            stride=1,
        )
        pred_world_descriptor["translation_offset"] = len(payload)
        pred_world_descriptor["rotation_offset"] = (
            len(payload) + pred_world_descriptor["translation_count"] * 2
        )
        payload.extend(pred_world_bytes)
        while len(payload) % 4:
            payload.append(0)
        tracking_bbox_offset = len(payload)
        payload.extend(tracking_bbox.astype("<f4", copy=False).tobytes())
        target_bbox_offset = len(payload)
        payload.extend(target_bbox.astype("<f4", copy=False).tobytes())
        asset = f"assets/{sequence}.joints"
        asset_payload = bytes(payload)
        (output / asset).write_bytes(asset_payload)

        case_metrics = matched[1]["metrics"]
        case_payload = {
                "case_id": sequence,
                "frames": frames,
                "fps": float(prediction.output_fps),
                "asset": asset,
                "asset_bytes": len(asset_payload),
                "asset_sha256": hashlib.sha256(asset_payload).hexdigest(),
                "prediction": pred_descriptor,
                "prediction_motion": pred_motion_descriptor,
                "prediction_world_motion": pred_world_descriptor,
                "tracking_bbox_offset": tracking_bbox_offset,
                "tracking_bbox_count": int(tracking_bbox.size),
                "tracking_bbox_format": "cx_cy_size_pixels",
                "target_bbox_offset": target_bbox_offset,
                "target_bbox_count": int(target_bbox.size),
                "target_bbox_format": "xyxy_pixels_from_official_poses2d",
                "targets": target_descriptors,
                "prediction_valid_offset": pred_mask_offset,
                "error_offset": error_offset,
                "error_count": frames,
                "association": assignment,
                "coverage_percent": matched[1]["coverage_percent"],
                "evaluated_frames": matched[1]["evaluated_frames"],
                "target_valid_frames": matched[1]["target_valid_frames"],
                "metrics": {
                    "pa_mpjpe_mm": case_metrics.get("pa_mpjpe_mm"),
                    "mpjpe_mm": case_metrics.get("mpjpe_mm"),
                    "accel_mps2": case_metrics.get("accel_mps2"),
                },
            }
        for key in (
            "video",
            "video_frames",
            "video_fps",
            "video_width",
            "video_height",
            "bbox_coordinate_width",
            "bbox_coordinate_height",
            "video_sha256",
            "video_distribution",
        ):
            if key in existing_cases.get(sequence, {}):
                case_payload[key] = existing_cases[sequence][key]
        cases.append(case_payload)

    manifest = {
        "schema_version": 1,
        "title": "GEM-SMPL · 3DPW Test",
        "task": "monocular_motion_capture",
        "protocol": metrics["protocol"],
        "population_sequences": len(cases),
        "population_tracks": metrics["population_tracks"],
        "selection": "focused_preview" if selected_cases else "full_population",
        "coverage_percent": metrics["coverage_percent"],
        "metrics": metrics["metrics"],
        "source_model": cases and "NVlabs/GEM-SMPL",
        "checkpoint_sha256": (
            load_monocular_capture_result(
                prediction_dir / f"{cases[0]['case_id']}.motius.npz"
            ).checkpoint_sha256
            if cases
            else None
        ),
        "body_model_url": "assets/smpl_model/",
        "mesh_replay": "licensed_neutral_smpl_gpu_skinning",
        "parents": list(SMPL24_PARENTS),
        "coordinate_transform": "camera_opencv_to_three_y_up_z_back",
        "distribution": (
            "local_only_licensed_3dpw"
            if cases and all("video" in case for case in cases)
            else "local_only_licensed_derived_3dpw"
        ),
        "contains_licensed_input_video": bool(
            cases and all("video" in case for case in cases)
        ),
        "cases": cases,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "cases": len(cases),
                "assets_bytes": sum(path.stat().st_size for path in assets.iterdir()),
            }
        )
    )


if __name__ == "__main__":
    main()
