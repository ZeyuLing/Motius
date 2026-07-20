#!/usr/bin/env python3
"""Build the full AIST++ GT feature pool used by Bailando FID/Diversity."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import pickle
from pathlib import Path
import sys
import zipfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.evaluation.metrics.dance_features import (  # noqa: E402
    extract_geometric_features,
    extract_kinetic_features,
)
from motius.evaluation.music_to_dance import root_anchor_motion  # noqa: E402
from motius.motion.representation.aistpp import aistpp_smpl24_fk  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--json-root",
        type=Path,
        action="append",
        help="Directory containing preprocessed AIST++ JSON files; repeatable.",
    )
    source.add_argument(
        "--data-zip",
        type=Path,
        help="Official Bailando data.zip containing train and test JSON files.",
    )
    source.add_argument(
        "--motions-zip",
        type=Path,
        help="Complete official AIST++ motions.zip used by the paper reference pool.",
    )
    source.add_argument(
        "--motions-root",
        type=Path,
        help="Directory containing the complete official AIST++ motion PKLs.",
    )
    parser.add_argument(
        "--smpl-skeleton",
        type=Path,
        help="Calibrated SMPL-24 offsets for --motions-zip.",
    )
    parser.add_argument(
        "--ignore-list",
        type=Path,
        help="Official AIST++ ignore_list.txt for --motions-zip.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional frame cap. The official full AIST++ reference pool is uncropped.",
    )
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


_ARCHIVE: zipfile.ZipFile | None = None
_MAX_FRAMES: int | None = None
_REST_OFFSETS: np.ndarray | None = None


def _init_worker(
    path: str | None,
    max_frames: int | None,
    skeleton_path: str | None = None,
) -> None:
    global _ARCHIVE, _MAX_FRAMES, _REST_OFFSETS
    _ARCHIVE = zipfile.ZipFile(path) if path is not None else None
    _MAX_FRAMES = int(max_frames) if max_frames is not None else None
    if skeleton_path is None:
        _REST_OFFSETS = None
    else:
        with np.load(skeleton_path, allow_pickle=False) as payload:
            _REST_OFFSETS = np.asarray(payload["rest_offsets"], dtype=np.float32)


def _features_from_payload(name: str, payload: dict):
    joints = np.asarray(payload["dance_array"], dtype=np.float32).reshape(-1, 24, 3)
    if _MAX_FRAMES is not None:
        joints = joints[:_MAX_FRAMES]
    joints = root_anchor_motion(joints)
    return (
        name,
        extract_kinetic_features(joints),
        extract_geometric_features(joints),
        len(joints),
    )


def _features_from_path(path_string: str):
    path = Path(path_string)
    with path.open("r", encoding="utf-8") as handle:
        return _features_from_payload(path.stem, json.load(handle))


def _features_from_zip(name: str):
    if _ARCHIVE is None:
        raise RuntimeError("Zip worker was not initialized")
    with _ARCHIVE.open(name) as handle:
        payload = json.load(io.TextIOWrapper(handle, encoding="utf-8"))
    return _features_from_payload(Path(name).stem, payload)


def _features_from_motion_zip(name: str):
    if _ARCHIVE is None or _REST_OFFSETS is None:
        raise RuntimeError("AIST++ motion worker was not initialized")
    with _ARCHIVE.open(name) as handle:
        try:
            payload = pickle.load(handle, encoding="latin1")
        except TypeError:
            payload = pickle.load(handle)
    return _features_from_motion_payload(Path(name).stem, payload)


def _features_from_motion_path(path_string: str):
    path = Path(path_string)
    with path.open("rb") as handle:
        try:
            payload = pickle.load(handle, encoding="latin1")
        except TypeError:
            payload = pickle.load(handle)
    return _features_from_motion_payload(path.stem, payload)


def _features_from_motion_payload(name: str, payload: dict):
    if _REST_OFFSETS is None:
        raise RuntimeError("AIST++ motion worker was not initialized")
    poses = np.asarray(payload["smpl_poses"])
    translation = np.asarray(payload["smpl_trans"])
    if _MAX_FRAMES is not None:
        poses = poses[:_MAX_FRAMES]
        translation = translation[:_MAX_FRAMES]
    joints = aistpp_smpl24_fk(
        poses,
        translation,
        payload["smpl_scaling"],
        _REST_OFFSETS,
    )
    return _features_from_payload(name, {"dance_array": joints})


def build_jobs(args: argparse.Namespace):
    if args.json_root:
        candidates = sorted(
            str(path) for root in args.json_root for path in root.glob("*.json")
        )
        skipped = [path for path in candidates if Path(path).stat().st_size == 0]
        jobs = [path for path in candidates if Path(path).stat().st_size > 0]
        return (
            jobs[: args.max_samples],
            _features_from_path,
            _init_worker,
            (None, args.max_frames),
            skipped,
        )

    if args.motions_zip is not None or args.motions_root is not None:
        if args.smpl_skeleton is None or args.ignore_list is None:
            raise ValueError(
                "AIST++ motion sources require --smpl-skeleton and --ignore-list"
            )
        ignored = {
            Path(line.strip()).stem
            for line in args.ignore_list.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        if args.motions_zip is not None:
            with zipfile.ZipFile(args.motions_zip) as archive:
                candidates = sorted(
                    item.filename
                    for item in archive.infolist()
                    if item.filename.startswith("motions/")
                    and item.filename.endswith(".pkl")
                )
            worker = _features_from_motion_zip
            archive_path = str(args.motions_zip)
        else:
            candidates = sorted(str(path) for path in args.motions_root.glob("*.pkl"))
            worker = _features_from_motion_path
            archive_path = None
        skipped = [name for name in candidates if Path(name).stem in ignored]
        jobs = [name for name in candidates if Path(name).stem not in ignored]
        if args.max_samples is not None:
            jobs = jobs[: args.max_samples]
        return (
            jobs,
            worker,
            _init_worker,
            (archive_path, args.max_frames, str(args.smpl_skeleton)),
            skipped,
        )

    prefixes = ("data/aistpp_train_wav/", "data/aistpp_test_full_wav/")
    with zipfile.ZipFile(args.data_zip) as archive:
        candidates = sorted(
            [
                item
                for item in archive.infolist()
                if item.filename.endswith(".json")
                and item.filename.startswith(prefixes)
            ],
            key=lambda item: item.filename,
        )
    skipped = [item.filename for item in candidates if item.file_size == 0]
    jobs = [item.filename for item in candidates if item.file_size > 0]
    if args.max_samples is not None:
        jobs = jobs[: args.max_samples]
    return (
        jobs,
        _features_from_zip,
        _init_worker,
        (str(args.data_zip), args.max_frames),
        skipped,
    )


def main() -> None:
    args = parse_args()
    names = []
    kinetic = []
    geometric = []
    jobs, worker, initializer, initargs, skipped = build_jobs(args)
    for name in skipped:
        print(f"skip source entry: {name}", flush=True)
    workers = max(1, min(int(args.workers), len(jobs)))
    with mp.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        rows = pool.imap_unordered(worker, jobs, chunksize=1)
        for index, (name, kinetic_row, geometric_row, frames) in enumerate(
            rows, start=1
        ):
            names.append(name)
            kinetic.append(kinetic_row)
            geometric.append(geometric_row)
            print(f"[{index}/{len(jobs)}] {name} frames={frames}", flush=True)

    order = np.argsort(names)
    names = [names[index] for index in order]
    kinetic = [kinetic[index] for index in order]
    geometric = [geometric[index] for index in order]

    if len(names) < 2:
        raise ValueError("At least two AIST++ motions are required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        names=np.asarray(names),
        kinetic=np.stack(kinetic),
        geometric=np.stack(geometric),
        skipped=np.asarray(skipped, dtype=str),
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "num_samples": len(names),
                "max_frames": args.max_frames,
                "kinetic_shape": list(np.shape(kinetic)),
                "geometric_shape": list(np.shape(geometric)),
                "skipped": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
