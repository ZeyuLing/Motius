#!/usr/bin/env python3
"""Encode the complete AIST++ GT pool with the Motius joint evaluator."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import pickle
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.evaluators.tmr import TMRTextMotionEvaluator  # noqa: E402
from motius.evaluation.music_to_dance import prepare_utmr_dance_motion  # noqa: E402
from motius.motion.representation.aistpp import aistpp_smpl24_fk  # noqa: E402


_REST_OFFSETS: np.ndarray | None = None
_MAX_SECONDS = 20.0
_TARGET_FPS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions-root", type=Path, required=True)
    parser.add_argument("--smpl-skeleton", type=Path, required=True)
    parser.add_argument("--ignore-list", type=Path, required=True)
    parser.add_argument(
        "--evaluator",
        default="ZeyuLing/motius-evaluator-universal-smplh-joints66",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--motion-cache",
        type=Path,
        help="Reusable preprocessed joints66 pool; defaults beside --output.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--max-seconds", type=float, default=20.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _init_worker(skeleton: str, max_seconds: float, target_fps: float) -> None:
    global _REST_OFFSETS, _MAX_SECONDS, _TARGET_FPS
    with np.load(skeleton, allow_pickle=False) as payload:
        _REST_OFFSETS = np.asarray(payload["rest_offsets"], dtype=np.float32)
    _MAX_SECONDS = float(max_seconds)
    _TARGET_FPS = float(target_fps)


def _load_motion(path_string: str) -> tuple[str, np.ndarray]:
    if _REST_OFFSETS is None:
        raise RuntimeError("AIST++ worker was not initialized")
    path = Path(path_string)
    with path.open("rb") as handle:
        try:
            payload = pickle.load(handle, encoding="latin1")
        except TypeError:
            payload = pickle.load(handle)
    joints = aistpp_smpl24_fk(
        payload["smpl_poses"],
        payload["smpl_trans"],
        payload["smpl_scaling"],
        _REST_OFFSETS,
    )
    return (
        path.stem,
        prepare_utmr_dance_motion(
            joints,
            source_fps=60.0,
            target_fps=_TARGET_FPS,
            max_seconds=_MAX_SECONDS,
        ),
    )


def main() -> None:
    args = parse_args()
    ignored = {
        Path(line.strip()).stem
        for line in args.ignore_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    paths = sorted(
        path for path in args.motions_root.glob("*.pkl") if path.stem not in ignored
    )
    if args.max_samples is not None:
        paths = paths[: args.max_samples]
    if len(paths) < 2:
        raise ValueError("At least two valid AIST++ motion PKLs are required")

    motion_cache = args.motion_cache or args.output.with_name(
        f"{args.output.stem}_motions.npz"
    )
    if motion_cache.is_file():
        with np.load(motion_cache, allow_pickle=False) as payload:
            names = payload["names"].astype(str).tolist()
            lengths = np.asarray(payload["lengths"], dtype=np.int64)
            padded = np.asarray(payload["motions"], dtype=np.float32)
        motions = [padded[index, :length] for index, length in enumerate(lengths)]
        print(f"loaded motion cache: {motion_cache} ({len(motions)} clips)", flush=True)
    else:
        workers = max(1, min(int(args.workers), len(paths)))
        names: list[str] = []
        motions: list[np.ndarray] = []
        with mp.Pool(
            workers,
            initializer=_init_worker,
            initargs=(str(args.smpl_skeleton), args.max_seconds, args.target_fps),
        ) as pool:
            for index, (name, motion) in enumerate(
                pool.imap(_load_motion, map(str, paths), chunksize=1), start=1
            ):
                names.append(name)
                motions.append(motion)
                if index % 25 == 0 or index == len(paths):
                    print(f"materialized {index}/{len(paths)}", flush=True)
        lengths = np.asarray([len(motion) for motion in motions], dtype=np.int32)
        padded = np.zeros(
            (len(motions), int(lengths.max()), motions[0].shape[1]), dtype=np.float32
        )
        for index, motion in enumerate(motions):
            padded[index, : len(motion)] = motion
        motion_cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            motion_cache,
            names=np.asarray(names),
            motions=padded,
            lengths=lengths,
        )
        print(f"saved motion cache: {motion_cache}", flush=True)

    evaluator = TMRTextMotionEvaluator.from_pretrained(
        args.evaluator,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    embeddings = evaluator.encode_motions(motions).astype(np.float32)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, embeddings)
    audit_path = args.output.with_suffix(".json")
    audit = {
        "schema_version": 1,
        "dataset": "AIST++ v1 full valid motion pool",
        "evaluator": args.evaluator,
        "motion_representation": "canonical SMPL-22 joints66",
        "source_fps": 60.0,
        "target_fps": args.target_fps,
        "max_seconds": args.max_seconds,
        "num_samples": len(names),
        "embedding_shape": list(embeddings.shape),
        "names": names,
    }
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**audit, "names": f"{len(names)} entries"}, indent=2))


if __name__ == "__main__":
    main()
