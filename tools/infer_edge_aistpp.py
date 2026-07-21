#!/usr/bin/env python3
"""Run the Motius EDGE pipeline on one or more music tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from motius.models.edge.audio import extract_edge_jukebox_features
from motius.pipelines.edge import EDGEPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--audio", type=Path, nargs="+")
    parser.add_argument("--music-features", type=Path, nargs="+")
    parser.add_argument(
        "--case-manifest",
        type=Path,
        help="AIST++ gallery manifest whose case ids, audio, and frame counts are authoritative.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        help="Base directory for manifest-relative audio paths.",
    )
    parser.add_argument(
        "--feature-root",
        type=Path,
        help="Shared directory of one Jukebox feature array per music id.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--guidance-weight", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument("--jukebox-fp16", action="store_true")
    parser.add_argument("--jukebox-cache-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--extract-features-only", action="store_true")
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    modes = sum(bool(value) for value in (args.audio, args.music_features, args.case_manifest))
    if modes != 1:
        parser.error("Provide exactly one of --audio, --music-features, or --case-manifest")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if (args.extract_features_only or args.finalize_only) and not args.case_manifest:
        parser.error("--extract-features-only and --finalize-only require --case-manifest")
    if args.extract_features_only and args.finalize_only:
        parser.error("--extract-features-only and --finalize-only are mutually exclusive")
    if args.case_manifest and args.feature_root is None:
        parser.error("--case-manifest requires --feature-root")
    if args.extract_features_only and args.audio_root is None:
        parser.error("--extract-features-only requires --audio-root")
    return args


def _load_cases(path: Path) -> list[dict]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No cases found in {path}")
    records = []
    for index, item in enumerate(cases):
        case_id = str(item.get("case_id") or item.get("sample_id") or "")
        audio = item.get("audio")
        motion_descriptor = None
        if isinstance(item.get("motions"), dict) and item["motions"]:
            motion_descriptor = item["motions"].get("gt") or next(
                iter(item["motions"].values())
            )
        frames = int(
            item.get("frames")
            or item.get("display_frames")
            or (motion_descriptor or {}).get("display_frames")
            or (motion_descriptor or {}).get("frames")
            or 0
        )
        if not case_id or not audio or frames < 1:
            raise ValueError(f"Invalid case manifest row {index}: {item}")
        records.append(
            {
                "index": index,
                "case_id": case_id,
                "audio": str(audio),
                "music_id": Path(str(audio)).stem,
                "frames": frames,
                "duration": float(item.get("audio_end_seconds") or frames / 30.0),
            }
        )
    return records


def _write_manifest(
    path: Path,
    *,
    checkpoint: str,
    samples: list[str],
    elapsed_seconds: float,
    seed: int,
) -> None:
    payload = {
        "schema_version": 1,
        "task": "music_to_dance",
        "method": "EDGE",
        "dataset": "AIST++ official crossmodal 40-case package",
        "checkpoint": checkpoint,
        "representation": "EDGE-151 -> AIST++ SMPL-24 joints + motion135",
        "motion_fps": 30.0,
        "music_features": "Jukebox layer 66, 4800-D at 30 fps",
        "seed": seed,
        "num_samples": len(samples),
        "samples": samples,
        "elapsed_seconds": elapsed_seconds,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _extract_manifest_features(args, records: list[dict]) -> None:
    feature_root = args.feature_root.expanduser().resolve()
    feature_root.mkdir(parents=True, exist_ok=True)
    unique = {}
    for record in records:
        unique.setdefault(record["music_id"], record)
        unique[record["music_id"]]["duration"] = max(
            unique[record["music_id"]]["duration"], record["duration"]
        )
    for index, (music_id, record) in enumerate(sorted(unique.items())):
        destination = feature_root / f"{music_id}.npy"
        if destination.is_file() and not args.overwrite:
            print(f"[{index + 1}/{len(unique)}] skip feature {music_id}", flush=True)
            continue
        audio = Path(record["audio"])
        if not audio.is_absolute():
            audio = args.audio_root.expanduser().resolve() / audio
        features = extract_edge_jukebox_features(
            audio,
            max_seconds=record["duration"],
            fp16=args.jukebox_fp16,
            cache_dir=args.jukebox_cache_dir,
        )
        np.save(destination, features)
        print(
            f"[{index + 1}/{len(unique)}] feature {music_id} {features.shape}",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = _load_cases(args.case_manifest) if args.case_manifest else None
    if args.extract_features_only:
        _extract_manifest_features(args, records)
        return
    if args.finalize_only:
        missing = [
            record["case_id"]
            for record in records
            if not (args.output / f"{record['case_id']}.npz").is_file()
        ]
        if missing:
            raise FileNotFoundError(f"Missing {len(missing)} EDGE outputs: {missing}")
        _write_manifest(
            args.output / "manifest.json",
            checkpoint=args.checkpoint,
            samples=[record["case_id"] for record in records],
            elapsed_seconds=0.0,
            seed=args.seed,
        )
        print(f"finalized {len(records)} EDGE outputs", flush=True)
        return
    feature_dir = (
        args.feature_root.expanduser().resolve()
        if args.feature_root is not None
        else args.output / "music_features"
    )
    feature_dir.mkdir(exist_ok=True)
    pipeline = EDGEPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": Path(args.checkpoint).exists()},
        device=args.device,
    )
    if records is not None:
        inputs = [
            record
            for record in records
            if record["index"] % args.num_shards == args.shard_index
        ]
    else:
        inputs = args.audio or args.music_features
    completed = []
    started = time.time()
    for local_index, item in enumerate(inputs):
        if records is not None:
            record = item
            name = record["case_id"]
            path = feature_dir / f"{record['music_id']}.npy"
            case_seed = args.seed + record["index"]
            max_frames = record["frames"]
        else:
            record = None
            path = item.expanduser().resolve()
            name = path.stem
            case_seed = args.seed + local_index
            max_frames = args.max_frames
        output_path = args.output / f"{name}.npz"
        if output_path.is_file() and not args.overwrite:
            completed.append(name)
            print(f"[{local_index + 1}/{len(inputs)}] skip {name}", flush=True)
            continue
        case_started = time.time()
        feature_path = feature_dir / f"{name}.npy"
        if args.audio:
            if feature_path.is_file() and not args.overwrite:
                features = np.load(feature_path, allow_pickle=False)
            else:
                features = extract_edge_jukebox_features(
                    path,
                    max_seconds=args.max_seconds,
                    fp16=args.jukebox_fp16,
                    cache_dir=args.jukebox_cache_dir,
                )
                np.save(feature_path, features)
        else:
            features = np.load(path, allow_pickle=False)
        result = pipeline(
            music_features=features,
            max_frames=max_frames,
            seed=case_seed,
            sampling_steps=args.sampling_steps,
            guidance_weight=args.guidance_weight,
            eta=args.eta,
        )
        np.savez_compressed(
            output_path,
            joints=result.joints,
            edge_motion=result.edge_motion,
            motion_135=result.motion_135,
            contacts=result.contacts,
            fps=np.float32(result.fps),
            audio=np.asarray(
                str(record["audio"] if record is not None else path)
                if (args.audio or record is not None)
                else ""
            ),
            music_features=np.asarray(str(feature_path if args.audio else path)),
            seed=np.int64(case_seed),
        )
        completed.append(name)
        print(
            f"[{local_index + 1}/{len(inputs)}] {name} frames={len(result.joints)} "
            f"seconds={time.time() - case_started:.2f}",
            flush=True,
        )
    manifest_name = (
        f"manifest_shard_{args.shard_index:02d}.json"
        if args.num_shards > 1
        else "manifest.json"
    )
    _write_manifest(
        args.output / manifest_name,
        checkpoint=args.checkpoint,
        samples=completed,
        elapsed_seconds=time.time() - started,
        seed=args.seed,
    )
    print(f"wrote {manifest_name}: {len(completed)} samples", flush=True)


if __name__ == "__main__":
    main()
