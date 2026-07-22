#!/usr/bin/env python3
"""Run UniMuMo music-to-motion on the shared AIST++ case package."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import librosa
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.pipelines.unimumo import UniMuMoPipeline


DEFAULT_AUDIO_BASE_URL = (
    "https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/"
    "cases/"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--case-manifest", type=Path, required=True)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--audio-base-url", default=DEFAULT_AUDIO_BASE_URL)
    parser.add_argument("--audio-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seconds", type=float, default=10.0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if not 0 < args.max_seconds <= 10:
        parser.error("--max-seconds must be in (0, 10]")
    return args


def _load_cases(path: Path, max_samples: int | None) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"No AIST++ cases found in {path}")
    return cases if max_samples is None else cases[:max_samples]


def _cached_audio(args: argparse.Namespace, relative_path: str) -> Path:
    relative = Path(relative_path)
    if args.audio_root is not None:
        source = args.audio_root / relative
        if source.is_file():
            return source
    destination = args.audio_cache / relative.name
    if not destination.is_file():
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = args.audio_base_url.rstrip("/") + "/" + relative.as_posix()
        urllib.request.urlretrieve(url, destination)
    return destination


def smpl22_to_smpl24(joints: np.ndarray, hand_scale: float = 0.35) -> np.ndarray:
    """Append SMPL hand joints by continuing each elbow-to-wrist segment."""

    values = np.asarray(joints, dtype=np.float32)
    if values.ndim != 3 or values.shape[1:] != (22, 3):
        raise ValueError(f"Expected SMPL-22 joints, got {values.shape}")
    output = np.empty((len(values), 24, 3), dtype=np.float32)
    output[:, :22] = values
    output[:, 22] = values[:, 20] + hand_scale * (values[:, 20] - values[:, 18])
    output[:, 23] = values[:, 21] + hand_scale * (values[:, 21] - values[:, 19])
    return output


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = _load_cases(args.case_manifest, args.max_samples)
    pipeline = UniMuMoPipeline.from_pretrained(
        args.checkpoint,
        bundle_kwargs={"local_files_only": args.local_files_only},
        device=args.device,
    )
    completed = []
    started = time.time()
    for index, case in enumerate(cases):
        if index % args.num_shards != args.shard_index:
            continue
        case_id = str(case.get("case_id") or case.get("sample_id"))
        destination = args.output / f"{case_id}.npz"
        if destination.is_file() and not args.overwrite:
            completed.append(case_id)
            print(f"skip {case_id}", flush=True)
            continue
        audio_path = _cached_audio(args, str(case["audio"]))
        waveform, _ = librosa.load(audio_path, sr=32_000, mono=True)
        start = max(0.0, float(case.get("audio_start_seconds") or 0.0))
        available = max(0.0, len(waveform) / 32_000 - start)
        requested = float(case.get("audio_end_seconds") or available) - start
        duration = min(args.max_seconds, available, requested)
        if duration <= 0:
            raise ValueError(f"Empty audio interval for {case_id}")
        first = int(round(start * 32_000))
        last = first + int(duration * 32_000)
        references = [str(value) for value in case.get("references") or ()]
        style = references[0].strip().lower() if references else ""
        motion_prompt = f"The style of the dance is {style}." if style else ""
        case_started = time.time()
        result = pipeline.infer_music_to_motion(
            waveform[first:last],
            sample_rate=32_000,
            motion_prompt=motion_prompt,
            guidance_scale=args.guidance_scale,
            temperature=args.temperature,
            top_k=args.top_k,
            seed=args.seed + index,
        )
        joints24 = smpl22_to_smpl24(result.joints)
        np.savez_compressed(
            destination,
            joints=joints24,
            smpl22_joints=result.joints,
            humanml3d_263=result.motion,
            motion_codes=result.motion_codes,
            fps=np.float32(result.motion_fps),
            audio_path=np.asarray(str(audio_path)),
            audio_start_seconds=np.float32(start),
            audio_duration_seconds=np.float32(duration),
            motion_prompt=np.asarray(motion_prompt),
            seed=np.int64(args.seed + index),
        )
        completed.append(case_id)
        print(
            f"[{index + 1}/{len(cases)}] {case_id} "
            f"frames={len(joints24)} seconds={time.time() - case_started:.2f}",
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "task": "music_to_dance",
        "method": "UniMuMo",
        "dataset": "AIST++ shared crossmodal case package",
        "checkpoint": args.checkpoint,
        "representation": "HumanML3D-263 at 60 fps; SMPL-22 and SMPL-24 bridge",
        "conditioning": "music plus AIST++ genre prompt (official protocol)",
        "guidance_scale": args.guidance_scale,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
        "num_samples": len(completed),
        "samples": completed,
        "elapsed_seconds": time.time() - started,
    }
    (args.output / f"manifest_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
