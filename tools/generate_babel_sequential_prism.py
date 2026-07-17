#!/usr/bin/env python3
"""Generate PRISM BABEL compositions with a fixed, reproducible protocol."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.pipelines.prism import PRISMPipeline


SUPPORTED_PROTOCOLS = {
    "babel-official-val-shortmerge30-llm-joints66-actiongroups-v3",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model", default="kt")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--transformer-dtype", default="bf16")
    parser.add_argument("--text-dtype", default="bf16")
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--pad-to-frames", type=int, default=360)
    parser.add_argument("--ar-condition-frames", type=int, default=5)
    parser.add_argument(
        "--kafs-mode",
        choices=("none", "depth_driven", "uniform", "random"),
        default=None,
        help="Override the artifact's default KAFS mode.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--ids", default="", help="Comma-separated case ids.")
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("require 0 <= shard-index < num-shards")
    return args


def _save_smpl(path: Path, smpl: dict, *, metadata: dict) -> None:
    keep = {
        key: np.asarray(value)
        for key, value in smpl.items()
        if not key.startswith("_")
        and key
        in {
            "trans",
            "transl",
            "poses",
            "global_orient",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "left_hand_pose",
            "right_hand_pose",
            "gender",
            "betas",
            "expression",
            "mocap_framerate",
        }
    }
    for key, value in smpl.items():
        if key.startswith("_prism_"):
            keep[key] = np.asarray(value)
    keep.update({key: np.asarray(value) for key, value in metadata.items()})
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **keep)


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("protocol") not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unsupported protocol {manifest.get('protocol')!r}.")

    selected_ids = {value.strip() for value in args.ids.split(",") if value.strip()}
    cases = [
        (index, case)
        for index, case in enumerate(manifest.get("cases", []))
        if index % args.num_shards == args.shard_index
        and (not selected_ids or str(case["case_id"]) in selected_ids)
    ]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = PRISMPipeline.from_pretrained(
        args.model,
        bundle_kwargs={
            "device": args.device,
            "transformer_dtype": args.transformer_dtype,
            "text_dtype": args.text_dtype,
        },
        device=args.device,
    )
    effective_kafs_mode = args.kafs_mode or str(
        pipe.bundle.artifact_config.get("default_kafs_mode", "none")
    )

    generated = skipped = failed = 0
    failures: list[dict[str, str]] = []
    started = time.time()
    for case_index, case in cases:
        case_id = str(case["case_id"])
        output_path = output_dir / f"{case_id}.npz"
        if output_path.is_file() and not args.overwrite:
            skipped += 1
            continue
        captions = [str(segment["caption"]) for segment in case["segments"]]
        lengths = [
            int(segment["end_frame"]) - int(segment["start_frame"])
            for segment in case["segments"]
        ]
        try:
            result = pipe.sequential_generation(
                captions,
                segment_frames=lengths,
                pad_to_frames=args.pad_to_frames,
                ar_condition_frames=args.ar_condition_frames,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                kafs_mode=effective_kafs_mode,
                canonicalize=False,
                use_blend=False,
                seed=args.seed + case_index,
            )
            smpl = result["smpl"]
            expected = int(case["total_frames"])
            actual = int(np.asarray(smpl["transl"]).shape[0])
            if actual != expected:
                raise RuntimeError(
                    f"PRISM returned {actual} frames for {case_id}, expected {expected}"
                )
            _save_smpl(
                output_path,
                smpl,
                metadata={
                    "motius_model": str(args.model),
                    "motius_variant": str(result["variant"]),
                    "motius_seed": args.seed + case_index,
                    "motius_guidance_scale": args.guidance_scale,
                    "motius_num_inference_steps": args.num_inference_steps,
                    "motius_kafs_mode": effective_kafs_mode,
                    "motius_segment_lengths": np.asarray(lengths, dtype=np.int32),
                    "motius_pad_to_frames": args.pad_to_frames,
                },
            )
            generated += 1
            print(
                f"[{generated + skipped + failed}/{len(cases)}] generated {case_id}",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            failures.append({"case_id": case_id, "error": str(exc)})
            print(f"[failed] {case_id}: {exc}", flush=True)

    summary = {
        "protocol": manifest["protocol"],
        "model": args.model,
        "variant": pipe.bundle.variant,
        "artifact_dir": str(pipe.bundle.artifact_dir),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "kafs_mode": effective_kafs_mode,
        "pad_to_frames": args.pad_to_frames,
        "ar_condition_frames": args.ar_condition_frames,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "generated": generated,
        "skipped": skipped,
        "failed": failed,
        "elapsed_seconds": time.time() - started,
        "failures": failures[:100],
    }
    (output_dir / f"run_shard_{args.shard_index:03d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
