#!/usr/bin/env python3
"""Evaluate a Motius motion tracker in one registered physical backend."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import glob
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius import Pipeline
from motius.simulators.reference import load_g1_reference


PROTOCOLS = {
    "mujoco": "mujoco-g1-reference-tracking-50hz-v1",
    "isaaclab": "isaaclab-2.3.2-g1-reference-tracking-50hz-v1",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local or Hugging Face artifact")
    parser.add_argument(
        "--simulator",
        choices=sorted(PROTOCOLS),
        required=True,
        help="Physical engine; results from different engines are never merged",
    )
    parser.add_argument(
        "--reference",
        action="append",
        required=True,
        help="Reference file or glob; repeat to add sources",
    )
    parser.add_argument("--output", required=True, help="Directory under outputs/")
    parser.add_argument("--source-fps", type=float)
    parser.add_argument("--control-hz", type=float, default=50.0)
    parser.add_argument("--window-steps", type=int, default=1000)
    parser.add_argument("--minimum-remainder-steps", type=int, default=250)
    parser.add_argument(
        "--windows",
        choices=("first", "all"),
        default="first",
        help="Use one fixed-start episode per motion or all deterministic windows",
    )
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _expand_references(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(item) for item in glob.glob(pattern)]
        if not matches and Path(pattern).is_file():
            matches = [Path(pattern)]
        paths.extend(matches)
    unique = sorted({path.expanduser().resolve() for path in paths})
    if not unique:
        raise FileNotFoundError("No reference files matched --reference.")
    return unique


def _weighted_aggregate(cases: list[dict[str, Any]]) -> dict[str, float]:
    metric_names = sorted(
        {
            key
            for case in cases
            for key, value in case["metrics"].items()
            if isinstance(value, (int, float))
            and key not in {"completed_steps", "total_steps", "success_rate", "survival_rate"}
        }
    )
    weights = np.asarray(
        [max(int(case["metrics"]["completed_steps"]), 1) for case in cases],
        dtype=np.float64,
    )
    aggregate = {
        name: float(
            np.average(
                np.asarray([case["metrics"][name] for case in cases], dtype=np.float64),
                weights=weights,
            )
        )
        for name in metric_names
        if all(name in case["metrics"] for case in cases)
    }
    aggregate.update(
        {
            "success_rate": float(
                np.mean([case["metrics"]["success_rate"] for case in cases])
            ),
            "survival_rate": float(
                sum(case["metrics"]["completed_steps"] for case in cases)
                / max(sum(case["metrics"]["total_steps"] for case in cases), 1)
            ),
            "cases": len(cases),
            "completed_steps": int(
                sum(case["metrics"]["completed_steps"] for case in cases)
            ),
            "total_steps": int(sum(case["metrics"]["total_steps"] for case in cases)),
        }
    )
    return aggregate


def main() -> None:
    args = _arguments()
    output_root = Path(args.output).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline.from_pretrained(
        args.model,
        bundle_kwargs={"local_files_only": args.local_files_only},
    )

    cases: list[dict[str, Any]] = []
    stop = False
    for reference_path in _expand_references(args.reference):
        reference = load_g1_reference(
            reference_path,
            source_fps=args.source_fps,
            target_fps=args.control_hz,
        )
        windows = reference.iter_windows(
            args.window_steps,
            minimum_remainder_steps=args.minimum_remainder_steps,
        )
        if args.windows == "first":
            windows = list(windows)[:1]
        for window in windows:
            artifact_path = output_root / "rollouts" / f"{window.name}.npz"
            video_path = (
                output_root / "videos" / f"{window.name}.mp4"
                if args.render
                else None
            )
            result = pipeline.rollout_motion_tracking(
                window,
                simulator=args.simulator,
                render=args.render,
                video_path=video_path,
                output_path=artifact_path,
                control_hz=args.control_hz,
            )
            case = {
                "id": window.name,
                "source": str(reference_path),
                "artifact": str(artifact_path),
                "video": str(video_path) if video_path is not None else None,
                "frames": window.num_frames,
                "fps": window.fps,
                "metrics": result.metrics,
                "termination_reason": result.termination_reason,
            }
            cases.append(case)
            print(
                f"[{len(cases):04d}] {window.name}: "
                f"success={result.metrics['success_rate']:.0f} "
                f"survival={result.metrics['survival_rate']:.3f}"
            )
            if args.max_cases is not None and len(cases) >= args.max_cases:
                stop = True
                break
        if stop:
            break

    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "method": str(getattr(pipeline.bundle, "METHOD_NAME", pipeline.__class__.__name__)),
        "simulator": args.simulator,
        "protocol_id": PROTOCOLS[args.simulator],
        "control_hz": args.control_hz,
        "window_steps": args.window_steps,
        "minimum_remainder_steps": args.minimum_remainder_steps,
        "window_selection": args.windows,
        "aggregate": _weighted_aggregate(cases),
        "cases": cases,
    }
    destination = output_root / "results.json"
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["aggregate"], indent=2, sort_keys=True))
    print(f"Saved {destination}")


if __name__ == "__main__":
    main()
