#!/usr/bin/env python3
"""Evaluate Motius joint-level physical metrics for a motion directory."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.metrics.physical import (
    aggregate_physical_metrics,
    compute_physical_metrics,
    table_scaled_physical_metrics,
)


def _score(path: Path) -> tuple[dict[str, float] | None, str | None]:
    try:
        return compute_physical_metrics(np.load(path, allow_pickle=False)), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{path.name}: {type(exc).__name__}: {exc}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joints-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--ids-file", type=Path)
    parser.add_argument("--workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    joints_dir = args.joints_dir.resolve()
    if args.ids_file is None:
        files = sorted(joints_dir.glob("*.npy"))
        missing: list[str] = []
    else:
        ids = [
            line.strip()
            for line in args.ids_file.resolve().read_text().splitlines()
            if line.strip()
        ]
        files = [joints_dir / f"{motion_id}.npy" for motion_id in ids]
        missing = [path.stem for path in files if not path.is_file()]
        files = [path for path in files if path.is_file()]

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        scored = list(executor.map(_score, files))
    rows = [row for row, _ in scored]
    failures = [error for _, error in scored if error is not None]
    raw = aggregate_physical_metrics(rows)
    result = {
        "protocol": "Motius SMPL-22 joint-level physical metrics",
        "representation": "canonical SMPL-22 joints66, Y-up, metres",
        "joints_dir": str(joints_dir),
        "requested": len(files) + len(missing),
        "raw": raw,
        "table_scaled": table_scaled_physical_metrics(raw),
        "missing": len(missing),
        "failed": len(failures),
        "missing_ids": missing[:100],
        "failures": failures[:100],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if missing or failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
