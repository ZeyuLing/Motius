#!/usr/bin/env python3
"""Evaluate G1-38D predictions with the released Motius TMR-G1 evaluator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from motius.evaluation import TMRG1Evaluator


def _load_motion(path: Path) -> np.ndarray:
    value = np.load(path)
    if isinstance(value, np.lib.npyio.NpzFile):
        for key in ("g1_38", "motion", "pred", "arr_0"):
            if key in value:
                return np.asarray(value[key], dtype=np.float32)
        raise KeyError(f"No G1-38D array found in {path}; keys={value.files}")
    return np.asarray(value, dtype=np.float32)


def _motion_path(directory: Path, key: str) -> Path:
    for suffix in (".npy", ".npz"):
        path = directory / f"{key}{suffix}"
        if path.is_file():
            return path
    raise FileNotFoundError(f"Missing {key}.npy/.npz under {directory}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--checkpoint", default="ZeyuLing/motius-evaluator-g1-38d-tmr")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation/g1_tmr/metrics.json"),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    keys = [str(row.get("id") or row.get("keyid")) for row in rows]
    captions = [str(row.get("caption") or row.get("text")) for row in rows]
    predictions = [_load_motion(_motion_path(args.pred_dir, key)) for key in keys]
    references = None
    if args.reference_dir:
        references = [_load_motion(_motion_path(args.reference_dir, key)) for key in keys]

    evaluator = TMRG1Evaluator.from_pretrained(
        args.checkpoint, device=args.device, batch_size=args.batch_size
    )
    metrics = evaluator.evaluate(
        captions,
        predictions,
        references,
        chunk_size=args.chunk_size,
        n_repeats=args.repeats,
        seed=args.seed,
    )
    metrics.update(
        checkpoint=args.checkpoint,
        manifest=str(args.manifest),
        pred_dir=str(args.pred_dir),
        reference_dir=str(args.reference_dir) if args.reference_dir else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
