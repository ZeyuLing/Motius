#!/usr/bin/env python3
"""Evaluate paired InterHuman-262 prediction packs with InterCLIP."""

from __future__ import annotations

import argparse
from pathlib import Path

from motius.evaluation.evaluators import InterHuman262Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", required=True, help="Hub repo or artifact directory")
    parser.add_argument("--gt", required=True, help="Ground-truth native-262 NPZ")
    parser.add_argument("--pred", action="append", required=True, metavar="NAME=NPZ")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--retrieval-batch-size", type=int, default=96)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions = {}
    for item in args.pred:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            raise ValueError(f"--pred must be NAME=NPZ, got {item!r}")
        predictions[name] = path
    evaluator = InterHuman262Evaluator.from_pretrained(
        args.evaluator,
        device=args.device,
        batch_size=args.batch_size,
        retrieval_batch_size=args.retrieval_batch_size,
        retrieval_repeats=args.repeats,
    )
    results = evaluator.evaluate_npz(args.gt, predictions, seed=args.seed)
    evaluator.write_json(results, Path(args.output))
    print(Path(args.output))


if __name__ == "__main__":
    main()
