#!/usr/bin/env python3
"""Evaluate joints66 T2M predictions with a public Motius TMR artifact."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.evaluators.tmr import TMRTextMotionEvaluator


def _load_joints66(path: Path) -> np.ndarray:
    motion = np.asarray(np.load(path), dtype=np.float32)
    if motion.ndim == 3 and motion.shape[1:] == (22, 3):
        motion = motion.reshape(len(motion), 66)
    if motion.ndim != 2 or motion.shape[1] != 66:
        raise ValueError(f"Expected joints66 at {path}, got {motion.shape}.")
    if len(motion) < 2 or not np.isfinite(motion).all():
        raise ValueError(f"Motion at {path} must contain at least two finite frames.")
    return motion


def _prediction_candidates(keyid: str, annotation: Mapping[str, object]) -> list[str]:
    values = [keyid, str(annotation.get("path", keyid))]
    for value in tuple(values):
        for prefix in ("h3dtest_", "humanml3d_"):
            if value.startswith(prefix):
                values.append(value[len(prefix) :])
    return list(dict.fromkeys(values))


def load_protocol(
    dataset_dir: str | Path,
    split: str,
    predictions_dir: str | Path,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray], list[str]]:
    """Load paired selected captions, predictions, and references."""

    dataset_dir = Path(dataset_dir)
    predictions_dir = Path(predictions_dir)
    annotations = json.loads((dataset_dir / "annotations.json").read_text())
    keyids = [
        line.strip()
        for line in (dataset_dir / "splits" / f"{split}.txt").read_text().splitlines()
        if line.strip()
    ]
    captions: list[str] = []
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    used_keyids: list[str] = []
    for keyid in keyids:
        annotation = annotations.get(keyid)
        if not isinstance(annotation, Mapping):
            raise KeyError(f"Missing annotation for split key {keyid!r}.")
        text_items = annotation.get("annotations")
        if not isinstance(text_items, list) or len(text_items) != 1:
            raise ValueError(
                f"{keyid!r} must contain exactly one selected caption, got {text_items!r}."
            )
        caption = str(text_items[0].get("text", "")).strip()
        if not caption:
            raise ValueError(f"Selected caption for {keyid!r} is empty.")

        reference_stem = str(annotation.get("path", keyid))
        reference_path = dataset_dir / "motions" / f"{reference_stem}.npy"
        prediction_path = next(
            (
                predictions_dir / f"{stem}.npy"
                for stem in _prediction_candidates(keyid, annotation)
                if (predictions_dir / f"{stem}.npy").is_file()
            ),
            None,
        )
        if prediction_path is None:
            raise FileNotFoundError(
                f"No prediction for {keyid!r}; tried "
                f"{_prediction_candidates(keyid, annotation)} under {predictions_dir}."
            )
        captions.append(caption)
        predictions.append(_load_joints66(prediction_path))
        references.append(_load_joints66(reference_path))
        used_keyids.append(keyid)
    return captions, predictions, references, used_keyids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--split", default="humanml3d_test")
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument(
        "--evaluator",
        default="ZeyuLing/motius-evaluator-universal-smplh-joints66",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    captions, predictions, references, keyids = load_protocol(
        args.dataset_dir,
        args.split,
        args.predictions_dir,
    )
    evaluator = TMRTextMotionEvaluator.from_pretrained(
        args.evaluator,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    result = evaluator.evaluate(
        captions,
        predictions,
        references,
        chunk_size=args.chunk_size,
        n_repeats=args.n_repeats,
        seed=args.seed,
    )
    result.update(
        {
            "method": args.method,
            "split": args.split,
            "n_samples": len(keyids),
            "dataset_dir": str(Path(args.dataset_dir).resolve()),
            "predictions_dir": str(Path(args.predictions_dir).resolve()),
            "evaluator": args.evaluator,
            "caption_protocol": "one selected full-clip caption per split sample",
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
