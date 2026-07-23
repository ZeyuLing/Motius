#!/usr/bin/env python3
"""Evaluate a MotionStreamer-272 directory with the public evaluator."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.metrics.t2m import aggregate_t2m_metrics
from motius.models.motionstreamer import MotionStreamer272Evaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--texts-dir", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument(
        "--evaluator",
        default="ZeyuLing/motius-evaluator-motionstreamer-272",
    )
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _selected_caption(path: Path) -> str:
    def timestamp(value: str) -> float:
        value = value.strip().lower()
        return 0.0 if value in {"", "nan"} else float(value)

    captions = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split("#")
        if (
            len(parts) >= 4
            and timestamp(parts[2]) == 0
            and timestamp(parts[3]) == 0
        ):
            captions.append(parts[0].strip())
    if len(captions) != 1:
        raise ValueError(
            f"Expected one selected full-clip caption in {path}, got {captions}."
        )
    return captions[0]


def _load_pair(
    name: str,
    *,
    data_root: Path,
    texts_dir: Path,
    predictions_dir: Path,
):
    reference_path = data_root / "motion_data" / f"{name}.npy"
    prediction_path = predictions_dir / f"{name}.npy"
    caption_path = texts_dir / f"{name}.txt"
    for path in (reference_path, prediction_path, caption_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    reference = np.asarray(np.load(reference_path), dtype=np.float32)
    prediction = np.asarray(np.load(prediction_path), dtype=np.float32)
    return _selected_caption(caption_path), reference, prediction


def main() -> None:
    args = parse_args()
    names = [
        name.strip()
        for name in (args.data_root / "split" / "test.txt").read_text().splitlines()
        if name.strip()
    ]
    captions, references, predictions = [], [], []
    reference_lengths, prediction_lengths = [], []
    loader = partial(
        _load_pair,
        data_root=args.data_root,
        texts_dir=args.texts_dir,
        predictions_dir=args.predictions,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        pairs = list(executor.map(loader, names))
    for caption, reference, prediction in pairs:
        captions.append(caption)
        references.append(reference)
        predictions.append(prediction)
        reference_lengths.append(len(reference))
        prediction_lengths.append(len(prediction))

    evaluator = MotionStreamer272Evaluator.from_pretrained(
        args.evaluator, device=args.device
    )
    text_embeddings = evaluator.encode_texts(captions, batch_size=args.batch_size)
    reference_embeddings = evaluator.encode_motions(
        references, lengths=reference_lengths, batch_size=args.batch_size
    )
    prediction_embeddings = evaluator.encode_motions(
        predictions, lengths=prediction_lengths, batch_size=args.batch_size
    )
    result = aggregate_t2m_metrics(
        text_embeddings,
        reference_embeddings,
        prediction_embeddings,
        n_repeats=args.n_repeats,
        chunk=args.chunk_size,
        seed=args.seed,
        normalize_fid=True,
    )
    result.update(
        {
            "schema_version": 1,
            "method": args.method,
            "dataset": "HumanML3D official test, selected-caption protocol",
            "n_samples": len(names),
            "evaluator": args.evaluator,
            "representation": "MotionStreamer-272 at 30 fps",
            "caption_protocol": "one selected full-clip caption per sample",
            "fid_embedding_space": "l2_normalized",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
