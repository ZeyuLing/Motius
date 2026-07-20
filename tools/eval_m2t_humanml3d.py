#!/usr/bin/env python3
"""Evaluate model-independent HumanML3D M2T prediction records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from motius.evaluation import HumanMLM2TEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--protocol-manifest", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-bertscore", action="store_true")
    parser.add_argument("--no-normalize-predictions", action="store_true")
    parser.add_argument("--bert-device", default="cuda")
    parser.add_argument("--bert-model-type", default=None)
    parser.add_argument("--bert-batch-size", type=int, default=64)
    parser.add_argument(
        "--language-reference-mode",
        choices=("token", "raw"),
        default="token",
        help=(
            "Use TM2T token/lemma references for paper-compatible scores or raw "
            "HumanML3D captions for a lexical-sensitivity diagnostic."
        ),
    )
    parser.add_argument(
        "--semantic-artifact",
        default="",
        help="Local path or Hugging Face id for the official HumanML3D matching evaluator.",
    )
    parser.add_argument("--semantic-device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--io-workers", type=int, default=32)
    parser.add_argument(
        "--gt-from-protocol",
        action="store_true",
        help="Evaluate the first protocol reference without materialized records.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    semantic_evaluator = None
    if args.semantic_artifact:
        from motius.models.humanml3d_evaluator import HumanML3DMatchingBundle

        semantic_evaluator = HumanML3DMatchingBundle.from_pretrained(
            args.semantic_artifact,
            device=args.semantic_device,
        )
    evaluator = HumanMLM2TEvaluator(
        semantic_evaluator=semantic_evaluator,
        chunk_size=args.chunk_size,
        n_repeats=args.n_repeats,
        normalize_predictions=not args.no_normalize_predictions,
        compute_bertscore=not args.no_bertscore,
        bert_device=args.bert_device,
        bert_model_type=args.bert_model_type,
        bert_batch_size=args.bert_batch_size,
        language_reference_mode=args.language_reference_mode,
        io_workers=args.io_workers,
    )
    if args.gt_from_protocol:
        if not args.protocol_manifest:
            raise ValueError("--gt-from-protocol requires --protocol-manifest")
        records = evaluator.build_gt_records(
            args.protocol_manifest,
            data_root=args.data_root,
            max_samples=args.max_samples,
        )
        metrics = evaluator.evaluate_records(records)
    else:
        if not args.prediction_dir:
            raise ValueError("--prediction-dir is required unless using GT mode")
        metrics = evaluator.evaluate(
            args.prediction_dir,
            max_samples=args.max_samples,
            protocol_manifest=args.protocol_manifest,
            data_root=args.data_root,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
