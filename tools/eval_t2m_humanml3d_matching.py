#!/usr/bin/env python3
"""Evaluate selected-caption T2M outputs with the HumanML3D matcher."""

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
from motius.models.humanml3d_evaluator import HumanML3DMatchingBundle
from motius.motion.representation.convert import motion272_to_hml263
from tools.infer_unimumo_humanml3d import selected_caption


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--path-root", required=True, type=Path)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        help="Official HumanML3D new_joint_vecs directory; preferred for FID.",
    )
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--evaluator", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--n-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=32)
    parser.add_argument(
        "--allow-missing-references",
        action="store_true",
        help=(
            "Evaluate the exact official-reference intersection and report missing "
            "sample IDs instead of mixing in converted references."
        ),
    )
    return parser.parse_args()


def resolve(path: str, root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _load_record(
    item,
    *,
    predictions_dir: Path,
    path_root: Path,
    reference_dir: Path | None,
):
    name, record = item
    official_reference = (
        reference_dir / f"{name}.npy" if reference_dir is not None else None
    )
    if official_reference is not None and not official_reference.is_file():
        return name, None

    prediction_path = predictions_dir / f"{name}.npy"
    if not prediction_path.is_file():
        raise FileNotFoundError(prediction_path)
    caption_path = resolve(record["hierarchical_caption_path"], path_root)
    prediction = np.asarray(np.load(prediction_path), dtype=np.float32)
    if official_reference is not None:
        reference = np.asarray(np.load(official_reference), dtype=np.float32)
    else:
        reference_path = resolve(record["smplx_path"], path_root)
        reference272 = np.asarray(np.load(reference_path), dtype=np.float32)
        reference = motion272_to_hml263(
            reference272,
            src_fps=float(record.get("fps") or 30.0),
            dst_fps=20.0,
        )
    return name, (selected_caption(caption_path), prediction, reference)


def main() -> None:
    args = parse_args()
    protocol = json.loads(args.annotation.read_text(encoding="utf-8"))
    records = sorted(protocol["data_list"].items())
    captions: list[str] = []
    predictions: list[np.ndarray] = []
    references: list[np.ndarray] = []
    sample_ids: list[str] = []
    missing_reference_ids: list[str] = []
    loader = partial(
        _load_record,
        predictions_dir=args.predictions,
        path_root=args.path_root,
        reference_dir=args.reference_dir,
    )
    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        loaded_records = list(executor.map(loader, records))
    for name, payload in loaded_records:
        if payload is None:
            if not args.allow_missing_references:
                raise FileNotFoundError(args.reference_dir / f"{name}.npy")
            missing_reference_ids.append(name)
            continue
        caption, prediction, reference = payload
        captions.append(caption)
        predictions.append(prediction)
        references.append(reference)
        sample_ids.append(name)

    evaluator = HumanML3DMatchingBundle.from_pretrained(
        args.evaluator, device=args.device
    )
    text_embeddings = evaluator.encode_texts(captions)
    reference_embeddings = evaluator.encode_motions(references)
    prediction_embeddings = evaluator.encode_motions(predictions)
    result = aggregate_t2m_metrics(
        text_embeddings,
        reference_embeddings,
        prediction_embeddings,
        n_repeats=args.n_repeats,
        chunk=args.chunk_size,
        seed=args.seed,
        normalize_fid=False,
    )
    result.update(
        {
            "schema_version": 1,
            "method": args.method,
            "dataset": "HumanML3D official test, selected-caption protocol",
            "n_samples": len(sample_ids),
            "protocol_samples": len(records),
            "missing_reference_count": len(missing_reference_ids),
            "missing_reference_ids": missing_reference_ids,
            "evaluator": args.evaluator,
            "chunk_size": args.chunk_size,
            "caption_protocol": "macro -> meso -> micro selected full-clip caption",
            "reference_representation": (
                "official HumanML3D new_joint_vecs at 20 fps"
                if args.reference_dir is not None
                else "MotionStreamer-272 -> HumanML3D-263 at 20 fps fallback"
            ),
            "prediction_representation": "HumanML3D-263 at 20 fps",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
