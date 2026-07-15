#!/usr/bin/env python3
"""Export exact BABEL retrieval rankings for selected sequential viewer cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.evaluation.evaluators.tmr import TMRTextMotionEvaluator
from motius.evaluation.metrics import retrieval_audit
from motius.evaluation.sequential import caption_group_id, load_joints66
from motius.motion.skeleton.canonical import canonicalize_smpl22_joints


DEFAULT_CASES = ("val_919", "val_4869", "val_8738")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--evaluator",
        default="ZeyuLing/motius-evaluator-universal-smplh-joints66",
    )
    parser.add_argument(
        "--evaluator-id",
        help="Public evaluator identifier recorded in the audit when loading a local path.",
    )
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def _metadata(cases: list[dict]) -> list[dict[str, object]]:
    values = []
    for case_index, case in enumerate(cases):
        for segment_index, segment in enumerate(case["segments"]):
            values.append(
                {
                    "case_index": int(case_index),
                    "case_id": str(case["case_id"]),
                    "segment_index": int(segment_index),
                    "caption": str(segment["caption"]),
                    "start_frame": int(segment["start_frame"]),
                    "end_frame": int(segment["end_frame"]),
                }
            )
    return values


def _candidate_indices(count: int, queries: list[int], *, chunk: int, seed: int) -> list[int]:
    chunk = max(3, min(int(chunk), count))
    used = count // chunk * chunk
    query_set = set(queries)
    order = np.random.default_rng(seed).permutation(count)
    selected = set()
    for start in range(0, used, chunk):
        batch = order[start : start + chunk]
        if any(int(index) in query_set for index in batch):
            selected.update(map(int, batch))
    return sorted(selected)


def _enrich(record: dict[str, object], metadata: list[dict[str, object]]) -> dict[str, object]:
    result = dict(record)
    for direction in ("text_to_motion", "motion_to_text"):
        if direction not in result:
            continue
        ranking = dict(result[direction])
        ranking["top"] = [
            {**item, **metadata[int(item["sample_index"])]} for item in ranking["top"]
        ]
        result[direction] = ranking
    return result


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text())
    cases = list(manifest["cases"])
    metadata = _metadata(cases)
    selected_cases = set(args.case_ids)
    query_indices = [
        index for index, item in enumerate(metadata) if item["case_id"] in selected_cases
    ]
    missing = selected_cases - {str(item["case_id"]) for item in metadata}
    if missing:
        raise ValueError(f"Unknown case ids: {sorted(missing)}")

    captions = [str(item["caption"]) for item in metadata]
    candidate_indices = _candidate_indices(
        len(metadata), query_indices, chunk=args.chunk_size, seed=args.seed
    )
    candidates_by_case: dict[int, list[int]] = {}
    for index in candidate_indices:
        case_index = int(metadata[index]["case_index"])
        candidates_by_case.setdefault(case_index, []).append(index)

    reference_segments: list[np.ndarray] = []
    predicted_segments: list[np.ndarray] = []
    predictions_dir = args.predictions_dir.resolve()
    ordered_candidate_indices: list[int] = []
    for case_index, sample_indices in candidates_by_case.items():
        case = cases[case_index]
        reference_path = Path(case["reference_path"])
        if not reference_path.is_absolute():
            reference_path = manifest_path.parent / reference_path
        reference = load_joints66(reference_path)
        predicted = load_joints66(predictions_dir / f"{case['case_id']}.npy")
        for sample_index in sample_indices:
            segment = case["segments"][int(metadata[sample_index]["segment_index"])]
            region = slice(int(segment["start_frame"]), int(segment["end_frame"]))
            ordered_candidate_indices.append(sample_index)
            reference_segments.append(canonicalize_smpl22_joints(reference[region]))
            predicted_segments.append(canonicalize_smpl22_joints(predicted[region]))

    evaluator = TMRTextMotionEvaluator.from_pretrained(
        args.evaluator,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )
    candidate_captions = [captions[index] for index in ordered_candidate_indices]
    encoded_text = evaluator.encode_texts(candidate_captions)
    encoded_reference = evaluator.encode_motions(reference_segments)
    encoded_predicted = evaluator.encode_motions(predicted_segments)
    embedding_shape = (len(metadata), encoded_text.shape[1])
    text_embeddings = np.zeros(embedding_shape, dtype=encoded_text.dtype)
    reference_embeddings = np.zeros(embedding_shape, dtype=encoded_reference.dtype)
    predicted_embeddings = np.zeros(embedding_shape, dtype=encoded_predicted.dtype)
    text_embeddings[ordered_candidate_indices] = encoded_text
    reference_embeddings[ordered_candidate_indices] = encoded_reference
    predicted_embeddings[ordered_candidate_indices] = encoded_predicted
    group_ids = [caption_group_id(caption) for caption in captions]
    common = {
        "chunk": args.chunk_size,
        "seed": args.seed,
        "top_k": args.top_k,
        "positive_group_ids": group_ids,
        "query_indices": query_indices,
    }
    gt = retrieval_audit(text_embeddings, reference_embeddings, **common)
    flowmdm = retrieval_audit(text_embeddings, predicted_embeddings, **common)

    records = []
    for index in query_indices:
        records.append(
            {
                **metadata[index],
                "sample_index": index,
                "gt": _enrich(gt[index], metadata),
                "flowmdm": _enrich(flowmdm[index], metadata),
            }
        )
    payload = {
        "protocol": manifest["protocol"],
        "evaluator": args.evaluator_id or str(args.evaluator),
        "seed": args.seed,
        "chunk_size": args.chunk_size,
        "top_k": args.top_k,
        "candidate_samples_encoded": len(candidate_indices),
        "direction_note": {
            "text_to_motion": "Leaderboard R-Precision direction.",
            "motion_to_text": "Nearest captions for the selected motion subclip.",
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output.resolve()), "records": len(records)}, indent=2))


if __name__ == "__main__":
    main()
