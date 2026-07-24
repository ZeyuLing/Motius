#!/usr/bin/env python3
"""Build a deterministic, resume-aware shard plan for 3DPW inference."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


SCHEMA_REVISION = "motius_3dpw_monocular_shard_plan_v1"


def build_shard_plan(
    *,
    video_manifest: dict,
    ground_truth_index: dict,
    prediction_dir: Path,
    num_shards: int,
    max_sequences: int | None = None,
) -> dict:
    """Balance unfinished videos by estimated target-person frame count."""

    if num_shards < 1:
        raise ValueError("num_shards must be positive.")
    records = sorted(
        video_manifest["videos"],
        key=lambda item: item["sequence_id"],
    )
    if max_sequences is not None:
        if max_sequences < 1:
            raise ValueError("max_sequences must be positive.")
        records = records[:max_sequences]

    target_counts = Counter(
        artifact["public_manifest"]["metadata"]["sequence_id"]
        for artifact in ground_truth_index["artifacts"]
    )
    record_ids = {record["sequence_id"] for record in records}
    missing_targets = sorted(
        sequence_id
        for sequence_id in record_ids
        if target_counts[sequence_id] < 1
    )
    if missing_targets:
        raise ValueError(
            "Videos have no target tracks in the ground-truth index: "
            + ", ".join(missing_targets)
        )

    completed = {
        path.name.removesuffix(".motius.npz")
        for path in prediction_dir.glob("*.motius.npz")
        if path.is_file()
    }
    pending = [
        record for record in records if record["sequence_id"] not in completed
    ]
    weighted = sorted(
        (
            (
                int(record["frames"]) * target_counts[record["sequence_id"]],
                record["sequence_id"],
            )
            for record in pending
        ),
        key=lambda item: (-item[0], item[1]),
    )

    assignments: list[list[str]] = [[] for _ in range(num_shards)]
    estimated_person_frames = [0] * num_shards
    for weight, sequence_id in weighted:
        shard_id = min(
            range(num_shards),
            key=lambda index: (estimated_person_frames[index], index),
        )
        assignments[shard_id].append(sequence_id)
        estimated_person_frames[shard_id] += weight

    return {
        "schema_revision": SCHEMA_REVISION,
        "num_shards": num_shards,
        "population": len(records),
        "completed_at_plan_time": sum(
            record["sequence_id"] in completed for record in records
        ),
        "pending": len(pending),
        "estimated_person_frames": estimated_person_frames,
        "assignments": {
            str(index): sequence_ids
            for index, sequence_ids in enumerate(assignments)
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-manifest", type=Path, required=True)
    parser.add_argument("--ground-truth-index", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    payload = build_shard_plan(
        video_manifest=json.loads(args.video_manifest.read_text()),
        ground_truth_index=json.loads(args.ground_truth_index.read_text()),
        prediction_dir=args.prediction_dir,
        num_shards=args.num_shards,
        max_sequences=args.max_sequences,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
