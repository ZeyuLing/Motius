#!/usr/bin/env python3
"""Build the public BABEL sequential-generation evaluation protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.motion import babel135_to_joints, encode_babel135, infer_smpl22_offsets


def standardize_babel_text(value: object) -> str:
    text = str(value or "").strip()
    compact = text.lower().replace("-", " ").replace("_", " ")
    if compact in {"t pose", "tpose"}:
        return "t-pose"
    if compact in {"a pose", "apose"}:
        return "a-pose"
    return text


def extract_reference_segments(
    sample: Mapping[str, object],
    annotation: Mapping[str, object],
    *,
    min_frames: int = 30,
    max_frames: int = 200,
) -> Iterable[tuple[str, int, int]]:
    frame_ann = annotation.get("frame_ann")
    if not frame_ann:
        return
    nframes = len(sample["poses"])
    fps = float(sample["fps"])
    for label in frame_ann.get("labels", []):
        caption = standardize_babel_text(label.get("proc_label"))
        if not caption or caption.lower() == "transition":
            continue
        start = int(float(label["start_t"]) * fps)
        end = min(nframes, int(float(label["end_t"]) * fps))
        if min_frames <= end - start <= max_frames:
            yield caption, start, end


def infer_protocol_offsets(
    samples: Sequence[Mapping[str, object]], *, sample_count: int = 32
) -> np.ndarray:
    if not samples:
        raise ValueError("BABEL data is empty.")
    count = min(max(1, int(sample_count)), len(samples))
    indices = np.linspace(0, len(samples) - 1, count, dtype=np.int64)
    candidates = []
    for index in indices:
        sample = samples[int(index)]
        candidates.append(
            infer_smpl22_offsets(
                sample["poses"], sample["trans"], sample["joint_positions"]
            )
        )
    offsets = np.median(np.stack(candidates), axis=0).astype(np.float32)
    offsets[0] = np.median(np.stack(candidates), axis=0)[0]
    return offsets


def _joints(sample: Mapping[str, object], start: int, end: int, offsets: np.ndarray) -> np.ndarray:
    features = encode_babel135(
        np.asarray(sample["poses"])[start:end],
        np.asarray(sample["trans"])[start:end],
    )
    return babel135_to_joints(features, bone_offsets=offsets).reshape(len(features), 66)


def _sample_candidates(
    candidates: Sequence[dict[str, object]], maximum: int | None, rng: np.random.Generator
) -> list[dict[str, object]]:
    if maximum is None or len(candidates) <= maximum:
        return list(candidates)
    selected = np.sort(rng.choice(len(candidates), size=int(maximum), replace=False))
    return [candidates[int(index)] for index in selected]


def _relative(path: Path, parent: Path) -> str:
    return path.resolve().relative_to(parent.resolve()).as_posix()


def build_protocol(
    samples: Sequence[Mapping[str, object]],
    annotations: Mapping[str, Mapping[str, object]],
    compositions: Sequence[Mapping[str, object]],
    output_root: Path,
    *,
    transition_frames: int = 30,
    seed: int = 0,
    offset_samples: int = 32,
    max_reference_segments: int | None = 2048,
    max_reference_transitions: int | None = 2048,
) -> dict[str, object]:
    output_root.mkdir(parents=True, exist_ok=True)
    offsets = infer_protocol_offsets(samples, sample_count=offset_samples)
    offsets_path = output_root / "smpl22_offsets_y.npy"
    np.save(offsets_path, offsets)

    segment_candidates = []
    transition_candidates = []
    rng = np.random.default_rng(int(seed))
    for sample_index, sample in enumerate(samples):
        babel_id = str(sample["babel_id"])
        annotation = annotations.get(babel_id)
        if annotation is not None:
            for label_index, (caption, start, end) in enumerate(
                extract_reference_segments(sample, annotation)
            ):
                segment_candidates.append(
                    {
                        "sample_index": sample_index,
                        "item_index": label_index,
                        "babel_id": babel_id,
                        "caption": caption,
                        "start_frame": start,
                        "end_frame": end,
                    }
                )

        nframes = len(sample["poses"])
        count = nframes // int(transition_frames)
        for window_index in range(count):
            maximum = nframes - int(transition_frames)
            start = int(rng.integers(0, maximum)) if maximum > 0 else 0
            end = start + int(transition_frames)
            transition_candidates.append(
                {
                    "sample_index": sample_index,
                    "item_index": window_index,
                    "babel_id": babel_id,
                    "start_frame": start,
                    "end_frame": end,
                }
            )

    reference_segments = _sample_candidates(
        segment_candidates, max_reference_segments, np.random.default_rng(int(seed) + 1)
    )
    reference_transitions = _sample_candidates(
        transition_candidates,
        max_reference_transitions,
        np.random.default_rng(int(seed) + 2),
    )

    selected_by_sample: dict[int, list[tuple[str, int, dict[str, object]]]] = {}
    for kind, items in (
        ("segment", reference_segments),
        ("transition", reference_transitions),
    ):
        for pool_index, item in enumerate(items):
            selected_by_sample.setdefault(int(item["sample_index"]), []).append(
                (kind, pool_index, item)
            )

    segment_motions: list[np.ndarray | None] = [None] * len(reference_segments)
    transition_motions: list[np.ndarray | None] = [None] * len(reference_transitions)
    for sample_index, selected_items in selected_by_sample.items():
        sample = samples[sample_index]
        full_joints = _joints(sample, 0, len(sample["poses"]), offsets)
        for kind, pool_index, item in selected_items:
            start = int(item["start_frame"])
            end = int(item["end_frame"])
            destination = segment_motions if kind == "segment" else transition_motions
            destination[pool_index] = full_joints[start:end]
            item["pool_index"] = pool_index
            del item["sample_index"]
            del item["item_index"]

    def save_pool(path: Path, motions: Sequence[np.ndarray | None]) -> None:
        if not motions or any(item is None for item in motions):
            raise RuntimeError(f"Incomplete reference pool for {path}")
        arrays = [np.asarray(item, dtype=np.float32) for item in motions]
        lengths = np.asarray([len(item) for item in arrays], dtype=np.int32)
        padded = np.zeros((len(arrays), int(lengths.max()), 66), dtype=np.float32)
        for index, motion in enumerate(arrays):
            padded[index, : len(motion)] = motion
        np.savez(path, motions=padded, lengths=lengths)

    segment_pool_path = output_root / "reference_segments.npz"
    transition_pool_path = output_root / "reference_transitions.npz"
    save_pool(segment_pool_path, segment_motions)
    save_pool(transition_pool_path, transition_motions)

    cases = []
    for item in compositions:
        captions = [standardize_babel_text(value) for value in item["text"]]
        lengths = [int(value) for value in item["lengths"]]
        if len(captions) != len(lengths) or not captions:
            raise ValueError(f"Invalid composition {item.get('id')!r}.")
        cursor = 0
        segments = []
        for caption, length in zip(captions, lengths):
            segments.append(
                {"caption": caption, "start_frame": cursor, "end_frame": cursor + length}
            )
            cursor += length
        cases.append(
            {
                "case_id": str(item["id"]),
                "scenario": str(item.get("scenario", "unspecified")),
                "total_frames": cursor,
                "segments": segments,
            }
        )

    manifest = {
        "protocol": "babel-flowmdm-val-joints66-v2",
        "split": "val",
        "fps": 30,
        "motion_representation": "canonical SMPL-22 joints66",
        "composition_source": "FlowMDM babel_val_set.json",
        "reference_source": "BABEL val frame annotations and SMPL-H motions",
        "reference_sampling": {
            "segments": "deterministic uniform sample from valid BABEL val action segments",
            "transitions": "deterministic uniform sample from BABEL val 30-frame windows",
            "max_segments": max_reference_segments,
            "max_transitions": max_reference_transitions,
        },
        "transition_frames": int(transition_frames),
        "seed": int(seed),
        "smpl22_offsets": _relative(offsets_path, output_root),
        "reference_segment_pool": _relative(segment_pool_path, output_root),
        "reference_transition_pool": _relative(transition_pool_path, output_root),
        "cases": cases,
        "reference_segments": reference_segments,
        "reference_transitions": reference_transitions,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--babel-val",
        default="data/babel/babel-smplh-30fps-male/val.pth.tar",
    )
    parser.add_argument("--annotations", default="data/babel/babel-teach/val.json")
    parser.add_argument(
        "--compositions",
        default="data/babel/flowmdm_eval_protocol/dataset/babel_val_set.json",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/evaluation/babel_sequential/protocol_v2",
    )
    parser.add_argument("--transition-frames", type=int, default=30)
    parser.add_argument("--offset-samples", type=int, default=32)
    parser.add_argument("--max-reference-segments", type=int, default=2048)
    parser.add_argument("--max-reference-transitions", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = joblib.load(args.babel_val)
    annotations = json.loads(Path(args.annotations).read_text())
    compositions = json.loads(Path(args.compositions).read_text())
    manifest = build_protocol(
        samples,
        annotations,
        compositions,
        Path(args.output_root).resolve(),
        transition_frames=args.transition_frames,
        seed=args.seed,
        offset_samples=args.offset_samples,
        max_reference_segments=args.max_reference_segments,
        max_reference_transitions=args.max_reference_transitions,
    )
    print(
        json.dumps(
            {
                "cases": len(manifest["cases"]),
                "segments": sum(len(item["segments"]) for item in manifest["cases"]),
                "reference_segments": len(manifest["reference_segments"]),
                "reference_transitions": len(manifest["reference_transitions"]),
                "output": str(Path(args.output_root).resolve() / "manifest.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
