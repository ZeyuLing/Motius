#!/usr/bin/env python3
"""Build a Three.js audit viewer for captioned BABEL episodes."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


DEFAULT_CASES = ("val_919", "val_4869", "val_8738")
COLORS = (
    "#e15d44",
    "#1976d2",
    "#19a974",
    "#d69e18",
    "#8b5cf6",
    "#d9468c",
    "#0f9fa8",
)


def _load_joints(path: Path) -> np.ndarray:
    value = np.asarray(np.load(path), dtype=np.float32)
    if value.ndim == 2 and value.shape[1] == 66:
        value = value.reshape(-1, 22, 3)
    if value.ndim != 3 or value.shape[1:] != (22, 3):
        raise ValueError(f"Expected (T,22,3) joints at {path}, got {value.shape}.")
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite joints at {path}.")
    value = value.copy()
    value[..., 0] -= value[0, 0, 0]
    value[..., 2] -= value[0, 0, 2]
    value[..., 1] -= value[..., 1].min()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--predictions-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--case-ids", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument(
        "--retrieval-audit",
        type=Path,
        help="Optional exact R-Precision ranking export from export_babel_retrieval_audit.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    source = json.loads(manifest_path.read_text())
    cases = {item["case_id"]: item for item in source["cases"]}
    audit = None
    retrieval_records = {}
    if args.retrieval_audit:
        audit = json.loads(args.retrieval_audit.resolve().read_text())
        if audit.get("protocol") != source.get("protocol"):
            raise ValueError("Retrieval audit protocol does not match the sequence manifest.")
        retrieval_records = {
            (item["case_id"], int(item["segment_index"])): item
            for item in audit.get("records", [])
        }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assets = args.output_dir / "assets"
    assets.mkdir(exist_ok=True)
    exported = []
    for case_id in args.case_ids:
        case = cases[case_id]
        reference = _load_joints(manifest_path.parent / case["reference_path"])
        prediction = _load_joints(args.predictions_dir.resolve() / f"{case_id}.npy")
        frames = min(len(reference), len(prediction), int(case["total_frames"]))
        reference = np.ascontiguousarray(reference[:frames], dtype="<f4")
        prediction = np.ascontiguousarray(prediction[:frames], dtype="<f4")
        reference_file = assets / f"{case_id}_gt_joints.f32"
        prediction_file = assets / f"{case_id}_flowmdm_joints.f32"
        reference.tofile(reference_file)
        prediction.tofile(prediction_file)
        segments = []
        for index, segment in enumerate(case["segments"]):
            if int(segment["start_frame"]) >= frames:
                continue
            exported_segment = {
                "caption": segment["caption"],
                "raw_label": segment.get("raw_label", ""),
                "start_frame": int(segment["start_frame"]),
                "end_frame": min(int(segment["end_frame"]), frames),
                "color": COLORS[index % len(COLORS)],
            }
            if audit is not None:
                record = retrieval_records.get((case_id, index))
                if record is None:
                    raise ValueError(f"Retrieval audit is missing {case_id} segment {index}.")
                exported_segment["retrieval"] = {
                    "gt": record["gt"],
                    "flowmdm": record["flowmdm"],
                }
            segments.append(exported_segment)
        exported.append(
            {
                "case_id": case_id,
                "frames": frames,
                "fps": float(source.get("fps", 30.0)),
                "gt_file": reference_file.relative_to(args.output_dir).as_posix(),
                "prediction_file": prediction_file.relative_to(args.output_dir).as_posix(),
                "segments": segments,
            }
        )
    viewer_source = Path(__file__).with_name("babel_sequential_viewer.html")
    shutil.copy2(viewer_source, args.output_dir / "index.html")
    payload = {
        "protocol": source["protocol"],
        "method": "FlowMDM",
        "parents": [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
        "episodes": exported,
    }
    if audit is not None:
        payload["retrieval"] = {
            "evaluator": audit["evaluator"],
            "seed": int(audit["seed"]),
            "chunk_size": int(audit["chunk_size"]),
            "top_k": int(audit["top_k"]),
            "direction_note": audit["direction_note"],
        }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"output": str(args.output_dir.resolve()), "cases": args.case_ids}, indent=2))


if __name__ == "__main__":
    main()
