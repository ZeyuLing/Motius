#!/usr/bin/env python3
"""Publish the complete HumanML3D tokenizer reconstruction result pack."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / (
    "docs/leaderboards/hf_space_motion_reconstruction/"
    "reconstruction_results.json"
)

METHODS = (
    ("t2mgpt", "T2M-GPT / MotionGPT", "T2M-GPT VQ-VAE", "HumanML3D-263 bridge"),
    ("momask", "MoMask", "MoMask RVQ-VAE", "HumanML3D-263 bridge"),
    ("mld", "MLD / MotionLCM", "MLD VAE", "HumanML3D-263 bridge"),
    ("mogents", "MoGenTS", "MoGenTS tokenizer", "HumanML3D-263 bridge"),
    ("motiongpt3", "MotionGPT3", "MotionGPT3 tokenizer", "HumanML3D-263 bridge"),
    (
        "motionstreamer",
        "MotionStreamer",
        "MotionStreamer Causal-TAE",
        "native MotionStreamer-272 bridge",
    ),
    (
        "gotozero",
        "GoToZero / MotionMillion",
        "GoToZero / MotionMillion FSQ",
        "native MotionStreamer-272 bridge",
    ),
    ("prism", "PRISM", "PRISM 2D latent VAE", "native motion135"),
    ("vermo", "VerMo", "VerMo tokenizer", "native motion135"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-root",
        required=True,
        type=Path,
        help="HumanML3D reconstruction root containing metrics_summary.json.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def physics_row(artifact_root: Path, method: str) -> dict:
    path = artifact_root / "ms272" / method / "metrics" / "physics.json"
    payload = json.loads(path.read_text())
    if int(payload["raw"]["n"]) != 4042:
        raise ValueError(f"{method}: incomplete physics population")
    return payload["table"]


def main() -> None:
    args = parse_args()
    artifact_root = args.artifact_root.expanduser().resolve()
    summary = json.loads((artifact_root / "metrics_summary.json").read_text())
    source_rows = {row["method"]: row for row in summary["methods"]}
    missing = [method for method, *_ in METHODS if method not in source_rows]
    if missing:
        raise KeyError(f"Missing reconstruction rows: {missing}")

    gt_physics = physics_row(artifact_root, "gt_hml263_bridge")
    rows = [
        {
            "method": "GT",
            "version": "HumanML3D-263 bridge input",
            "is_reference": True,
            "semantic_samples": 3972,
            "geometry_samples": 4042,
            "rfid": 0.0,
            "embedding_l2": 0.0,
            "mpjpe_mm": 0.0,
            "pa_mpjpe_mm": 0.0,
            "mpjre_deg": 0.0,
            "slide": gt_physics["Slide"],
            "float": gt_physics["Float"],
            "penetration": gt_physics["Penet"],
            "jitter": gt_physics["Jitter"],
            "physical_scope": "HumanML3D-263 bridge input",
        }
    ]
    for method, label, version, scope in METHODS:
        source = source_rows[method]
        physics = physics_row(artifact_root, method)
        if int(source["geom_used"]) != 4042 or int(source["rfid_n"]) != 3972:
            raise ValueError(f"{method}: incomplete benchmark population")
        rows.append(
            {
                "method": label,
                "version": version,
                "is_reference": False,
                "semantic_samples": int(source["rfid_n"]),
                "geometry_samples": int(source["geom_used"]),
                "rfid": source["rfid"],
                "embedding_l2": source["emb_l2"],
                "mpjpe_mm": source["mpjpe_mm"],
                "pa_mpjpe_mm": source["pa_mpjpe_mm"],
                "mpjre_deg": source["mpjre_deg"],
                "slide": physics["Slide"],
                "float": physics["Float"],
                "penetration": physics["Penet"],
                "jitter": physics["Jitter"],
                "physical_scope": scope,
            }
        )

    payload = {
        "schema_version": 2,
        "benchmark": "Motion Reconstruction · HumanML3D",
        "updated": date.today().isoformat(),
        "protocol": {
            "split": "HumanML3D test, 4,042 motions",
            "semantic_population": "3,972 clips with at least 60 frames",
            "geometry_population": 4042,
            "semantic_evaluator": (
                "MotionStreamer HumanML3D paired reconstruction embedding protocol"
            ),
            "physical_protocol": "Motius SMPL-22 diagnostics; values use table scale",
            "ranking": "GT is a calibration reference and excluded from model ranking",
        },
        "rows": rows,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"output": str(output), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
