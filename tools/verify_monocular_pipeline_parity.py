#!/usr/bin/env python3
"""Compare two Motius monocular stage traces."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.utils.monocular_parity import MonocularParityTrace


GEM_X_POSTPROCESS_ATOL = 3e-6
GEM_X_POSTPROCESS_FIELDS = (
    "06_model_output/body_params_global/body_pose",
    "06_model_output/body_params_global/transl",
    "06_model_output/body_params_incam/body_pose",
    "06_model_output/net_outputs/decode_dict/body_pose",
    "06_model_output/net_outputs/pred_body_params_global/body_pose",
    "06_model_output/net_outputs/pred_body_params_global/transl",
    "06_model_output/net_outputs/pred_body_params_incam/body_pose",
    "07_geometry/joints_camera",
    "07_geometry/joints_world",
    "07_geometry/vertices_camera",
    "07_geometry/vertices_world",
    "08_public_result/joints_camera",
    "08_public_result/joints_world",
    "08_public_result/poses_axis_angle",
    "08_public_result/root_translation_world",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("exact", "gem-x"),
        default="exact",
        help=(
            "exact requires bitwise equality; gem-x keeps every field exact "
            "except the audited CUDA contact-postprocess fields"
        ),
    )
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--atol", type=float, default=0.0)
    args = parser.parse_args()

    reference = MonocularParityTrace.load(args.reference)
    candidate = MonocularParityTrace.load(args.candidate)
    field_atol = (
        {field: GEM_X_POSTPROCESS_ATOL for field in GEM_X_POSTPROCESS_FIELDS}
        if args.profile == "gem-x"
        else None
    )
    report = reference.compare(
        candidate,
        rtol=args.rtol,
        atol=args.atol,
        field_atol=field_atol,
    )
    if report.exact:
        print(
            f"PASS: {report.compared_fields} fields match across "
            f"{len(reference.stage_names)} stages under {args.profile!r} policy."
        )
        return
    for mismatch in report.mismatches:
        print(
            f"FAIL {mismatch.stage}/{mismatch.field}: {mismatch.reason}",
            file=sys.stderr,
        )
    raise SystemExit(1)


if __name__ == "__main__":
    main()
