#!/usr/bin/env python3
"""Generate the CondMDI control preview through its public Motius API."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius import Pipeline
from tools.generate_omnicontrol_native_demos import _save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()

    pipe = Pipeline.from_pretrained(
        args.artifact,
        bundle_kwargs={"respacing": "ddim100"},
        device=args.device,
    )
    reference_caption = "a person walks forward and waves with the right hand"
    reference = pipe.infer_text_to_motion(
        [reference_caption],
        [args.frames],
        seed=17,
    )[0]
    controlled = pipe.infer_kinematic_motion_control(
        ["text plus a constrained right-wrist trajectory"],
        [reference],
        lengths=[len(reference)],
        control_mode="joints",
        joint_indices=[21],
        seed=23,
    )[0]
    _save(
        args.output,
        controlled,
        "text plus a constrained right-wrist trajectory",
        "condmdi_kinematic_motion_control",
        condition=reference,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
