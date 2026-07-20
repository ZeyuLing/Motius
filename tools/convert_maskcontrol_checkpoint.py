#!/usr/bin/env python3
"""Convert released MaskControl weights to a self-contained Motius artifact."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from motius.models.maskcontrol import MaskControlBundle


CONTROL_NAME = "z2024-08-27-21-07-55_CtrlNet_randCond1-196_l1.5XEnt.5TTT__cross"
VQ_NAME = "rvq_nq6_dc512_nc512_noshare_qdp0.2"
RESIDUAL_NAME = "tres_nlayer8_ld384_ff1024_rvq6ns_cdp0.2_sw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        required=True,
        help="Directory containing the extracted official model folders.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-length-estimator", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.raw_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    vq_dir = root / VQ_NAME
    control_dir = root / CONTROL_NAME
    residual_dir = root / RESIDUAL_NAME
    length_path = root / "length_estimator" / "model" / "finest.tar"

    required = {
        "control": control_dir / "model" / "latest.tar",
        "vq": vq_dir / "model" / "net_best_fid.tar",
        "residual": residual_dir / "model" / "net_best_fid.tar",
        "mean": vq_dir / "meta" / "mean.npy",
        "std": vq_dir / "meta" / "std.npy",
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("missing official MaskControl files:\n" + "\n".join(missing))

    bundle = MaskControlBundle(
        control_weights_path=str(required["control"]),
        vq_weights_path=str(required["vq"]),
        residual_weights_path=str(required["residual"]),
        length_weights_path=(
            None
            if args.no_length_estimator or not length_path.exists()
            else str(length_path)
        ),
        mean_path=str(required["mean"]),
        std_path=str(required["std"]),
        raw_control_checkpoint=True,
        load_length_estimator=not args.no_length_estimator,
        device=args.device,
    )
    bundle.save_pretrained(str(output), safe_serialization=True, include_clip=True)
    print(output)


if __name__ == "__main__":
    main()
