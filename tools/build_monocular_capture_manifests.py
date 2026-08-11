#!/usr/bin/env python3
"""Build publishable indexes for user-supplied licensed HMR benchmarks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.evaluation.monocular_capture import (
    build_3dpw_test_samples,
    build_emdb_samples,
    write_monocular_capture_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--3dpw-root", dest="threedpw_root", type=Path)
    parser.add_argument("--emdb-root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory under outputs/evaluation/monocular_capture/.",
    )
    return parser.parse_args()


def _require_output_contract(path: Path) -> None:
    parts = path.resolve().parts
    if "outputs" not in parts:
        raise ValueError("--output-dir must live under the repository outputs tree.")


def main() -> None:
    args = parse_args()
    if args.threedpw_root is None and args.emdb_root is None:
        raise SystemExit("Provide --3dpw-root and/or --emdb-root.")
    _require_output_contract(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.threedpw_root is not None:
        samples = build_3dpw_test_samples(args.threedpw_root)
        payload = write_monocular_capture_manifest(
            samples,
            args.output_dir / "3dpw_test_manifest.json",
            dataset_license="3DPW license; user-supplied data required",
            source="https://virtualhumans.mpi-inf.mpg.de/3DPW/",
        )
        print(
            f"3DPW: {payload['population']} tracks, "
            f"{payload['total_frames']} frames"
        )
    if args.emdb_root is not None:
        samples = build_emdb_samples(args.emdb_root)
        payload = write_monocular_capture_manifest(
            samples,
            args.output_dir / "emdb_manifest.json",
            dataset_license="EMDB license; approved user download required",
            source="https://eth-ait.github.io/emdb/",
        )
        print(
            f"EMDB: {payload['population']} protocol tracks, "
            f"{payload['total_frames']} frames"
        )


if __name__ == "__main__":
    main()
