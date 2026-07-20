#!/usr/bin/env python3
"""Build the shared HumanML3D TM2T evaluation population manifest."""

from __future__ import annotations

import argparse
import json
from collections import Counter

from motius.evaluation.m2t import (
    load_humanml3d_m2t_samples,
    write_humanml3d_m2t_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split-file", default="test.txt")
    parser.add_argument("--io-workers", type=int, default=32)
    args = parser.parse_args()

    samples = load_humanml3d_m2t_samples(
        args.data_root,
        args.split_file,
        io_workers=args.io_workers,
    )
    output = write_humanml3d_m2t_manifest(
        samples,
        args.output,
        data_root=args.data_root,
        split_file=args.split_file,
    )
    summary = {
        "output": str(output.resolve()),
        "num_samples": len(samples),
        "num_sources": len({sample.source_id for sample in samples}),
        "full_motion": sum(sample.sample_id == sample.source_id for sample in samples),
        "temporal_subclips": sum(sample.sample_id != sample.source_id for sample in samples),
        "reference_counts": dict(
            sorted(Counter(len(sample.captions) for sample in samples).items())
        ),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
