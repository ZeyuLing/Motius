#!/usr/bin/env python3
"""Export the complete HYMotion-V2M inference artifact for Hugging Face Hub."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.hymotion_v2m import HyMotionV2MBundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--body-model-path", type=Path, required=True)
    parser.add_argument("--sam3d-checkpoint", type=Path, required=True)
    parser.add_argument("--sam3d-mhr", type=Path, required=True)
    parser.add_argument("--sam3d-config", type=Path, required=True)
    parser.add_argument("--yolox-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.expanduser().resolve()
    outputs_root = (ROOT / "outputs").resolve()
    try:
        output.relative_to(outputs_root)
    except ValueError as exc:
        raise ValueError("--output-dir must live under repository outputs/.") from exc

    bundle = HyMotionV2MBundle.from_pretrained(
        str(args.checkpoint_dir.expanduser().resolve()),
        body_model_path=str(args.body_model_path.expanduser().resolve()),
        device="cpu",
    )
    bundle.save_pretrained(
        str(output),
        preprocessor_assets={
            "sam3d_ckpt": str(args.sam3d_checkpoint.expanduser().resolve()),
            "sam3d_mhr": str(args.sam3d_mhr.expanduser().resolve()),
            "sam3d_config": str(args.sam3d_config.expanduser().resolve()),
            "yolox_ckpt": str(args.yolox_checkpoint.expanduser().resolve()),
        },
    )
    print(output)


if __name__ == "__main__":
    main()
