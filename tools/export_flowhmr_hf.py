#!/usr/bin/env python3
"""Export official FlowHMR weights and video frontend in the Motius Hub layout."""
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motius.models.flowhmr import FlowHMRBundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--body-model', required=True)
    p.add_argument('--j-regressor', required=True)
    p.add_argument('--sam-dir', type=Path, required=True)
    p.add_argument('--yolox', required=True)
    p.add_argument('--vggt', required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    args.output_dir.resolve().relative_to((ROOT / 'outputs').resolve())
    bundle = FlowHMRBundle.from_pretrained(args.checkpoint, body_model_path=args.body_model,
                                         j_regressor_path=args.j_regressor)
    bundle.save_pretrained(args.output_dir, source_assets={
        'preprocessor/sam/model.ckpt': args.sam_dir / 'model.ckpt',
        'preprocessor/sam/model_config.yaml': args.sam_dir / 'model_config.yaml',
        'preprocessor/sam/assets/mhr_model.pt': args.sam_dir / 'assets/mhr_model.pt',
        'preprocessor/yolox_l.pth': args.yolox,
        'preprocessor/vggt_omega_1b_512.pt': args.vggt,
    })
    print(args.output_dir)


if __name__ == '__main__':
    main()
