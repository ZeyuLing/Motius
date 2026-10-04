"""Run the official FlowHMR checkpoint and save Motius monocular capture NPZ."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.pipelines.flowhmr import FlowHMRPipeline
from motius.motion.representation.monocular_capture import save_monocular_capture_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", help="Motius Hub repo or official base/latest checkpoint directory")
    parser.add_argument("video")
    parser.add_argument("--body-model", required=True)
    parser.add_argument("--j-regressor", required=True)
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-config")
    parser.add_argument("--mhr-path")
    parser.add_argument("--yolox-checkpoint")
    parser.add_argument("--work-dir", default="outputs/inference/flowhmr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--postprocess", action="store_true")
    args = parser.parse_args()
    pipe = FlowHMRPipeline.from_pretrained(args.checkpoint,
        bundle_kwargs=dict(body_model_path=args.body_model, j_regressor_path=args.j_regressor,
                           device=args.device), vggt_checkpoint=args.vggt_checkpoint,
        preprocessor_kwargs={k: v for k, v in dict(sam3d_ckpt=args.sam_checkpoint, sam3d_config=args.sam_config,
                                 sam3d_mhr=args.mhr_path, yolox_ckpt=args.yolox_checkpoint).items() if v is not None})
    result = pipe.infer_monocular_motion_capture(args.video, work_dir=args.work_dir,
                                                seeds=[args.seed], postprocess=args.postprocess)
    save_monocular_capture_result(result, Path(args.work_dir) / "result.npz")


if __name__ == "__main__":
    main()
