"""Run the official FlowHMR checkpoint and save Motius monocular capture NPZ."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motius.pipelines.flowhmr import FlowHMRPipeline
from motius.motion.representation.monocular_capture import save_monocular_capture_result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", nargs="?", default="ZeyuLing/Motius-FlowHMR-Latest",
                        help="Optional Hub repo or base/latest directory (default: Motius Latest)")
    parser.add_argument("video")
    parser.add_argument("--body-model", help="Licensed neutral SMPL-H NPZ; defaults to MOTIUS_SMPLH_MODEL or body_models")
    parser.add_argument("--j-regressor", default=os.environ.get('MOTIUS_J_REGRESSOR'))
    parser.add_argument("--cache-dir", help="Hugging Face download/cache directory")
    parser.add_argument("--mode", choices=("standard", "lightweight", "reference"), default="standard",
                        help="standard uses official SAM with shared optimizations; lightweight uses Fast SAM; reference uses the old original scheduling")
    parser.add_argument("--lightweight", action="store_true",
                        help="Legacy alias for --mode lightweight")
    parser.add_argument("--sam-backend", choices=('auto', 'official', 'fast'), default='auto',
                        help="Override the mode's SAM frontend (official or Fast)")
    parser.add_argument("--attention-query-chunk-size", type=int, default=256,
                        help="Query batch size for --lightweight; smaller uses less attention memory")
    parser.add_argument("--sam-frame-batch-size", type=int, default=None,
                        help="Independent frames per SAM batch; optimized modes default to 8, use 1 for the previous schedule")
    parser.add_argument("--vggt-attention-backend", choices=("auto", "torch", "cudnn"), default="auto",
                        help="Streamed VGGT exact attention kernel: auto measures installed kernels, torch preserves the previous choice")
    parser.add_argument("--vggt-memory-mode", choices=("auto", "reference", "streamed"), default="auto",
                        help="auto stages VGGT modules in lightweight mode while retaining global attention")
    parser.add_argument("--frontend-precision", choices=("source", "float16", "float32"), default="source",
                        help="source follows upstream precision; float16 selects mixed FP16 frontend arithmetic; float32 selects comparison arithmetic")
    parser.add_argument("--vggt-frame-batch-size", type=int, default=8)
    parser.add_argument("--vggt-query-chunk-size", type=int, default=4096)
    parser.add_argument("--vggt-token-chunk-size", type=int, default=4096)
    parser.add_argument("--vggt-checkpoint")
    parser.add_argument("--sam-checkpoint")
    parser.add_argument("--sam-config")
    parser.add_argument("--mhr-path")
    parser.add_argument("--yolox-checkpoint")
    parser.add_argument("--work-dir", default="outputs/inference/flowhmr")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cpu-threads", type=int, default=min(4, os.cpu_count() or 1),
                        help="Bound CPU threads for inference startup (default: up to 4)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Process only the first N frames of the input video")
    parser.add_argument("--vggt-frame-interval", type=int, default=1,
                        help="Sample every Nth frame for VGGT camera estimation")
    parser.add_argument("--postprocess", action="store_true")
    parser.add_argument("--profile", action="store_true",
                        help="Save total runtime and per-process peak CUDA allocation beside result.npz")
    args = parser.parse_args()
    if args.lightweight:
        args.mode = "lightweight"
    staged = args.mode != "reference"
    if args.sam_backend == "auto":
        args.sam_backend = "fast" if args.mode == "lightweight" else "official"
    if args.sam_frame_batch_size is None:
        args.sam_frame_batch_size = 8 if staged else 1
    if min(args.vggt_frame_batch_size, args.vggt_query_chunk_size, args.vggt_token_chunk_size) < 1:
        parser.error("VGGT batch sizes must be positive")
    if args.sam_frame_batch_size is not None and args.sam_frame_batch_size < 1:
        parser.error("--sam-frame-batch-size must be positive")
    if args.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive")
    if args.vggt_frame_interval < 1:
        parser.error("--vggt-frame-interval must be positive")
    if args.attention_query_chunk_size < 1:
        parser.error("--attention-query-chunk-size must be positive")
    from motius.pipelines.flowhmr.lightweight import resolve_body_model
    args.body_model = resolve_body_model(args.body_model)
    import torch
    torch.set_num_threads(args.cpu_threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.frontend_precision == 'float32':
        torch.backends.cudnn.allow_tf32 = False
    profile_cuda = args.profile and torch.device(args.device).type == 'cuda' and torch.cuda.is_available()
    if profile_cuda:
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    pipe = FlowHMRPipeline.from_pretrained(args.checkpoint,
        bundle_kwargs=dict(body_model_path=args.body_model, j_regressor_path=args.j_regressor,
                           device=args.device, cache_dir=args.cache_dir), vggt_checkpoint=args.vggt_checkpoint,
        profile=args.profile, lightweight=staged, sam_backend=args.sam_backend, vggt_memory_mode=args.vggt_memory_mode, vggt_attention_backend=args.vggt_attention_backend, sam_frame_batch_size=args.sam_frame_batch_size, attention_query_chunk_size=args.attention_query_chunk_size,
        frontend_precision=args.frontend_precision, vggt_frame_batch_size=args.vggt_frame_batch_size,
        vggt_query_chunk_size=args.vggt_query_chunk_size, vggt_token_chunk_size=args.vggt_token_chunk_size,
        preprocessor_kwargs={k: v for k, v in dict(sam3d_ckpt=args.sam_checkpoint, sam3d_config=args.sam_config,
                                 sam3d_mhr=args.mhr_path, yolox_ckpt=args.yolox_checkpoint).items() if v is not None})
    if profile_cuda:
        torch.cuda.synchronize(args.device)
    model_loading_seconds = time.perf_counter()-started
    result = pipe.infer_monocular_motion_capture(args.video, work_dir=args.work_dir,
                                                max_frames=args.max_frames,
                                                vggt_frame_interval=args.vggt_frame_interval,
                                                seeds=[args.seed], postprocess=args.postprocess)
    output = Path(args.work_dir) / "result.npz"
    save_monocular_capture_result(result, output)
    if args.profile:
        if profile_cuda:
            torch.cuda.synchronize(args.device)
        profile = dict(cpu_threads=torch.get_num_threads(), motion_tf32=False, cudnn_tf32=torch.backends.cudnn.allow_tf32, model_loading_seconds=model_loading_seconds, stages=pipe.stage_timings,
                       total_seconds_including_model_loading=time.perf_counter() - started,
                       mode=args.mode, lightweight=args.mode == "lightweight",
                       inference_settings=result.metadata['inference_settings'])
        if profile_cuda:
            profile.update(gpu=torch.cuda.get_device_name(args.device),
                           peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                           peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        (Path(args.work_dir) / 'inference_profile.json').write_text(json.dumps(profile, indent=2)+'\n')
    print(f'Saved FlowHMR motion to {output}')


if __name__ == "__main__":
    main()
