"""Measure one FlowHMR execution path in a fresh process.

Run reference and --lightweight separately on the same GPU and cached inputs.
SAM timing excludes loading and has one warm-up frame. Attention timing excludes
loading and warms up on at most 30 frames. Video mode runs the public CLI and
includes loading. This tool does not claim full-training or motion-quality scores.
"""
import argparse
import gc
import json
from pathlib import Path
import runpy
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def measure(function):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return result, dict(seconds=elapsed, baseline_allocated_bytes=baseline,
                        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                        peak_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['attention', 'sam', 'video'], required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--body-model', type=Path, required=True)
    parser.add_argument('--frontend-cache', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=30)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--cfg-scale', type=float, default=1.0,
                        help='Classifier-free guidance for the attention stage')
    parser.add_argument('--lightweight', action='store_true')
    parser.add_argument('--query-chunk-size', type=int, default=256)
    args = parser.parse_args()
    if min(args.frames, args.steps, args.query_chunk_size) < 1:
        parser.error('frames, steps and query chunk size must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError('This benchmark requires CUDA')
    torch.cuda.init()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)

    # Give ordinary context-aware idle-load supervisors time to yield. Do not
    # stop other jobs or change their configuration to run a benchmark.
    torch.zeros(1, device='cuda')
    time.sleep(3)
    if args.stage == 'video':
        entry = Path(__file__).with_name('infer_flowhmr.py')
        sys.argv = [str(entry), str(args.checkpoint), str(args.video),
                    '--body-model', str(args.body_model), '--work-dir', str(args.output_dir),
                    '--max-frames', str(args.frames), '--seed', str(args.seed), '--profile']
        if args.lightweight:
            sys.argv.extend(['--lightweight', '--attention-query-chunk-size', str(args.query_chunk_size)])
        runpy.run_path(str(entry), run_name='__main__')
        return

    cached = torch.load(args.frontend_cache, map_location='cpu', weights_only=True)
    if args.stage == 'sam':
        import cv2
        from motius.pipelines.flowhmr.preprocess import FlowHMRVideoPreprocessor
        assets = args.checkpoint / 'preprocessor'
        pre = FlowHMRVideoPreprocessor(device='cuda', sam_token_only=args.lightweight,
                  sam3d_ckpt=str(assets / 'sam/model.ckpt'),
                  sam3d_config=str(assets / 'sam/model_config.yaml'),
                  sam3d_mhr=str(assets / 'sam/assets/mhr_model.pt'))
        extractor = pre.build_sam_extractor()
        cap = cv2.VideoCapture(str(args.video))
        ok, image = cap.read(); cap.release()
        if not ok:
            raise ValueError('Could not read warm-up video frame')
        extractor.extract_frame(cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                  bboxes=cached['bboxes'][:1].numpy(), cam_int=cached['camera_K'][:1])
        def infer():
            return pre.extract_features(str(args.video), cached['bboxes'], cached['camera_K'], max_frames=args.frames)
        try:
            result, metrics = measure(infer)
        finally:
            pre.release()
    else:
        from motius.pipelines.flowhmr import FlowHMRPipeline
        pipe = FlowHMRPipeline.from_pretrained(str(args.checkpoint),
                   bundle_kwargs=dict(body_model_path=str(args.body_model), device='cuda'),
                   lightweight=args.lightweight, attention_query_chunk_size=args.query_chunk_size)
        pipe.sampler.validation_steps = args.steps
        count = len(cached['features'])
        repeats = (args.frames + count - 1) // count
        feature = cached['features'].repeat(repeats, 1)[:args.frames]
        camera = cached['camera_RT'].repeat(repeats, 1, 1)[:args.frames]
        with torch.inference_mode():
            pipe.infer_from_feature(feature[:min(30, args.frames)], camera[:min(30, args.frames)], seeds=[args.seed], cfg_scale=args.cfg_scale)
            result, metrics = measure(lambda: pipe.infer_from_feature(feature, camera, seeds=[args.seed], cfg_scale=args.cfg_scale))
        metrics.update(checkpoint_sha256=pipe.bundle.checkpoint_sha256,
                       steps=args.steps, seed=args.seed, cfg_scale=args.cfg_scale,
                       repeated_cached_features=args.frames > count, live_sam_or_vggt=False)
    torch.save(result, args.output_dir / 'output.pt')
    metrics.update(stage=args.stage, frames=args.frames, lightweight=args.lightweight,
                   query_chunk_size=args.query_chunk_size if args.lightweight else 0,
                   gpu=torch.cuda.get_device_name(), torch_version=str(torch.__version__),
                   full_hand_features=True, sampler_postprocess=False, runs=1)
    (args.output_dir / 'benchmark.json').write_text(json.dumps(metrics, indent=2)+'\n')
    print(json.dumps(metrics), flush=True)


if __name__ == '__main__':
    main()
