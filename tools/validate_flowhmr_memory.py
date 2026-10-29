"""Measure full-sequence VGGT camera inference under an allocator budget."""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import torch
from motius.pipelines.flowhmr.camera_estimation import extract_vggt_camera


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--video', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=750)
    p.add_argument('--budget-gib', type=float, default=6)
    p.add_argument('--mode', choices=('reference', 'streamed'), default='streamed')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    torch.cuda.set_per_process_memory_fraction(args.budget_gib * 2**30 / torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    camera = extract_vggt_camera(args.video, expected_len=args.frames, frame_range=(0, args.frames-1),
        checkpoint_path=json.loads(args.assets.read_text())['preprocessor/vggt_omega_1b_512.pt'],
        device='cuda', cache_model=False, memory_mode=args.mode)
    torch.cuda.synchronize()
    np.savez(args.output/'camera.npz', camera_RT=camera.camera_RT, camera_K=camera.K)
    report = dict(seconds=time.perf_counter()-start, frames=len(camera.K), camera_interval=1,
        memory_mode=args.mode, budget_gib=args.budget_gib, gpu=torch.cuda.get_device_name(),
        torch=torch.__version__, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(), host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (args.output/'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
