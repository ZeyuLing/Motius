"""Synthetic IK regression/timing harness, NOT paper evaluation data.

python tools/benchmark_position_ik.py --device cpu --frames 120 --repeats 5
python tools/benchmark_position_ik.py --device cuda --frames 300 --repeats 20

End-to-end IK timing includes conversion, projection and final FK verification;
excludes model inference, setup, metrics below, and printing. GPU synchronized.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from motius.motion.pipeline_utils.position_constraint import PositionConstraint, PositionConstraintSolver
from motius.motion.pipeline_utils.position_ik import _rotation_exp
from motius.motion.skeleton.fk import differentiable_fk, fk_to_motion135, motion135_to_fk


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.frames < 2 or args.repeats < 1 or args.threads < 1:
        parser.error('frames >= 2, repeats >= 1 and threads >= 1 required')
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    device = torch.device(args.device)
    offsets = torch.zeros(22, 3, device=device)
    offsets[1] = torch.tensor([.1, -.1, 0.], device=device)
    offsets[2] = torch.tensor([-.1, -.1, 0.], device=device)
    offsets[[3, 6, 9, 12, 15], 1] = .1
    offsets[[4, 5, 7, 8], 1] = -.4
    offsets[[10, 11], 2] = .15
    offsets[[13, 16], 0] = .12
    offsets[[14, 17], 0] = -.12
    offsets[[18, 20], 0] = .3
    offsets[[19, 21], 0] = -.3
    t = torch.linspace(0, 2 * torch.pi, args.frames, device=device)
    phase = torch.arange(22, device=device)[None, :, None] / 7
    vectors = .2 * torch.sin(t[:, None, None] + phase + torch.arange(3, device=device))
    reference = _rotation_exp(vectors)
    raw_q = reference @ _rotation_exp(torch.randn_like(vectors) * .08)
    root = torch.zeros(args.frames, 3, device=device)
    targets = differentiable_fk(reference, root, offsets)[0]
    raw = fk_to_motion135(raw_q, root)
    cases = {
        'dense_wrists': [(f, j) for f in range(args.frames) for j in (20, 21)],
        'sparse_wrists': [(f, j) for f in range(0, args.frames, 30) for j in (20, 21)],
        'dense_lower_body': [(f, j) for f in range(args.frames) for j in (1, 2, 4, 5, 7, 8, 10, 11)],
        'mixed_synthetic_not_P1_P4': sorted(set(
            [(f, j) for f in range(0, args.frames, 15) for j in range(22)] +
            [(f, j) for f in range(args.frames) for j in (0, 20, 21)])),
    }
    solver = PositionConstraintSolver(offsets)

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)

    report = {'kind': 'synthetic engineering benchmark; not paper results',
              'torch': torch.__version__, 'device': str(device), 'frames': args.frames,
              'repeats': args.repeats, 'threads': args.threads, 'rows': []}
    for name, entries in cases.items():
        cues = [PositionConstraint(f, j, targets[f, j]) for f, j in entries]
        solver.solve(raw, cues)  # warmup, same inputs for each repetition
        times = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            output, returned_error = solver.solve(raw, cues)
            sync()
            times.append(time.perf_counter() - start)
        actual = motion135_to_fk(output, offsets)[0]
        errors = torch.stack([(actual[f, j] - targets[f, j]).norm() for f, j in entries]) * 1000
        row = {'task': name, 'cues': len(cues), 'ik_seconds_mean': statistics.mean(times),
               'ik_seconds_median': statistics.median(times), 'mean_error_mm': float(errors.mean()),
               'cue_p95_mm': float(torch.quantile(errors, .95)), 'max_error_mm': float(errors.max()),
               'all_cues_within_0.1mm': bool((errors <= .1).all()),
               'returned_max_error_mm': returned_error * 1000}
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
