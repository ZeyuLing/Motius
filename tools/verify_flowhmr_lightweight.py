"""Summarize paired fresh-process runs from benchmark_flowhmr_lightweight.py.

Expected names: attention_{30,2100,3789}_{reference,light}, sam_30_*,
video_30_* and latest_cfg2_*. Reports measured errors, not universal guarantees.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F

from motius.motion.representation.flowhmr.geometry import rot6d_to_rotation_matrix
from motius.motion.representation.monocular_capture import load_monocular_capture_result


def rotation_error(left, right):
    a, b = (rot6d_to_rotation_matrix(x.double()) for x in (left, right))
    r = a.transpose(-1, -2) @ b
    sine = torch.stack((r[..., 2, 1]-r[..., 1, 2], r[..., 0, 2]-r[..., 2, 0],
                        r[..., 1, 0]-r[..., 0, 1]), -1).norm(dim=-1)/2
    degrees = torch.atan2(sine, (r.diagonal(dim1=-2, dim2=-1).sum(-1)-1)/2).rad2deg()
    return dict(max_rotation_degrees=degrees.max().item(), mean_rotation_degrees=degrees.mean().item())


def motion_error(a, b, translation='trans'):
    for values in (a, b):
        for key in ('rot6d', translation):
            if not torch.isfinite(values[key]).all():
                raise ValueError(f'Non-finite {key}')
    return dict(rotation_error(a['rot6d'], b['rot6d']),
                max_root_translation_mm=(a[translation]-b[translation]).norm(dim=-1).max().item()*1000,
                max_rot6d_absolute_error=(a['rot6d']-b['rot6d']).abs().max().item())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = dict(source_issue='https://github.com/flowhmr/flowhmr/issues/4',
                  note='Single H200 runs. Long sampling repeats measured 30-frame conditioning; '
                       'it is not a long-video end-to-end benchmark. Lightweight is numerically, '
                       'not bitwise, equivalent on the tested real video.', cases={})
    for name in ('attention_30', 'attention_2100', 'attention_3789', 'latest_cfg2', 'sam_30', 'video_30'):
        folders = [args.runs / f'{name}_{suffix}' for suffix in ('reference', 'light')]
        metric_name = 'inference_profile.json' if name == 'video_30' else 'benchmark.json'
        metrics = [json.loads((p / metric_name).read_text()) for p in folders]
        case = dict(reference=metrics[0], lightweight=metrics[1])
        if name == 'video_30':
            tracks = [load_monocular_capture_result(p / 'result.npz').tracks[0] for p in folders]
            values = [dict(rot6d=torch.from_numpy(t.native_parameters['rot6d']),
                           trans=torch.from_numpy(t.root_translation_world)) for t in tracks]
            case.update(motion_error(*values))
        else:
            values = [torch.load(p / 'output.pt', map_location='cpu', weights_only=True) for p in folders]
            if name == 'sam_30':
                if not all(torch.isfinite(v).all() for v in values):
                    raise ValueError('Non-finite SAM features')
                case['branches'] = {}
                for index, branch in enumerate(('body_first', 'left_hand', 'right_hand')):
                    a, b = [v[..., index*1024:(index+1)*1024].double() for v in values]
                    case['branches'][branch] = dict(max_absolute_error=(a-b).abs().max().item(),
                        min_cosine_similarity=F.cosine_similarity(a, b, dim=-1).min().item())
            else:
                if metrics[0]['checkpoint_sha256'] != metrics[1]['checkpoint_sha256']:
                    raise ValueError('Checkpoint mismatch')
                case.update(motion_error(*values))
                case['all_returned_tensors_equal'] = all(torch.equal(values[0][k], values[1][k]) for k in values[0])
        report['cases'][name] = case
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'Saved measured comparisons to {args.output}')


if __name__ == '__main__':
    main()
