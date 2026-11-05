"""Compare native InterGen sampling with an independent, unmodified source checkout.

The source worker substitutes only clip.load's unused visual tower construction:
its text tower is instantiated with official CLIP layers, then every parameter is
strictly replaced from the same released InterGen checkpoint as the native worker.
No forward computation, sampler, or source loss implementation is substituted.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source, artifact = args.upstream.resolve(), args.artifact.resolve()
    if args.worker == 'upstream':
        import clip
        from clip.model import Transformer, LayerNorm
        from types import SimpleNamespace
        from torch import nn
        def text_tower(*a, **kw):
            return SimpleNamespace(token_embedding=nn.Embedding(49408, 768),
                transformer=Transformer(768, 12, 12,
                    attn_mask=torch.empty(77, 77).fill_(float('-inf')).triu_(1)),
                positional_embedding=nn.Parameter(torch.empty(77, 768)),
                ln_final=LayerNorm(768), dtype=torch.float32), None
        clip.load = text_tower
        sys.path.insert(0, str(source))
        from configs import get_config
        from models import InterGen
        cfg = get_config(str(source / 'configs/model.yaml'))
        model = InterGen(cfg)
        model.load_state_dict(load_file(str(artifact / 'model.safetensors')), strict=True)
        model = model.to(args.device).eval()
        mean = torch.from_numpy(np.load(artifact / 'global_mean.npy')).float().to(args.device)
        std = torch.from_numpy(np.load(artifact / 'global_std.npy')).float().to(args.device)
    else:
        from motius.pipelines.intergen import InterGenPipeline
        pipeline = InterGenPipeline.from_pretrained(str(artifact), bundle_kwargs={'device': args.device})
        model = pipeline.bundle.network
    output = {}
    for seed, length, prompts in [(13, 30, ['two people shake hands']),
                                   (19, 60, ['two people hug', 'two people kick each other'])]:
        key = f'seed{seed}_frames{length}_batch{len(prompts)}'
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            if args.worker == 'upstream':
                batch = dict(text=prompts, motion_lens=torch.full((len(prompts),), length,
                             dtype=torch.long, device=args.device))
                result = model.forward_test(batch)['output'].reshape(len(prompts), length, 2, 262)
                result = result * std + mean
            else:
                result = pipeline(prompts, motion_len=length, seed=seed, return_numpy=False)
        output[key] = result.detach().cpu()
    torch.save(output, args.output_dir / f'{args.worker}.pt')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--worker', choices=['native', 'upstream'])
    args = parser.parse_args()
    args.upstream = args.upstream.resolve(); args.artifact = args.artifact.resolve()
    args.output_dir = args.output_dir.resolve(); args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args); return
    report_path = args.output_dir / 'source_parity.json'
    report = dict(complete=False, comparison='original source with released checkpoint',
        upstream_commit=subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        source_files={str(p.relative_to(args.upstream)): digest(p)
                      for folder in ['models', 'utils', 'configs', 'common']
                      for p in (args.upstream / folder).rglob('*.py')},
        artifact_files={name: digest(args.artifact / name)
                        for name in ['model.safetensors','global_mean.npy','global_std.npy']},
        source_setup='Official CLIP text architecture only; strict full checkpoint load; no forward substitutions',
        tests=[])
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    # The original code reads ./data; supply only the released statistics in an isolated cwd.
    work = args.output_dir / 'source-cwd'; (work / 'data').mkdir(parents=True, exist_ok=True)
    for name in ['global_mean.npy', 'global_std.npy']:
        shutil.copyfile(args.artifact / name, work / 'data' / name)
    root = Path(__file__).resolve().parents[1]
    for kind in ['upstream', 'native']:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(root)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), '--upstream', str(args.upstream),
            '--artifact', str(args.artifact), '--output-dir', str(args.output_dir), '--device', args.device,
            '--worker', kind], cwd=work if kind == 'upstream' else root, env=env, check=True)
    import torch
    original = torch.load(args.output_dir / 'upstream.pt', weights_only=True)
    native = torch.load(args.output_dir / 'native.pt', weights_only=True)
    for key in original:
        a,b = original[key],native[key]
        delta = (a-b).abs()
        report['tests'].append(dict(case=key, shape=list(a.shape), bitwise_equal=torch.equal(a,b),
                                   max_abs_error=delta.max().item()))
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    report.update(complete=True, torch_version=torch.__version__,
        native_files={str(p.relative_to(root)):digest(p) for folder in
            ['motius/models/intergen','motius/pipelines/intergen','motius/diffusion/intergen']
            for p in (root/folder).rglob('*.py')})
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['tests'], indent=2))


if __name__ == '__main__': main()
