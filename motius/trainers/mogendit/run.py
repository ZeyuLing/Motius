"""Single-process / torchrun orchestration for the dedicated MoreDiff trainer.

Data loading is synchronous, preserving crop/mask/noise RNG and exact resume
without worker-prefetch state. Checkpoints include per-rank RNG and data cursors.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, default_collate

from motius.models.mogendit import MoGenDiTBundle
from motius.datasets.mogendit import MoGenDiTDataset
from .trainer import MoGenDiTTrainer


def capture_rng():
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=(state[0], state[1].tolist(), *state[2:]),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state['python'])
    numpy = state['numpy']
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state(state['cuda'].cpu())


def atomic_save(value, path):
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def export_artifact(path, config, state):
    path.mkdir(parents=True, exist_ok=True)
    atomic_save({'model': state}, path / 'model.pt')
    (path / 'mogendit_config.json').write_text(json.dumps(config, indent=2) + '\n')
    (path / 'model_index.json').write_text(json.dumps({
        'pipeline_class': 'motius.pipelines.mogendit.MoGenDiTPipeline',
        'bundle_class': 'motius.models.mogendit.MoGenDiTBundle',
        'tasks': ['motion_repair'],
    }, indent=2) + '\n')


def run(config, work_dir, *, resume=None, init_weights=None, device='auto'):
    config = copy.deepcopy(config)
    if resume and init_weights:
        raise ValueError('resume and init_weights are mutually exclusive')
    settings = config['training']
    for key in ('batch_size','max_steps','save_interval','log_interval'):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f'{key} must be a positive integer')
    for key in ('lr','grad_clip'):
        if not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'{key} must be finite and positive')
    if not 0 <= settings['ema_decay'] < 1 or settings['ema_start_step'] < 0:
        raise ValueError('Require 0 <= ema_decay < 1 and ema_start_step >= 0')
    if not math.isfinite(settings['weight_decay']) or settings['weight_decay'] < 0:
        raise ValueError('weight_decay must be finite and nonnegative')
    if config['model'].get('fps', 30) != 30 and config['trainer'].get('motion_degradation', True):
        raise ValueError('Native motion degradation assumes 30 fps; resample data before training')
    world = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    if device.type == 'cuda':
        device = torch.device('cuda', local_rank if world > 1 else device.index or 0)
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    try:
        return _train(config, Path(work_dir), resume, init_weights, device, rank, world)
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _train(config, work_dir, resume, init_weights, device, rank, world):
    settings = config['training']
    seed = settings['seed'] + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    bundle = MoGenDiTBundle(**config['model']).to(device)
    model = bundle.net
    if init_weights:
        bundle.load_weights(init_weights)
    data_config = dict(config['data'])
    data_config.update(global_pose=bundle.config['global_pose'], fps=bundle.config['fps'])
    data = MoGenDiTDataset(**data_config)
    # DDP broadcasts rank 0 initialization before the EMA snapshot is made.
    if world > 1:
        bundle.net = DistributedDataParallel(model, device_ids=[device.index] if device.type == 'cuda' else None,
                                             broadcast_buffers=False)
    optimizer = torch.optim.AdamW(bundle.trainable_parameters(), lr=settings['lr'],
                                   weight_decay=settings['weight_decay'])
    ema = {k: v.detach().clone() for k,v in model.state_dict().items()}
    trainer = MoGenDiTTrainer(bundle, **config['trainer'])
    progress = SimpleNamespace(global_step=0)
    trainer.runner = progress
    epoch, offset = 0, 0
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=True)
        expected = copy.deepcopy(config)
        previous = copy.deepcopy(state['config'])
        # These scheduling limits do not alter the optimizer trajectory.
        for c in (expected, previous):
            for k in ('max_steps','save_interval','log_interval'):
                c['training'].pop(k)
        if expected != previous or state['world_size'] != world or state['dataset_fingerprint'] != data.fingerprint:
            raise ValueError('Resume requires the same model/training/data configuration, data files and world size')
        model.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        for key in ema:
            ema[key].copy_(state['ema'][key])
        progress.global_step = state['step']
        epoch, offset = state['epoch'], state['offset']
        restore_rng(state['rng_by_rank'][rank])
    if progress.global_step >= settings['max_steps']:
        raise ValueError('max_steps must exceed the checkpoint step')
    if not resume and (work_dir / 'latest.pt').exists():
        raise FileExistsError('work_dir already contains latest.pt; use --resume or a new directory')
    if rank == 0:
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / 'training_config.json').write_text(json.dumps(config, indent=2)+'\n')
    if world > 1:
        dist.barrier()
    sampler = DistributedSampler(data, num_replicas=world, rank=rank, shuffle=True,
                                 seed=settings['seed'], drop_last=False)
    model.train()
    while progress.global_step < settings['max_steps']:
        sampler.set_epoch(epoch)
        order = list(sampler)
        if not 0 <= offset < len(order):
            raise ValueError('Invalid saved data cursor')
        while offset < len(order) and progress.global_step < settings['max_steps']:
            indices = order[offset:offset+settings['batch_size']]
            batch = default_collate([data[i] for i in indices])
            optimizer.zero_grad(set_to_none=True)
            output = trainer.train_step(batch)
            finite = torch.isfinite(output['loss']).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(f'Nonfinite loss at step {progress.global_step}')
            output['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            # Original trainer updates EMA before incrementing its completed-step counter.
            if progress.global_step >= settings['ema_start_step']:
                with torch.no_grad():
                    for key, value in model.state_dict().items():
                        ema[key].mul_(settings['ema_decay']).add_(value.detach(), alpha=1-settings['ema_decay'])
            progress.global_step += 1
            offset += len(indices)
            next_epoch, next_offset = (epoch + 1, 0) if offset == len(order) else (epoch, offset)
            if rank == 0 and progress.global_step % settings['log_interval'] == 0:
                print(json.dumps(dict(step=progress.global_step, loss=float(output['loss'].detach()), **output['log_vars'])), flush=True)
            final = progress.global_step == settings['max_steps']
            if final or progress.global_step % settings['save_interval'] == 0:
                rng = capture_rng()
                rngs = [None] * world
                if world > 1:
                    dist.all_gather_object(rngs, rng)
                else:
                    rngs[0] = rng
                if rank == 0:
                    checkpoint = dict(format_version=1, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                      ema=ema, step=progress.global_step, epoch=next_epoch, offset=next_offset,
                                      config=config, world_size=world, rng_by_rank=rngs,
                                      dataset_fingerprint=data.fingerprint)
                    atomic_save(checkpoint, work_dir / 'latest.pt')
                    # Inference exports use the same bundle format as migrated checkpoints.
                    if final:
                        export_artifact(work_dir / 'model', bundle.config, model.state_dict())
                        export_artifact(work_dir / 'ema', bundle.config, ema)
                if world > 1:
                    dist.barrier()
        epoch, offset = next_epoch, next_offset
    return progress.global_step


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--data-root', nargs='+', type=Path)
    parser.add_argument('--work-dir', required=True, type=Path)
    parser.add_argument('--max-steps', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--init-weights', type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.data_root:
        config['data']['data_roots'] = [str(p.resolve()) for p in args.data_root]
    if args.max_steps is not None:
        config['training']['max_steps'] = args.max_steps
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    run(config, args.work_dir, resume=args.resume, init_weights=args.init_weights, device=args.device)


if __name__ == '__main__':
    main()
