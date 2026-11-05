"""Independently verify ProtoMotions replay, GAE, EMA and minibatch ordering."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from verify_prompthmr_video_parity import tensors, compare_full


def worker(args):
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(861)
    np.random.seed(861)
    if args.worker == 'upstream':
        sys.path.insert(0, str(args.upstream))
        from lightning.fabric import Fabric
        from protomotions.agents.ppo.agent import PPO
        from protomotions.agents.amp.component import AMPTrainingComponent
        from protomotions.agents.utils.data import ExperienceBuffer, DictDataset
        from protomotions.agents.utils.replay_buffer import ReplayBuffer

        config = torch.load(args.artifact / 'resolved_configs.pt', weights_only=False)[
            'agent'
        ]
        config.amp_parameters.discriminator_replay_keep_prob = 0.7
        agent = object.__new__(PPO)
        agent.config = config
        agent.device = torch.device('cuda')
        agent.fabric = Fabric(accelerator='cuda', devices=1, precision='32-true')
        agent.gamma = 0.99
        agent.tau = 0.95
        agent.adv_mean_ema = torch.zeros(1, device='cuda')
        agent.adv_std_ema = torch.ones(1, device='cuda')
        component = object.__new__(AMPTrainingComponent)
        component.agent = agent
        component.use_disc_critic = True
        component.replay_buffer = ReplayBuffer(13, device=torch.device('cuda'))
        # Bind the same original PPO -> AMP advantage composition. No objective
        # or normalization arithmetic is replaced in this reference path.
        agent.compute_advantages = lambda: component.add_advantages(
            PPO.compute_advantages(agent)
        )
    else:
        from motius.datasets.protomotions_online import (
            AMPReplay,
            ProtoMotionsMinibatches,
        )
        from motius.simulators.rl.protomotions_advantages import (
            ProtoMotionsAdvantageNormalization,
            task_style_advantages,
        )

        normalization = ProtoMotionsAdvantageNormalization('cuda')
        replay = AMPReplay(13, 'cuda')
    results = {}
    for iteration in range(3):
        torch.manual_seed(862 + iteration)
        values = {
            name: torch.randn(32, 4, 1, device='cuda')
            for name in ['value', 'next_value', 'disc_value', 'next_disc_value']
        }
        values.update(
            rewards=torch.randn(32, 4, device='cuda') * (iteration + 1) * 4,
            amp_rewards=torch.randn(32, 4, device='cuda') * (iteration + 1) * 2,
            dones=(torch.rand(32, 4, device='cuda') < 0.13).long(),
        )
        if args.worker == 'upstream':
            buffer = ExperienceBuffer(4, 32, torch.device('cuda'))
            for name, value in values.items():
                buffer.register_key(name, shape=value.shape[2:], dtype=value.dtype)
                buffer.batch_update_data(name, value)
            for name in ['returns', 'disc_returns', 'advantages']:
                buffer.register_key(name)
            agent.experience_buffer = buffer
            agent.pre_process_dataset()
            result = {
                name: getattr(buffer, name)
                for name in ['returns', 'disc_returns', 'advantages']
            }
            result['mean'] = agent.adv_mean_ema
            result['std'] = agent.adv_std_ema
        else:
            result = task_style_advantages(values, normalization)
            result.update(mean=normalization.mean, std=normalization.std)
        results['advantages/' + str(iteration)] = tensors(result)
    # Capacity overflow, circular writes, full-buffer thinning and oversize
    # insertion use both CPU and CUDA RNG streams in the original implementation.
    for index, size in enumerate([7, 9, 31, 5]):
        torch.manual_seed(880 + index)
        values = {
            'one': torch.randn(size, 3, device='cuda'),
            'two': torch.arange(size, device='cuda')[:, None],
        }
        if args.worker == 'upstream':
            component.update_disc_replay_buffer(dict(values))
            buffer = component.replay_buffer
            result = dict(
                head=buffer._head,
                full=buffer._is_full,
                length=len(buffer),
                buffers={name: getattr(buffer, name) for name in buffer._buffer_keys},
                sample=buffer.sample(17),
            )
        else:
            replay.update(values, keep_probability=0.7)
            result = dict(
                head=replay.head,
                full=replay.full,
                length=len(replay),
                buffers=replay.buffers,
                sample=replay.sample(17),
            )
        results['replay/' + str(index)] = tensors(
            dict(
                **result,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            )
        )
    # Expert data can be shorter than the rollout; preserve modulo indexing and
    # NumPy reshuffling at the next mini epoch.
    torch.manual_seed(899)
    np.random.seed(899)
    values = dict(
        rollout=torch.randn(24, 5, device='cuda'),
        expert=torch.randn(7, 3, device='cuda'),
    )
    if args.worker == 'upstream':
        dataset = DictDataset(4, values, shuffle=True)
        batches = []
        for index in range(12):
            offset = index % len(dataset)
            if offset == 0 and index != 0:
                dataset.shuffle()
            batches.append(tensors(dataset[offset]))
    else:
        batches = [tensors(batch) for _, batch in ProtoMotionsMinibatches(values, 4, 2)]
    results['minibatches'] = batches
    results['final_cpu_rng'] = torch.get_rng_state()
    results['final_cuda_rng'] = torch.cuda.get_rng_state()
    torch.save(results, args.output_dir / (args.worker + '.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'artifact', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', choices=['upstream', 'native'])
    args = parser.parse_args()
    for name in ['upstream', 'artifact', 'output_dir']:
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / 'source_parity.json'
    report = dict(
        complete=False,
        scope='Actual original task/style GAE, EMA advantage normalization, circular AMP replay and thinning, CPU/CUDA RNG and NumPy-shuffled modulo minibatches versus native data components. Synthetic tensors; not online neural collection, physical simulation or full training protocol.',
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native']:
        (args.output_dir / (kind + '.pt')).unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, '-u', __file__, *sys.argv[1:], '--worker', kind],
            cwd=args.upstream if kind == 'upstream' else root,
            env={**os.environ, 'PYTHONPATH': str(root)},
            check=True,
        )
    import torch

    torch.set_num_threads(4)
    try:
        compare_full(
            torch.load(args.output_dir / 'upstream.pt', weights_only=True),
            torch.load(args.output_dir / 'native.pt', weights_only=True),
            '',
            report['tests'],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Original ProtoMotions GAE, EMA, replay and shuffled minibatches exact')


if __name__ == '__main__':
    main()
