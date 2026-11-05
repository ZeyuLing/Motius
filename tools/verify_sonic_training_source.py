"""Original SONIC PPO/auxiliary objectives and gradients versus native Trainer."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from verify_sonic_network_source import SCHEMA, FEATURES, original_roots
from verify_prompthmr_video_parity import tensors, compare_full


def canonical(policy, critic, *, gradients=False):
    def values(module):
        return (
            {name: value.grad for name, value in module.named_parameters()}
            if gradients
            else module.state_dict()
        )

    result = {}
    for name, value in values(policy).items():
        result['policy.' + name.removeprefix('actor_module.')] = value
    for name, value in values(critic).items():
        result[
            'critic.'
            + name.replace('critic_module.', 'network.').replace(
                'running_mean_std.', 'normalizer.'
            )
        ] = value
    return result


def worker(args):
    import torch
    from accelerate import Accelerator
    from omegaconf import OmegaConf
    from types import SimpleNamespace

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(731)
    checkpoint = args.artifact / 'sonic_release/last.pt'
    accelerator = Accelerator(mixed_precision='no')
    if args.worker == 'upstream':
        policy, critic = original_roots(
            args.upstream,
            checkpoint,
            checkpoint.parent / 'config.yaml',
            include_auxiliary=True,
        )
        from gear_sonic.trl.trainer.ppo_trainer import PolicyAndValueWrapper
        from gear_sonic.trl.trainer.ppo_trainer_aux_loss import TRLAuxLossPPOTrainer

        model = PolicyAndValueWrapper(policy, critic)
        trainer = object.__new__(TRLAuxLossPPOTrainer)
        trainer.config = OmegaConf.create({})
        trainer.args = SimpleNamespace(
            cliprange_value=0.2,
            cliprange=0.2,
            vf_coef=1.0,
            learning_rate=2e-5,
            max_grad_norm=0.1,
            fp16=False,
        )
        trainer.accelerator = accelerator
        trainer.use_symmetry = False
        trainer.multi_critic_advantage_weights = None
        trainer.entropy_coef = 0.01
        trainer.desired_kl = 0.01
        trainer.adaptive_lr_min = 1e-5
        trainer.adaptive_lr_max = 2e-4
        trainer.compute_imgaug_bc_loss = False
        trainer.compute_aux_loss = True
        trainer.aux_loss_scale = 1.0
        trainer.model = model
        trainer.use_apex = False
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
        trainer.optimizer = optimizer
    else:
        from motius.models.sonic import SONICPolicyBundle
        from motius.trainers.sonic import SONICPPOTrainer

        model = SONICPolicyBundle.from_original_checkpoint(
            checkpoint, device='cuda'
        ).train()
        trainer = SONICPPOTrainer(model)
        trainer.accelerator = accelerator
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.0)
        trainer.set_optimizers({'policy_value': optimizer})
    results = {}
    for iteration in range(2):
        torch.manual_seed(732 + iteration)
        batch, sequence = 3, 2
        observations = {
            name: torch.randn(batch, sequence, *shape, device='cuda') * 0.12
            for name, shape in SCHEMA.items()
        }
        # G1 must be active on every SMPL sample for the released alignment
        # objective. Include all three pairwise intersections and unmatched G1.
        modes = torch.tensor(
            [[1, 0, 1], [1, 1, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1], [1, 0, 1]],
            device='cuda',
        ).float()
        observations['encoder_index'] = modes.reshape(batch, sequence, 3)
        proprioception = torch.randn(batch, sequence, 930, device='cuda') * 0.12
        critic_obs = torch.randn(batch, sequence, 1645, device='cuda') * 0.12
        prepared = dict(
            features={
                name: torch.cat([observations[key] for key in keys], -1)
                for name, keys in FEATURES.items()
            },
            encoder_masks={name: modes[:, i].bool() for i, name in enumerate(FEATURES)},
            proprioception=proprioception,
            critic_observations=critic_obs,
            actions=torch.randn(batch, sequence, 29, device='cuda') * 0.1,
            old_mean=torch.randn(batch, sequence, 29, device='cuda') * 0.1,
            old_sigma=torch.rand(batch, sequence, 29, device='cuda') * 0.1 + 0.3,
            values=torch.randn(batch, sequence, 1, device='cuda') * 0.1,
            returns=torch.randn(batch, sequence, 1, device='cuda') * 0.1,
            old_logprobs=torch.randn(batch, sequence, device='cuda') * 0.1,
            advantages=torch.randn(batch, sequence, 1, device='cuda') * 0.1,
            padding_mask=torch.tensor(
                [[False, False], [False, True], [True, False]], device='cuda'
            ),
            padding_mask_value=torch.tensor(
                [[False, True], [False, False], [True, False]], device='cuda'
            ),
        )
        torch.manual_seed(749 + iteration)
        if args.worker == 'upstream':
            flat = torch.cat(
                [observations[name].flatten(start_dim=2) for name in SCHEMA], -1
            )
            source_batch = dict(
                mb_obs_dict={
                    'actor_obs': proprioception,
                    'critic_obs': critic_obs,
                    'tokenizer': flat,
                },
                mb_actions=prepared['actions'],
                episode_attnmask=None,
                micro_batch_inds=None,
                **{
                    'mb_' + target: prepared[name]
                    for name, target in [
                        ('old_mean', 'old_mu'),
                        ('old_sigma', 'old_sigma'),
                        ('values', 'values'),
                        ('returns', 'return'),
                        ('old_logprobs', 'logprobs'),
                        ('advantages', 'advantage'),
                        ('padding_mask', 'padding_mask'),
                        ('padding_mask_value', 'padding_mask_p1'),
                    ]
                },
            )
            forward = trainer._forward_model(model, source_batch)
            output = trainer._compute_loss(forward, source_batch)
            result = dict(
                loss=output['loss'],
                ppo=output['ppo_loss_dict'],
                auxiliary=output['aux_loss_dict']['aux_losses_dict'],
                auxiliary_loss=output['aux_loss_dict']['total_aux_loss'],
                value=forward['value_results'],
            )
            learning_rate = trainer.args.learning_rate
        else:
            output = trainer.train_step(prepared)
            result = {
                name: output[name]
                for name in ['loss', 'ppo', 'auxiliary', 'auxiliary_loss', 'value']
            }
            learning_rate = trainer.learning_rate
        optimizer.zero_grad()
        result['loss'].backward()
        if args.worker == 'upstream':
            trainer._gradient_clipping()
            state = canonical(policy, critic)
            gradients = canonical(policy, critic, gradients=True)
        else:
            accelerator.clip_grad_norm_(model.parameters(), 0.1)
            state = model.state_dict()
            gradients = {name: value.grad for name, value in model.named_parameters()}
        results[str(iteration)] = tensors(
            dict(
                losses=result,
                gradients=gradients,
                state=state,
                learning_rate=learning_rate,
                cuda_rng=torch.cuda.get_rng_state(),
            )
        )
        optimizer.step()
        results[str(iteration) + '/update'] = tensors(
            dict(
                state=(
                    canonical(policy, critic)
                    if args.worker == 'upstream'
                    else model.state_dict()
                ),
                optimizer=optimizer.state_dict(),
            )
        )
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
        scope='Released SONIC neural weights; actual original policy/value wrapper, PPO objective, KL rate adaptation, all five default auxiliary losses and gradient clipping versus native Trainer. Two synthetic mixed/multi-hot prepared minibatches and controlled matching AdamW updates. Not rollout collection, full PPO epochs, distributed protocol or physical environment.',
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        artifact=json.loads((args.artifact / 'provenance.json').read_text()),
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native']:
        (args.output_dir / (kind + '.pt')).unlink(missing_ok=True)
        subprocess.run(
            [sys.executable, '-u', __file__, *sys.argv[1:], '--worker', kind],
            cwd=args.upstream if kind == 'upstream' else root,
            env={
                **os.environ,
                'PYTHONPATH': str(root),
                'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
            },
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
    print(
        'Original SONIC PPO and all default auxiliary losses, gradients and controlled AdamW updates exact'
    )


if __name__ == '__main__':
    main()
