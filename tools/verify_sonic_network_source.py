"""Compare the released SONIC neural roots with the independent original code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from verify_prompthmr_video_parity import compare_full, tensors


SCHEMA = {
    'encoder_index': (3,),
    'command_multi_future_nonflat': (10, 58),
    'motion_anchor_ori_b_mf_nonflat': (10, 6),
    'command_multi_future_lower_body': (240,),
    'vr_3point_local_target': (9,),
    'vr_3point_local_orn_target': (12,),
    'motion_anchor_ori_b': (6,),
    'smpl_joints_multi_future_local_nonflat': (10, 69),
    'smpl_root_ori_b_multi_future': (10, 6),
    'joint_pos_multi_future_wrist_for_smpl': (10, 9),
}
FEATURES = {
    'g1': ['command_multi_future_nonflat', 'motion_anchor_ori_b_mf_nonflat'],
    'teleop': [
        'command_multi_future_lower_body',
        'vr_3point_local_target',
        'vr_3point_local_orn_target',
        'motion_anchor_ori_b',
    ],
    'smpl': [
        'smpl_joints_multi_future_local_nonflat',
        'smpl_root_ori_b_multi_future',
        'joint_pos_multi_future_wrist_for_smpl',
    ],
}


def original_roots(
    upstream, checkpoint, configuration, schema=SCHEMA, *, include_auxiliary=False
):
    import torch
    from omegaconf import OmegaConf

    sys.path.insert(0, str(upstream))
    # The same compatibility import used by the original SONIC trainer.
    import trl.trainer.ppo_config
    from trl.experimental.ppo import ppo_trainer
    import trl.trainer.utils

    trl.trainer.utils.OnlineTrainerState = ppo_trainer.OnlineTrainerState
    states = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    from gear_sonic.trl.modules.actor_critic_modules import Actor, Critic
    from types import SimpleNamespace

    text = configuration.read_text()
    for before, after in [
        ('groot.rl.trl.', 'gear_sonic.trl.'),
        ('groot.rl.envs.', 'gear_sonic.envs.'),
        ('groot.rl.utils.', 'gear_sonic.utils.'),
        ('groot.rl.agents.modules.modules.', 'gear_sonic.trl.modules.base_module.'),
        ('groot.rl.agents.', 'gear_sonic.trl.'),
    ]:
        text = text.replace(before, after)
    config = OmegaConf.create(text)
    algo = config.algo.config
    backbone = algo.actor.backbone.copy()
    # These parameter-free training objectives are outside the inference
    # graph. The original neural architecture and all its tensors remain.
    if not include_auxiliary:
        backbone.aux_loss_func = {}
        backbone.aux_loss_coef = {}
    env = SimpleNamespace(
        obs=SimpleNamespace(
            group_obs_dims={'tokenizer': schema},
            group_obs_names={'tokenizer': list(schema)},
        ),
        robot=SimpleNamespace(
            actions_dim=29, algo_obs_dim_dict={'actor_obs': 930, 'critic_obs': 1645}
        ),
    )
    policy = Actor(
        env, algo, backbone, input_obs_dict=True, has_aux_loss=include_auxiliary
    )
    critic = Critic(env, algo, algo.critic.backbone, running_mean_std=True)
    policy.load_state_dict(states['policy_state_dict'], strict=True)
    critic.load_state_dict(states['value_state_dict'], strict=True)
    policy.cuda()
    critic.cuda()
    return policy, critic


def worker(args):
    import torch
    from omegaconf import OmegaConf

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(501)
    checkpoint = args.artifact / 'sonic_release/last.pt'
    configuration = checkpoint.parent / 'config.yaml'
    if args.worker == 'upstream':
        policy, critic = original_roots(args.upstream, checkpoint, configuration)

        def state():
            result = {}
            for name, value in policy.state_dict().items():
                result['policy.' + (name.removeprefix('actor_module.'))] = value
            for name, value in critic.state_dict().items():
                name = name.replace('critic_module.', 'network.').replace(
                    'running_mean_std.', 'normalizer.'
                )
                result['critic.' + name] = value
            return result

    else:
        from motius.models.sonic import SONICPolicyBundle

        bundle = SONICPolicyBundle.from_original_checkpoint(checkpoint, device='cuda')
        policy = bundle.policy
        critic = bundle.critic
        state = bundle.state_dict
    results = {'loaded_neural_tensors': tensors(state())}
    cases = [
        (1, 1, False, [0]),
        (3, 1, False, [0, 1, 2]),
        (2, 2, True, [0, 1, 2, 2]),
        (2, 2, True, [2, 2, 0, 1]),
    ]
    for index, (batch, sequence, training, modes) in enumerate(cases):
        torch.manual_seed(502 + index)
        policy.train(training)
        critic.train(training)
        observations = {
            name: torch.randn(batch, sequence, *shape, device='cuda') * 0.13
            for name, shape in SCHEMA.items()
        }
        selected = torch.tensor(modes, device='cuda').reshape(batch, sequence)
        observations['encoder_index'] = torch.nn.functional.one_hot(selected, 3).float()
        if index == 3:
            observations['encoder_index'][
                0, 0, 0
            ] = 1  # Source multi-hot SMPL/G1 training mode.
        proprioception = torch.randn(batch, sequence, 930, device='cuda') * 0.13
        critic_obs = torch.randn(batch, sequence, 1645, device='cuda') * 0.13
        if args.worker == 'upstream':
            flat = torch.cat(
                [observations[name].flatten(start_dim=2) for name in SCHEMA], -1
            )
            output = policy.actor_module(
                {'actor_obs': proprioception, 'tokenizer': flat},
                compute_aux_loss=False,
                return_dict=True,
            )
            reconstruction = torch.cat(
                list(output['decoded_outputs']['g1_kin'].values()), -1
            )
            result = dict(
                mean=output['action_mean'],
                reconstruction=reconstruction,
                encoded_tokens=output['encoded_tokens'],
                encoded_latents=output['encoded_latents'],
            )
            value = critic.evaluate({'critic_obs': critic_obs})
        else:
            features = {
                name: torch.cat([observations[key] for key in keys], -1)
                for name, keys in FEATURES.items()
            }
            masks = {
                name: observations['encoder_index'][..., i].reshape(-1).bool()
                for i, name in enumerate(FEATURES)
            }
            output = policy(features, proprioception, masks)
            result = {
                name: output[name]
                for name in [
                    'mean',
                    'reconstruction',
                    'encoded_tokens',
                    'encoded_latents',
                ]
            }
            value = critic(critic_obs)
        result['value'] = value
        results[str(index)] = tensors(
            dict(outputs=result, state=state(), cuda_rng=torch.cuda.get_rng_state())
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
        scope='All 72 released policy/value neural tensors loaded independently using actual original TRL checkpoint classes; original modality encoders/FSQ/dynamic and kinematic decoders/value normalization versus native neural roots, including sequence, mixed-mode and multi-hot inputs. Synthetic prepared observations; no deployment packing, PPO objective, gradients or physical environment certification.',
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        artifact=json.loads((args.artifact / 'provenance.json').read_text()),
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native']:
        expected = args.output_dir / (kind + '.pt')
        expected.unlink(missing_ok=True)
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
    print('Original SONIC neural roots and released checkpoint tensors are exact')


if __name__ == '__main__':
    main()
