"""Compare independent RSL-RL learn() with Motius' actual online runner.

The environment is a deterministic tensor fixture, not an Isaac simulator.
Both paths execute real G1 policy networks, collection, GAE and PPO updates.
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from verify_beyondmimic_ppo_source import published_options
from verify_prompthmr_video_parity import tensors, compare_full


def worker(args):
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    class Environment:
        num_envs = 4
        num_actions = 29
        device = 'cuda'

        def __init__(self):
            self.step_index = 0
            self.state = torch.arange(4, device='cuda').float()[:, None] * .01
            self.trace = []

        def get_observations(self):
            obs = torch.sin(self.state + torch.arange(160, device='cuda') * .01)
            critic = torch.cos(self.state + torch.arange(286, device='cuda') * .01)
            return obs, dict(observations=dict(critic=critic))

        def step(self, actions):
            self.step_index += 1
            self.state = torch.tanh(self.state * .97 + actions.mean(-1, keepdim=True) * .02)
            obs, extras = self.get_observations()
            rewards = 1. - actions.square().mean(-1)
            done = torch.tensor([self.step_index % period == 0 for period in [7, 11, 13, 17]], device='cuda')
            extras['time_outs'] = torch.tensor([self.step_index % 11 == 0] * 4, device='cuda')
            self.trace.append(tensors(dict(actions=actions, observations=obs, critic=extras['observations']['critic'],
                                          rewards=rewards, dones=done, time_outs=extras['time_outs'])))
            return obs, rewards, done, extras

    neural, algorithm = published_options(args.upstream)
    env = Environment()
    snapshots = []
    def snapshot(policy, actor_norm, critic_norm, optimizer, learning_rate, rollout):
        snapshots.append(tensors(dict(state=policy.state_dict(), actor_norm=actor_norm.state_dict(),
            critic_norm=critic_norm.state_dict(), optimizer=optimizer.state_dict(), learning_rate=learning_rate,
            gradients={key: value.grad for key, value in policy.named_parameters()}, rollout=rollout,
            cuda_rng=torch.cuda.get_rng_state(), cpu_rng=torch.get_rng_state())))

    torch.manual_seed(271)
    if args.worker == 'upstream':
        sys.path.insert(0, str(args.rsl_rl))
        from rsl_rl.runners import OnPolicyRunner
        options = dict(policy={**neural, 'class_name': 'ActorCritic'},
            algorithm={**algorithm, 'class_name': 'PPO'}, num_steps_per_env=24,
            save_interval=500, empirical_normalization=True)
        runner = OnPolicyRunner(env, options, device='cuda')
        # Suppress source logging/code-state collection only; learn()/PPO remain original.
        runner.disable_logs = True
        torch.save(dict(model_state_dict=runner.alg.policy.state_dict(),
            obs_norm_state_dict=runner.obs_normalizer.state_dict(),
            privileged_obs_norm_state_dict=runner.privileged_obs_normalizer.state_dict()),
            args.output_dir / 'initial_source_checkpoint.pt')
        update = runner.alg.update
        def capture():
            result = update()
            storage = runner.alg.storage
            rollout = {name: getattr(storage, 'privileged_observations' if name == 'critic_observations' else name)
                       for name in ['observations', 'critic_observations', 'actions', 'values', 'actions_log_prob',
                                    'mu', 'sigma', 'returns', 'advantages']}
            snapshot(runner.alg.policy, runner.obs_normalizer, runner.privileged_obs_normalizer,
                     runner.alg.optimizer, runner.alg.learning_rate, rollout)
            return result
        runner.alg.update = capture
        torch.manual_seed(1234)
        runner.learn(2)
    else:
        from accelerate import Accelerator
        from motius.runner.online_tracking import build_beyondmimic_runner, beyondmimic_training_options
        options = beyondmimic_training_options()
        options['policy'] = neural
        options['algorithm'] = algorithm
        runner = build_beyondmimic_runner(env, args.output_dir / 'native_runner',
            accelerator=Accelerator(mixed_precision='no'), options=options, max_iterations=2,
            checkpoint=args.output_dir / 'initial_source_checkpoint.pt', random_episode_lengths=False)
        update = runner.trainer.train_step
        def capture(rollout):
            result = update(rollout)
            snapshot(runner.bundle.network, runner.bundle.actor_normalizer, runner.bundle.critic_normalizer,
                     runner.optimizers['policy'], runner.trainer.learning_rate, rollout)
            return result
        runner.trainer.train_step = capture
        torch.manual_seed(1234)
        runner.train()
        assert runner.global_step == 2
        assert runner.bundle.actor_normalizer.count.item() == 192
    torch.save(dict(snapshots=snapshots, trace=env.trace, steps=env.step_index), args.output_dir / (args.worker + '.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'rsl-rl', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--worker', choices=['upstream', 'native'])
    args = parser.parse_args()
    for name in ['upstream', 'rsl_rl', 'output_dir']:
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    path = args.output_dir / 'source_parity.json'
    report = dict(complete=False, comparison='Independent original RSL-RL OnPolicyRunner.learn versus actual Motius AccelerateRunner',
        scope='Two 24x4 online rollouts in a deterministic tensor environment; real original G1 network, normalization, sampling, timeout bootstrapping, GAE, 40 PPO minibatch updates and optimizer/checkpoint integration. Not physical simulation or a released pretrained policy.',
        upstream_commit=subprocess.check_output(['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        rsl_rl_commit=subprocess.check_output(['git', '-C', str(args.rsl_rl), 'rev-parse', 'HEAD'], text=True).strip(), tests=[])
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native']:
        env = {**os.environ, 'PYTHONPATH': str(root), 'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
        subprocess.run([sys.executable, __file__, *sys.argv[1:], '--worker', kind],
                       cwd=args.upstream if kind == 'upstream' else root, env=env, check=True)
    import torch
    try:
        compare_full(torch.load(args.output_dir / 'upstream.pt', weights_only=True),
                     torch.load(args.output_dir / 'native.pt', weights_only=True), '', report['tests'])
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Original RSL-RL learn and native online runner are exact')


if __name__ == '__main__':
    main()
