"""Compare released G1 PPO/AMP neural roots with pristine ProtoMotions."""

from __future__ import annotations
import argparse, json, os, subprocess, sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors, compare_full


def worker(args):
    import torch

    rank = 0
    fabric = None
    if args.distributed:
        rank = int(os.environ['RANK'])
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        if args.worker == 'upstream':
            from lightning.fabric import Fabric

            fabric = Fabric(
                accelerator='cuda', devices=2, strategy='ddp', precision='32-true'
            )
            fabric.launch()
        else:
            torch.distributed.init_process_group('nccl')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(813)
    if args.worker == 'upstream':
        sys.path.insert(0, str(args.upstream))
        from tensordict import TensorDict
        from protomotions.agents.amp.model import AMPModel
        from protomotions.agents.utils.normalization import (
            materialize_lazy_running_stats_from_state_dict,
        )

        config = torch.load(
            args.artifact / 'resolved_configs.pt',
            map_location='cpu',
            weights_only=False,
        )
        model = AMPModel(config['agent'].model)
        weights = torch.load(
            args.artifact / 'last.ckpt', map_location='cpu', weights_only=False
        )['model']
        materialize_lazy_running_stats_from_state_dict(model, weights)
        model.load_state_dict(weights, strict=True)
        model.cuda()
        if fabric is not None:
            from protomotions.agents.utils.normalization import RunningMeanStd

            for module in model.modules():
                if isinstance(module, RunningMeanStd):
                    module.fabric = fabric
    else:
        from motius.models.protomotions import ProtoMotionsPolicyBundle

        model = ProtoMotionsPolicyBundle.from_original_checkpoint(
            args.artifact / 'last.ckpt', device='cuda'
        )
    results = {}
    dimensions = dict(
        noisy_reduced_coords_obs=64,
        noisy_mimic_reduced_coords_target_poses=256,
        clean_reduced_coords_obs=64,
        clean_mimic_reduced_coords_target_poses=256,
        max_coords_obs=493,
        mimic_max_coords_target_poses=3168,
        historical_max_coords_obs=3451,
        historical_previous_processed_actions=29,
    )
    for index, (training, batch_size) in enumerate(
        [(False, 1), (False, 3), (True, 4), (True, 2)]
    ):
        batch_size += rank
        model.train(training)
        model.zero_grad(set_to_none=True)
        torch.manual_seed(814 + index + rank * 10)
        inputs = {
            name: torch.randn(batch_size, size, device='cuda') * 0.37
            for name, size in dimensions.items()
        }
        torch.manual_seed(910 + index)
        if args.worker == 'upstream':
            source = model(TensorDict(inputs, batch_size=[batch_size]))
            output = {
                name: source[name]
                for name in [
                    'mean_action',
                    'action',
                    'neglogp',
                    'value',
                    'disc_logits',
                    'disc_value',
                ]
            }
        else:
            result = model(inputs)
            distribution = torch.distributions.Normal(result['mean'], result['std'])
            action = distribution.sample()
            output = dict(
                mean_action=result['mean'],
                action=action,
                neglogp=-distribution.log_prob(action).sum(-1),
                **{
                    name: result[name]
                    for name in ['value', 'disc_logits', 'disc_value']
                },
            )
        # A shared probe checks neural backpropagation, not the PPO/AMP objective.
        loss = sum(
            value.square().mean() for name, value in output.items() if name != 'action'
        )
        loss.backward()
        state = model.state_dict()
        gradients = {name: value.grad for name, value in model.named_parameters()}
        if args.worker == 'upstream':

            def canonical(values):
                prefixes = {
                    '_actor.mu.': 'actor.mean.',
                    '_actor.': 'actor.',
                    '_critic.': 'critic.',
                    '_discriminator.models.0.': 'discriminator.',
                    '_disc_critic.models.0.': 'discriminator_critic.',
                }
                renamed = {}
                for key, value in values.items():
                    for source, target in prefixes.items():
                        if key.startswith(source):
                            renamed[
                                (target + key[len(source) :]).replace(
                                    'running_obs_norm.', ''
                                )
                            ] = value
                            break
                    else:
                        raise ValueError('Unexpected original neural key: ' + key)
                return renamed

            state = canonical(state)
            gradients = canonical(gradients)
        results[str(index)] = tensors(
            dict(
                output=output,
                probe_loss=loss,
                state=state,
                gradients=gradients,
                cuda_rng=torch.cuda.get_rng_state(),
            )
        )
    if args.worker == 'native' and rank == 0:
        model.save_pretrained(args.output_dir / 'exported')
        restored = type(model).from_pretrained(
            args.output_dir / 'exported', device='cuda'
        )
        for key, value in model.state_dict().items():
            assert torch.equal(value, restored.state_dict()[key]), key
    suffix = ('_rank' + str(rank)) if args.distributed else ''
    torch.save(results, args.output_dir / (args.worker + suffix + '.pt'))
    if args.distributed:
        torch.distributed.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'artifact', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--distributed', action='store_true')
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
        scope='Released G1 actor, both value networks and AMP discriminator; prepared observation features, mean/action/log-probability/value/logits, running statistics and shared neural gradient probes. Not the source PPO/AMP training objective or physical rollout.',
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        artifact=json.loads((args.artifact / 'provenance.json').read_text()),
        tests=[],
    )
    if args.distributed:
        report[
            'scope'
        ] += ' Two ranks with unequal local batch sizes; actual original Fabric normalization collectives versus native synchronized moments. Neural gradient probes are local; this does not certify DDP optimizer updates.'
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native']:
        env = {
            **os.environ,
            'PYTHONPATH': str(root),
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        }
        prefix = (
            [
                sys.executable,
                '-m',
                'torch.distributed.run',
                '--standalone',
                '--nproc-per-node',
                '2',
            ]
            if args.distributed
            else [sys.executable]
        )
        subprocess.run(
            [*prefix, __file__, *sys.argv[1:], '--worker', kind],
            cwd=args.upstream if kind == 'upstream' else root,
            env=env,
            check=True,
        )
    import torch

    torch.set_num_threads(4)
    try:
        for rank in range(2 if args.distributed else 1):
            suffix = ('_rank' + str(rank)) if args.distributed else ''
            compare_full(
                torch.load(
                    args.output_dir / ('upstream' + suffix + '.pt'), weights_only=True
                ),
                torch.load(
                    args.output_dir / ('native' + suffix + '.pt'), weights_only=True
                ),
                'rank' + str(rank),
                report['tests'],
            )
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print('Released ProtoMotions G1 neural comparisons exact')


if __name__ == '__main__':
    main()
