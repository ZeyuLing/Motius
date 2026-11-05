"""Launch native BeyondMimic neural roots, simulator and AccelerateRunner."""

from pathlib import Path
import argparse, os
from .trainer import BeyondMimicTrainer

TASK_VARIANTS = {
    'Tracking-Flat-G1-v0': 'flat',
    'Tracking-Flat-G1-Wo-State-Estimation-v0': 'without_state_estimation',
    'Tracking-Flat-G1-Low-Freq-v0': 'low_frequency',
}
DEFAULT_TASK = 'Tracking-Flat-G1-v0'
DEFAULT_OUTPUT_ROOT = Path('outputs/training/beyondmimic')
DEFAULT_ASSET_ROOT = Path('checkpoints/robots')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=tuple(TASK_VARIANTS), default=DEFAULT_TASK)
    parser.add_argument(
        '--motion_file', '--motion-file', dest='motion_file', type=Path, required=True
    )
    parser.add_argument(
        '--asset-root',
        type=Path,
        default=Path(
            os.environ.get('MOTIUS_BEYONDMIMIC_ASSET_ROOT', DEFAULT_ASSET_ROOT)
        ),
    )
    parser.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        '--num_envs', '--num-envs', dest='num_envs', type=int, default=4096
    )
    parser.add_argument(
        '--max_iterations', '--max-iterations', dest='max_iterations', type=int
    )
    parser.add_argument(
        '--run_name', '--run-name', dest='run_name', default='lafan1_g1'
    )
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--auto-resume', action='store_true')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument(
        '--logger', choices=['tensorboard', 'none'], default='tensorboard'
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    # Kit prepends its bundled Python packages at startup. Import the training
    # runtime first so its installed boto3/botocore dependencies stay paired.
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from isaaclab.app import AppLauncher

    local_rank = os.environ.get('LOCAL_RANK')
    if local_rank is not None:
        args.device = 'cuda:' + local_rank
    launcher = AppLauncher(headless=args.headless, device=args.device)
    environment = None
    try:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        from motius.runner.online_tracking import build_beyondmimic_runner
        from motius.simulators.isaaclab.beyondmimic import (
            create_beyondmimic_environment,
        )
        from motius.pipelines.beyondmimic import BeyondMimicPipeline

        work_dir = args.output_root.expanduser().resolve() / args.run_name
        accelerator = Accelerator(
            mixed_precision='no',
            log_with='tensorboard' if args.logger == 'tensorboard' else None,
            project_dir=str(work_dir),
        )
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        variant = TASK_VARIANTS[args.task]
        environment = create_beyondmimic_environment(
            args.motion_file,
            args.asset_root,
            num_envs=args.num_envs,
            device=accelerator.device,
            seed=seed,
            variant=variant,
            output_dir=work_dir,
        )
        runner = build_beyondmimic_runner(
            environment,
            work_dir,
            accelerator=accelerator,
            variant=variant,
            max_iterations=args.max_iterations,
            checkpoint=args.checkpoint,
            auto_resume=args.auto_resume,
        )
        accelerator.init_trackers('beyondmimic')
        runner.train()
        if accelerator.is_main_process:
            command = environment.environment.command_manager.get_term('motion')
            reference = {
                name: getattr(command.motion, name).detach().cpu().numpy()
                for name in [
                    'joint_pos',
                    'joint_vel',
                    'body_pos_w',
                    'body_quat_w',
                    'body_lin_vel_w',
                    'body_ang_vel_w',
                ]
            }
            from motius.simulators.isaaclab.beyondmimic.factory import tracking_metadata

            pipeline = BeyondMimicPipeline(
                runner.bundle,
                reference_motion=reference,
                control_hz=1 / environment.environment.step_dt,
                metadata=tracking_metadata(environment),
            )
            pipeline.save_pretrained(work_dir / 'exported')
    finally:
        if environment is not None:
            environment.close()
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        launcher.app.close()


if __name__ == '__main__':
    main()
