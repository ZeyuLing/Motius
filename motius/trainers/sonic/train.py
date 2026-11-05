"""Native SONIC G1 online PPO through the shared AccelerateRunner."""

from pathlib import Path
import argparse
import os
import shutil
from .online import SONICOnlinePPOTrainer

SonicTrainer = SONICOnlinePPOTrainer
UPSTREAM_COMMIT = "4141c34280abb67c82e115342a8720f4a83d750d"
OUTPUT_ROOT = Path("outputs/training/sonic")
SonicTrainer.upstream_commit = UPSTREAM_COMMIT


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--smpl-motion-file", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path)
    parser.add_argument("--work-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--steps-per-env", type=int)
    parser.add_argument("--learning-epochs", type=int)
    parser.add_argument("--num-mini-batches", type=int)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--terrain-type", choices=["plane", "trimesh"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--kit-args", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser


def export_policy(checkpoint, output_dir, *, configuration=None, arguments=None):
    """Export current native or original neural tensors to a complete Pipeline."""
    import torch
    from motius.models.sonic import SONICPolicyBundle
    from motius.configuration.sonic import original_neural_options

    if arguments:
        raise ValueError("Use --configuration for the native exporter")
    checkpoint = Path(checkpoint).expanduser().resolve()
    if checkpoint.is_dir() and (checkpoint / "sonic_neural.json").is_file():
        bundle = SONICPolicyBundle.from_pretrained(checkpoint)
    elif checkpoint.is_dir() and (checkpoint / "model.pt").is_file():
        configuration = Path(configuration or checkpoint.parent / "config.yaml")
        states = torch.load(
            checkpoint / "model.pt", map_location="cpu", weights_only=True
        )
        original = dict(
            policy_state_dict={
                "std" if name == "std" else "actor_module." + name: value
                for name, value in states["policy"].items()
            },
            value_state_dict={
                name.replace("network.", "critic_module.", 1).replace(
                    "normalizer.", "running_mean_std.", 1
                ): value
                for name, value in states["critic"].items()
            },
        )
        bundle = SONICPolicyBundle(original_neural_options(original, configuration))
        bundle.policy.load_state_dict(states["policy"], strict=True)
        bundle.critic.load_state_dict(states["critic"], strict=True)
    else:
        bundle = SONICPolicyBundle.from_original_checkpoint(
            checkpoint, configuration=configuration
        )
    destination = Path(output_dir).expanduser().resolve()
    bundle.save_pretrained(destination)
    return destination


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    from isaaclab.app import AppLauncher

    device = (
        "cuda:" + os.environ["LOCAL_RANK"]
        if "LOCAL_RANK" in os.environ
        else args.device
    )
    launcher = AppLauncher(
        headless=args.headless, device=device, kit_args=args.kit_args
    )
    environment = None
    try:
        import torch
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from motius.configuration.sonic import original_training_options
        from motius.runner.sonic_tracking import build_sonic_runner
        from motius.simulators.sonic.environment import build_sonic_environment

        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        accelerator = Accelerator(mixed_precision="no")
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        checkpoint = args.checkpoint.expanduser().resolve()
        configuration = (
            args.configuration or checkpoint.parent / "config.yaml"
        ).resolve()
        work_dir = args.work_dir.expanduser().resolve()
        options = original_training_options(configuration)
        if args.steps_per_env is not None:
            options["steps"] = args.steps_per_env
        for argument, key in [
            (args.learning_epochs, "learning_epochs"),
            (args.num_mini_batches, "num_mini_batches"),
        ]:
            if argument is not None:
                options["trainer"][key] = argument
        if args.save_interval is not None:
            options["save_interval"] = args.save_interval
        environment = build_sonic_environment(
            configuration,
            num_envs=args.num_envs,
            assets=args.asset_root,
            motion_file=args.motion_file,
            smpl_motion_file=args.smpl_motion_file,
            work_dir=work_dir,
            terrain_type=args.terrain_type,
            seed=seed,
        )
        initial_observations = environment.reset_all()
        runner, collector = build_sonic_runner(
            environment,
            work_dir,
            checkpoint=checkpoint,
            configuration=configuration,
            schema=environment.schema,
            options=options,
            accelerator=accelerator,
            max_iterations=args.max_iterations,
            auto_resume=args.auto_resume,
        )
        if runner.global_step == 0:
            collector.observations = initial_observations
        if accelerator.is_main_process:
            work_dir.mkdir(parents=True, exist_ok=True)
            destination = work_dir / "config.yaml"
            if configuration != destination:
                shutil.copy2(configuration, destination)
        runner.train()
        if accelerator.is_main_process:
            runner.bundle.save_pretrained(work_dir / "policy")
        accelerator.wait_for_everyone()
    finally:
        if environment is not None:
            environment.close()
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        launcher.app.close()


if __name__ == "__main__":
    main()
