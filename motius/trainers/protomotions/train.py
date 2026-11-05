"""Train the native G1 PPO/AMP model through the shared AccelerateRunner."""

from pathlib import Path
import argparse
import os
from .trainer import ProtoMotionsPPOTrainer

# Preserve the public name while exposing a real framework Trainer.
ProtoMotionsTrainer = ProtoMotionsPPOTrainer


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        type=Path,
        help="Original resolved training YAML; defaults beside the checkpoint",
    )
    parser.add_argument(
        "--environment-configuration",
        type=Path,
        help="Original resolved training PT; defaults beside the checkpoint",
    )
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            os.environ.get(
                "MOTIUS_PROTOMOTIONS_OUTPUT_ROOT", "outputs/training/protomotions"
            )
        ),
    )
    parser.add_argument("--run-name", default="g1")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--discriminator-batch-size", type=int)
    parser.add_argument("--cap-discriminator-batch-size", type=int)
    parser.add_argument("--kit-args", default="")
    parser.add_argument("--auto-resume", action="store_true")
    parser.add_argument("--resume-original", action="store_true")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--logger", choices=["tensorboard", "none"], default="tensorboard"
    )
    return parser


def main(arguments=None):
    args = build_parser().parse_args(arguments)
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from isaaclab.app import AppLauncher

    if "LOCAL_RANK" in os.environ:
        args.device = "cuda:" + os.environ["LOCAL_RANK"]
    launcher = AppLauncher(
        headless=args.headless, device=args.device, kit_args=args.kit_args
    )
    environment = None
    try:
        from motius.runner.protomotions_tracking import build_protomotions_runner
        from motius.simulators.isaaclab.protomotions.factory import (
            create_protomotions_environment,
        )
        from motius.configuration.protomotions import (
            original_deployment_options,
            original_training_options,
        )
        from motius.pipelines.protomotions import ProtoMotionsPipeline
        import torch

        torch.set_float32_matmul_precision("high")
        work_dir = args.output_root.expanduser().resolve() / args.run_name
        accelerator = Accelerator(
            mixed_precision="no",
            log_with="tensorboard" if args.logger == "tensorboard" else None,
            project_dir=str(work_dir),
        )
        seed = args.seed + accelerator.process_index
        set_seed(seed)
        checkpoint = args.checkpoint.expanduser().resolve()
        configuration = (
            args.configuration or checkpoint.parent / "resolved_configs.yaml"
        )
        environment_configuration = args.environment_configuration or Path(
            configuration
        ).with_suffix(".pt")
        environment = create_protomotions_environment(
            environment_configuration,
            args.motion_file,
            args.asset_root,
            simulation_app=launcher.app,
            device=accelerator.device,
            num_envs=args.num_envs,
            seed=seed,
            headless=args.headless,
            output_dir=work_dir,
        )
        import yaml

        source_options = yaml.safe_load(Path(configuration).read_text())
        for override in args.overrides:
            key, separator, value = override.partition("=")
            if not separator or not key.startswith("agent."):
                raise ValueError(
                    "Training overrides must use existing agent.path=value keys"
                )
            node = source_options
            parts = key.split(".")
            for part in parts[:-1]:
                node = node[part]
            if parts[-1] not in node:
                raise KeyError(key)
            node[parts[-1]] = yaml.safe_load(value)
        options = original_training_options(source_options)
        if args.batch_size is not None:
            options["batch_size"] = args.batch_size
        if args.discriminator_batch_size is not None:
            options["trainer"][
                "discriminator_batch_size"
            ] = args.discriminator_batch_size
        if args.cap_discriminator_batch_size is not None:
            options["trainer"]["discriminator_batch_size"] = min(
                options["trainer"]["discriminator_batch_size"],
                args.cap_discriminator_batch_size,
            )
        runner = build_protomotions_runner(
            environment,
            work_dir,
            checkpoint=checkpoint,
            configuration=configuration,
            options=options,
            expert_provider=environment.expert_observations,
            accelerator=accelerator,
            max_iterations=args.max_iterations,
            auto_resume=args.auto_resume,
            resume_original=args.resume_original,
        )
        initial_epoch = runner.global_step
        initial_steps = (
            initial_epoch
            * environment.num_envs
            * options["num_steps"]
            * accelerator.num_processes
        )
        if args.resume_original:
            initial_steps = int(
                torch.load(checkpoint, map_location="cpu", weights_only=False).get(
                    "step_count", 0
                )
            )
        if accelerator.is_main_process:
            import shutil

            work_dir.mkdir(parents=True, exist_ok=True)
            for name in [
                "resolved_configs.pt",
                "resolved_configs_inference.pt",
                "resolved_configs_inference.yaml",
            ]:
                source = Path(configuration).parent / name
                destination = work_dir / name
                if source.is_file() and source.resolve() != destination.resolve():
                    shutil.copy2(source, destination)
            (work_dir / "resolved_configs.yaml").write_text(
                yaml.safe_dump(source_options, sort_keys=False)
            )
        accelerator.init_trackers("protomotions")
        runner.train()
        if accelerator.is_main_process:
            # Save the weights actually trained; source checkpoint metadata never
            # determines the exported neural state.
            for name in runner.bundle._save_ckpt_modules:
                setattr(
                    runner.bundle,
                    name,
                    accelerator.unwrap_model(getattr(runner.bundle, name)),
                )
            from motius.utils.protomotions_checkpoint import save_source_checkpoint

            actual_steps = (
                initial_steps
                + (runner.global_step - initial_epoch)
                * environment.num_envs
                * options["num_steps"]
                * accelerator.num_processes
            )
            save_source_checkpoint(
                runner, work_dir / "last.ckpt", step_count=actual_steps
            )
            inference_config = (
                Path(configuration).parent / "resolved_configs_inference.yaml"
            )
            deployment = original_deployment_options(
                inference_config if inference_config.is_file() else configuration
            )
            ProtoMotionsPipeline(runner.bundle, deployment=deployment).save_pretrained(
                work_dir / "exported"
            )
    finally:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        if environment is not None:
            environment.close()
        launcher.app.close()


if __name__ == "__main__":
    main()
