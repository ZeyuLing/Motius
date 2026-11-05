"""Compare actual original AMP.fit with native online AccelerateRunner."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from verify_protomotions_training_source import canonical
from verify_prompthmr_video_parity import tensors, compare_full


def worker(args):
    import numpy as np
    import torch
    import random

    random.seed(921)

    rank = int(os.environ.get("RANK", 0)) if args.distributed else 0
    world_size = 2 if args.distributed else 1
    if args.distributed:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(921)
    np.random.seed(921)
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

    class TensorTrackingFixture:
        num_envs = 4
        device = torch.device("cuda")
        motion_lib = None
        motion_manager = None

        def __init__(self):
            self.position = (
                torch.arange(4, device="cuda")[:, None].float() * 0.01 + rank * 0.03
            )
            self.age = torch.zeros(4, device="cuda", dtype=torch.long)
            self.steps = 0
            self.trace = []

        def get_obs(self):
            return {
                name: torch.sin(
                    self.position + torch.arange(width, device="cuda") * 0.013
                )
                * 0.13
                for name, width in dimensions.items()
            }

        def reset(self, indices=None):
            if indices is None:
                indices = torch.arange(4, device="cuda")
            self.position[indices] = indices[:, None].float() * 0.01 + rank * 0.03
            self.age[indices] = 0
            return self.get_obs(), {}

        def step(self, actions):
            self.steps += 1
            self.age += 1
            self.position = torch.tanh(
                self.position * 0.97 + actions.mean(-1, keepdim=True) * 0.02
            )
            obs = self.get_obs()
            rewards = 1.0 - actions.square().mean(-1)
            dones = self.age >= torch.tensor([7, 11, 13, 17], device="cuda")
            terminated = dones & (torch.arange(4, device="cuda") % 2 == 0)
            self.trace.append(
                tensors(
                    dict(
                        actions=actions,
                        observations=obs,
                        rewards=rewards,
                        dones=dones,
                        terminated=terminated,
                    )
                )
            )
            return obs, rewards, dones, terminated, {}

        def on_epoch_end(self, epoch):
            pass

    environment = TensorTrackingFixture()
    launcher = None
    if args.physical:
        # Import the training runtime before Kit prepends its bundled packages.
        import accelerate
        import lightning.fabric
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, device="cuda:0", kit_args="--allow-root")
        if args.worker == "upstream":
            sys.path.insert(0, str(args.upstream))
            from protomotions.utils.component_builder import build_all_components
            from protomotions.simulator.base_simulator.utils import (
                convert_friction_for_simulator,
            )
            from protomotions.envs.base_env.env import BaseEnv

            config = torch.load(
                args.artifact / "resolved_configs.pt",
                map_location="cpu",
                weights_only=False,
            )
            config["robot"].asset.asset_root = str(args.assets)
            config["simulator"].num_envs = 4
            config["simulator"].headless = True
            config["motion_lib"].motion_file = str(args.motion_file)
            config["env"].save_dir = str(args.output_dir / "upstream-environment")
            terrain, simulator = convert_friction_for_simulator(
                config["terrain"], config["simulator"]
            )
            components = build_all_components(
                terrain,
                config["scene_lib"],
                config["motion_lib"],
                simulator,
                config["robot"],
                torch.device("cuda:0"),
                simulation_app=launcher.app,
            )
            environment = BaseEnv(
                config=config["env"],
                robot_config=config["robot"],
                device=torch.device("cuda:0"),
                **components,
            )
        else:
            from motius.simulators.isaaclab.protomotions.factory import (
                create_protomotions_environment,
            )

            environment = create_protomotions_environment(
                args.artifact / "resolved_configs.pt",
                args.motion_file,
                args.assets,
                simulation_app=launcher.app,
                num_envs=4,
                device="cuda:0",
                seed=921,
                output_dir=args.output_dir / "native-environment",
            )
        environment.trace = []
        environment.steps = 0
        physical_step = environment.step

        def traced_step(actions):
            observations, rewards, dones, terminated, extras = physical_step(actions)
            state = environment.simulator.get_robot_state()
            environment.steps += 1
            environment.trace.append(
                tensors(
                    dict(
                        actions=actions,
                        observations=observations,
                        rewards=rewards,
                        dones=dones,
                        terminated=terminated,
                        joint_pos=state.dof_pos,
                        joint_vel=state.dof_vel,
                        body_pos=state.rigid_body_pos,
                        body_rot=state.rigid_body_rot,
                    )
                )
            )
            return observations, rewards, dones, terminated, extras

        environment.step = traced_step

    def expert_provider(count):
        return {
            "historical_max_coords_obs": torch.sin(
                torch.arange(count, device="cuda")[:, None] * 0.03
                + torch.arange(3451, device="cuda")[None] * 0.013
            )
            * 0.13
        }

    snapshots = []
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from lightning.fabric import Fabric
        from protomotions.agents.amp.agent import AMP

        config = torch.load(
            args.artifact / "resolved_configs.pt",
            map_location="cpu",
            weights_only=False,
        )["agent"]
        config.batch_size = 32
        config.amp_parameters.discriminator_batch_size = 4
        config.amp_parameters.discriminator_replay_size = 256
        config.training_max_steps = (
            (4 if args.resume_original else 2) * 128 * world_size
        )
        if args.disable_clip_skip:
            config.actor_clip_frac_threshold = None
        # Evaluation/logging/output persistence are outside this training
        # comparison. Every neural/rollout/PPO/AMP operation remains original.
        config.evaluator._target_ = (
            "protomotions.agents.evaluators.base_evaluator.BaseEvaluator"
        )
        config.evaluator.eval_metrics_every = None
        fabric = Fabric(accelerator="cuda", devices=world_size, precision="32-true")
        if args.distributed:
            fabric.launch()
        agent = AMP(
            fabric, environment, config, root_dir=args.output_dir / "upstream-runner"
        )
        agent.setup()
        checkpoint = torch.load(
            args.artifact / "last.ckpt", map_location="cpu", weights_only=False
        )
        if args.resume_original:
            checkpoint["epoch"] = 2
            checkpoint["step_count"] = 256 * world_size
            checkpoint.pop("evaluator", None)
            agent.load_parameters(checkpoint, load_training_state=True)
        else:
            agent._load_model_state_dict(checkpoint["model"])
        if not args.physical:
            agent.get_expert_disc_obs = expert_provider
        agent.post_epoch_logging = lambda logs: None
        agent.save = lambda **kwargs: None
        optimize = agent.optimize_model

        def capture():
            result = optimize()
            component = agent.amp_component
            optimizers = dict(
                actor=agent.actor_optimizer,
                critic=agent.critic_optimizer,
                discriminator=component.discriminator_optimizer,
                discriminator_critic=component.disc_critic_optimizer,
            )
            snapshots.append(
                tensors(
                    dict(
                        state=canonical(agent.model.state_dict()),
                        gradients=canonical(
                            {
                                name: value.grad
                                for name, value in agent.model.named_parameters()
                            }
                        ),
                        optimizers={
                            name: value.state_dict()
                            for name, value in optimizers.items()
                        },
                        mean=agent.adv_mean_ema,
                        std=agent.adv_std_ema,
                        bad_transitions=component.num_cumulative_bad_transitions,
                        cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),
                    )
                )
            )
            return result

        agent.optimize_model = capture
        torch.manual_seed(923)
        np.random.seed(923)
        agent.fit()
    else:
        from accelerate import Accelerator
        from motius.configuration.protomotions import original_training_options
        from motius.runner.protomotions_tracking import build_protomotions_runner

        options = original_training_options(args.artifact / "resolved_configs.yaml")
        options["batch_size"] = 32
        options["trainer"]["discriminator_batch_size"] = 4
        options["replay"]["capacity"] = 256
        if args.disable_clip_skip:
            options["trainer"]["actor_clip_frac_threshold"] = None
        native_checkpoint = args.artifact / "last.ckpt"
        if args.resume_original:
            state = torch.load(
                native_checkpoint, map_location="cpu", weights_only=False
            )
            state["epoch"] = 2
            state["step_count"] = 256 * world_size
            state.pop("evaluator", None)
            native_checkpoint = args.output_dir / ("resume_rank" + str(rank) + ".ckpt")
            torch.save(state, native_checkpoint)
        runner = build_protomotions_runner(
            environment,
            args.output_dir / "native-runner",
            checkpoint=native_checkpoint,
            configuration=args.artifact / "resolved_configs.yaml",
            resume_original=args.resume_original,
            options=options,
            expert_provider=(
                environment.expert_observations if args.physical else expert_provider
            ),
            accelerator=Accelerator(mixed_precision="no"),
            max_iterations=4 if args.resume_original else 2,
        )
        update = runner.trainer.train_step

        def neural_state(gradients=False):
            result = {}
            for name in runner.bundle._save_ckpt_modules:
                module = getattr(runner.bundle, name)
                module = getattr(module, "module", module)
                values = (
                    {key: value.grad for key, value in module.named_parameters()}
                    if gradients
                    else module.state_dict()
                )
                result.update(
                    {name + "." + key: value for key, value in values.items()}
                )
            return result

        # The trainer recursively consumes minibatches. Capture only its full
        # rollout boundary, not those nested objective calls.
        def capture(batch):
            result = update(batch)
            if "rollout_batches" in batch:
                data = runner.train_dataloader.dataset.collector
                snapshots.append(
                    tensors(
                        dict(
                            state=neural_state(),
                            gradients=neural_state(gradients=True),
                            optimizers={
                                name: value.state_dict()
                                for name, value in runner.optimizers.items()
                            },
                            mean=data.collector.normalization.mean,
                            std=data.collector.normalization.std,
                            bad_transitions=data.collector.bad_transitions,
                            cpu_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state(),
                        )
                    )
                )
            return result

        runner.trainer.train_step = capture
        torch.manual_seed(923)
        np.random.seed(923)
        runner.train()
        assert runner.global_step == (4 if args.resume_original else 2)
    suffix = "_rank" + str(rank) if args.distributed else ""
    torch.save(
        dict(snapshots=snapshots, trace=environment.trace, steps=environment.steps),
        args.output_dir / (args.worker + suffix + ".pt"),
    )
    if args.distributed and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    if launcher is not None:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        environment.close()
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--disable-clip-skip", action="store_true")
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--resume-original", action="store_true")
    parser.add_argument("--physical", action="store_true")
    parser.add_argument("--assets", type=Path)
    parser.add_argument("--motion-file", type=Path)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    if args.physical and (
        args.assets is None or args.motion_file is None or args.distributed
    ):
        parser.error("--physical requires --assets and --motion-file and a single GPU")
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        try:
            worker(args)
        except BaseException:
            import traceback

            (args.output_dir / (args.worker + "_error.txt")).write_text(
                traceback.format_exc()
            )
            raise
        return
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "source_parity.json"
    report = dict(
        complete=False,
        scope="Actual original AMP.fit versus native AccelerateRunner with released G1 neural weights and fresh optimizers. Two 32x4 rollouts in a deterministic tensor environment with prepared expert observations; original reset/sampling/GAE/style/replay/EMA/shuffle/PPO/AMP operations, all neural/optimizer states and RNG. Evaluation/logging/source file saving disabled. Not physical simulation or full motion-data protocol.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    if args.physical:
        report["scope"] = (
            "Actual original AMP.fit versus native AccelerateRunner in four real Isaac Sim 5.1 / Lab 2.3.2 G1 environments, with released weights and source-pinned robot assets. Two 32x4 rollouts, real FK motion-library expert sampling, all PPO/AMP optimizer and neural states, physical actions/observations/rewards/poses and RNG. Synthetic 128-frame reference, fresh optimizers and shared FP32/eager runtime; not full dataset quality or long training."
        )
    if args.resume_original:
        report[
            "scope"
        ] += " Original BaseAgent.load_parameters training-state restore versus native original-checkpoint restore; actual released Adam/EMA states, controlled epoch=2 and step_count=256*world_size, evaluation-only state omitted. Continues exactly two complete iterations from that counter."
    if args.disable_clip_skip:
        report[
            "scope"
        ] += " Additional coverage case: source and native actor clip-fraction skipping explicitly disabled to exercise every minibatch update."
        path.write_text(json.dumps(report, indent=2) + "\n")
    if args.distributed:
        report[
            "scope"
        ] += " Two actual GPU ranks with different local tensor-environment states; original Fabric versus native Accelerate DDP."
        path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        for rank in range(2 if args.distributed else 1):
            suffix = "_rank" + str(rank) if args.distributed else ""
            (args.output_dir / (kind + suffix + ".pt")).unlink(missing_ok=True)
        (args.output_dir / (kind + "_error.txt")).unlink(missing_ok=True)
        prefix = (
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node",
                "2",
            ]
            if args.distributed
            else [sys.executable, "-u"]
        )
        subprocess.run(
            [*prefix, __file__, *sys.argv[1:], "--worker", kind],
            cwd=args.upstream if kind == "upstream" else root,
            env={
                **os.environ,
                "PYTHONPATH": str(root),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            check=True,
        )
        for rank in range(2 if args.distributed else 1):
            suffix = "_rank" + str(rank) if args.distributed else ""
            if not (args.output_dir / (kind + suffix + ".pt")).is_file():
                raise RuntimeError(
                    kind + " worker did not produce fresh training results"
                )
    import torch

    torch.set_num_threads(4)
    try:
        for rank in range(2 if args.distributed else 1):
            suffix = "_rank" + str(rank) if args.distributed else ""
            compare_full(
                torch.load(
                    args.output_dir / ("upstream" + suffix + ".pt"), weights_only=True
                ),
                torch.load(
                    args.output_dir / ("native" + suffix + ".pt"), weights_only=True
                ),
                "rank" + str(rank),
                report["tests"],
            )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("Original AMP.fit and native online runner exact")


if __name__ == "__main__":
    main()
