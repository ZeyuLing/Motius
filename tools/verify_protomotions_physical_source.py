"""Independent source comparison for native G1 motion/environment components."""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from verify_prompthmr_video_parity import tensors, compare_full


def prepare(args):
    import torch

    if args.worker == "upstream":
        configuration = torch.load(
            args.artifact / "resolved_configs_inference.pt",
            map_location="cpu",
            weights_only=False,
        )
        from protomotions.components.pose_lib import fk_batch_mjcf_with_velocities
        from protomotions.components.motion_lib import MotionLib, MotionLibConfig
    else:
        from motius.configuration.protomotions.environment import (
            load_environment_configuration,
        )

        configuration = load_environment_configuration(
            args.artifact / "resolved_configs_inference.pt"
        )
        from motius.motion.kinematics.protomotions import fk_batch_mjcf_with_velocities
        from motius.datasets.protomotions_motion import MotionLib, MotionLibConfig

        assert not any(
            name == "protomotions" or name.startswith("protomotions.")
            for name in sys.modules
        )
    robot = configuration["robot"]
    info = robot.kinematic_info.to("cuda")
    count = 128
    time = torch.arange(count, device="cuda") / 50.0
    qpos = torch.zeros(count, 7 + info.num_dofs, device="cuda")
    qpos[:, 0] = time * 0.03
    qpos[:, 1] = 0.01 * torch.sin(time)
    qpos[:, 2] = robot.default_root_height
    qpos[:, 3] = torch.cos(time * 0.015)
    qpos[:, 6] = torch.sin(time * 0.015)
    qpos[:, 7:] = torch.as_tensor(robot.default_dof_pos, device="cuda")
    qpos[:, 7:] += 0.008 * torch.sin(
        time[:, None] + torch.arange(info.num_dofs, device="cuda")[None] * 0.1
    )
    state = fk_batch_mjcf_with_velocities(info, qpos, fps=50)
    # FK supplies body transforms/velocities; reference files additionally
    # carry prescribed joint trajectories and their finite-difference velocity.
    state.dof_pos = qpos[:, 7:].clone()
    state.dof_vel = torch.gradient(state.dof_pos, spacing=(1 / 50.0,), dim=(0,))[0]
    tagged = state.to_dict()
    tagged = {
        key: value.cpu() if isinstance(value, torch.Tensor) else value
        for key, value in tagged.items()
    }
    torch.save(tagged, args.output_dir / (args.worker + "_tagged.motion"))
    data = {key: value for key, value in tagged.items() if key != "state_conversion"}
    torch.save(data, args.output_dir / (args.worker + ".motion"))
    library = MotionLib(
        MotionLibConfig(motion_file=str(args.output_dir / (args.worker + ".motion"))),
        device="cuda",
    )
    ids = torch.zeros(4, dtype=torch.long, device="cuda")
    times = torch.tensor([0.0, 0.013, 0.51, 2.13], device="cuda")
    observed = library.get_motion_state(ids, times).to_dict()
    observed.pop("state_conversion", None)
    if args.worker == "native":
        # Public .motion files can contain the original non-neural enum. Verify
        # that the data reader needs no original package to interpret it.
        portable = load_environment_configuration(
            args.output_dir / "upstream_tagged.motion"
        )
        assert portable["state_conversion"].value == "common"
    torch.save(
        tensors(dict(motion=data, interpolated=observed, qpos=qpos)),
        args.output_dir / ("prepare_" + args.worker + ".pt"),
    )


def tracking(args, launcher):
    import random
    import numpy as np
    import torch
    from pathlib import Path

    motion = args.output_dir / "upstream.motion"
    if args.worker == "upstream":
        from protomotions.utils.component_builder import build_all_components
        from protomotions.simulator.base_simulator.utils import (
            convert_friction_for_simulator,
        )
        from protomotions.envs.base_env.env import BaseEnv
        from protomotions.agents.amp.model import AMPModel
        from protomotions.agents.amp.agent import AMP
        from protomotions.agents.evaluators.base_evaluator import BaseEvaluator
        from protomotions.agents.utils.normalization import (
            materialize_lazy_running_stats_from_state_dict,
        )

        random.seed(533)
        np.random.seed(533)
        torch.manual_seed(533)
        configs = torch.load(
            args.artifact / "resolved_configs_inference.pt",
            map_location="cpu",
            weights_only=False,
        )
        configs["robot"].asset.asset_root = str(args.assets)
        configs["simulator"].num_envs = 4
        configs["simulator"].headless = True
        configs["motion_lib"].motion_file = str(motion)
        configs["env"].save_dir = str(args.output_dir / "upstream-environment")
        terrain, simulator = convert_friction_for_simulator(
            configs["terrain"], configs["simulator"]
        )
        components = build_all_components(
            terrain,
            configs["scene_lib"],
            configs["motion_lib"],
            simulator,
            configs["robot"],
            torch.device("cuda:0"),
            simulation_app=launcher.app,
        )
        environment = BaseEnv(
            config=configs["env"],
            robot_config=configs["robot"],
            device=torch.device("cuda:0"),
            **components,
        )
        model = AMPModel(configs["agent"].model)
        weights = torch.load(
            args.artifact / "last.ckpt", map_location="cpu", weights_only=False
        )["model"]
        materialize_lazy_running_stats_from_state_dict(model, weights)
        model.load_state_dict(weights, strict=True)
        model.cuda().eval()
    else:
        from motius.simulators.isaaclab.protomotions.factory import (
            create_protomotions_environment,
        )
        from motius.pipelines.protomotions import ProtoMotionsPipeline

        environment = create_protomotions_environment(
            args.artifact / "resolved_configs_inference.pt",
            motion,
            args.assets,
            simulation_app=launcher.app,
            num_envs=4,
            device="cuda:0",
            seed=533,
            output_dir=args.output_dir / "native-environment",
        )
        pipeline = ProtoMotionsPipeline.from_original_checkpoint(
            args.artifact / "last.ckpt", device="cuda"
        )
        assert not any(
            name == "protomotions" or name.startswith("protomotions.")
            for name in sys.modules
        )
    results = {}
    step = environment.step
    reset = environment.reset
    counter = 0

    def observed_reset(indices=None):
        observations, extras = reset(indices)
        results[str(counter) + "/input"] = tensors(observations)
        return observations, extras

    def observed_step(actions):
        nonlocal counter
        observations, rewards, dones, terminated, extras = step(actions)
        state = environment.simulator.get_robot_state().to_dict()
        state.pop("state_conversion", None)
        results[str(counter) + "/output"] = tensors(
            dict(
                actions=actions,
                observations=observations,
                rewards=rewards,
                dones=dones,
                terminated=terminated,
                state=state,
                progress=environment.progress_buf,
                motion_ids=environment.motion_manager.motion_ids,
                motion_times=environment.motion_manager.motion_times,
                cpu_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state(),
            )
        )
        counter += 1
        if args.worker == "upstream" and counter >= args.steps:
            raise KeyboardInterrupt
        return observations, rewards, dones, terminated, extras

    environment.reset = observed_reset
    environment.step = observed_step
    try:
        # Reproduce a controlled state after checkpoint/model construction.
        # This avoids coupling physical reset seeds to constructor-only draws.
        random.seed(534)
        np.random.seed(534)
        torch.manual_seed(534)
        if args.worker == "upstream":
            # Execute the actual original evaluator loop, bounded by an
            # observational interruption after the requested physical step.
            agent = object.__new__(AMP)
            agent.model = model
            agent.env = environment
            agent.num_envs = 4
            agent.device = torch.device("cuda:0")
            evaluator = object.__new__(BaseEvaluator)
            evaluator.agent = agent
            with torch.no_grad():
                evaluator.simple_test_policy(collect_metrics=False)
        else:
            done_indices = None
            for index in range(args.steps):
                observations, _ = environment.reset(done_indices)
                actions = pipeline.infer_prepared_observations(
                    observations, source_evaluator=True
                )["actions"]
                _, _, dones, _, _ = environment.step(
                    torch.as_tensor(actions, device="cuda:0")
                )
                done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
        torch.save(results, args.output_dir / ("tracking_" + args.worker + ".pt"))
    finally:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        environment.close()


def worker(args):
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device="cuda:0", kit_args="--allow-root")
    try:
        import torch

        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.worker == "upstream":
            sys.path.insert(0, str(args.upstream))
        if args.phase == "prepare":
            prepare(args)
        else:
            tracking(args, launcher)
    except BaseException:
        import traceback

        error = traceback.format_exc()
        (args.output_dir / (args.phase + "_" + args.worker + "_error.txt")).write_text(
            error
        )
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        launcher.app.close()


def compare_tracking(source, native, args, report):
    """Keep functional outputs exact and expose PhysX sensor repeatability."""
    import torch

    reference_files = [
        args.output_dir / "tracking_upstream_first.pt",
        args.output_dir / "tracking_upstream_repeat.pt",
    ]
    baseline = [torch.load(path, weights_only=True) for path in reference_files]
    observed_repeat = 0.0
    for key in baseline[0]:
        if not key.endswith("/output"):
            continue
        left = baseline[0][key]["state"]["rigid_body_contact_forces"]
        right = baseline[1][key]["state"]["rigid_body_contact_forces"]
        observed_repeat = max(observed_repeat, (left - right).abs().max().item())
    report["contact_sensor_repeatability"] = dict(
        source_repeat_files=[str(path) for path in reference_files],
        source_self_repeat_max_abs=observed_repeat,
        explanation="Original PhysX contact-force sensor reductions vary between identical source-only runs. Only this diagnostic field uses the measured source-self absolute bound; all policy inputs/outputs, physical poses, rewards and RNG remain bitwise comparisons.",
        tests=[],
    )
    # Compare sensor diagnostics separately without changing either saved trace.
    source_functional = {}
    native_functional = {}
    for key in source:
        if not key.endswith("/output"):
            source_functional[key] = source[key]
            native_functional[key] = native[key]
            continue
        source_functional[key] = {**source[key], "state": dict(source[key]["state"])}
        native_functional[key] = {**native[key], "state": dict(native[key]["state"])}
        left = source_functional[key]["state"].pop("rigid_body_contact_forces")
        right = native_functional[key]["state"].pop("rigid_body_contact_forces")
        error = (left - right).abs().max().item()
        report["contact_sensor_repeatability"]["tests"].append(
            dict(
                tensor=key + "/state/rigid_body_contact_forces",
                bitwise_equal=torch.equal(left, right),
                max_abs_error=error,
                absolute_bound=observed_repeat,
            )
        )
        torch.testing.assert_close(left, right, rtol=0, atol=observed_repeat)
    compare_full(source_functional, native_functional, "", report["tests"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "assets", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--phase", choices=["prepare", "tracking"], default="prepare")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "assets", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / (args.phase + "_source_parity.json")
    report = dict(
        complete=False,
        scope="Actual original typed G1 configuration, FK/velocities, reference-motion serialization and interpolation versus native configuration/kinematics/data modules under Isaac Sim5.1 / Lab2.3.2. Synthetic nonzero 128-frame reference; no physical tracking or full training certification.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    if args.phase == "tracking":
        report["scope"] = (
            "Actual original BaseEvaluator.simple_test_policy versus native environment/Bundle/Pipeline in four real G1 physical environments. Source-pinned released checkpoint and robot assets, synthetic 128-frame FK reference, controlled seed after model construction and shared FP32/eager MDP runtime. Not full dataset quality or training protocol."
        )
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        expected = args.output_dir / (args.phase + "_" + kind + ".pt")
        expected.unlink(missing_ok=True)
        (args.output_dir / (args.phase + "_" + kind + "_error.txt")).unlink(
            missing_ok=True
        )
        subprocess.run(
            [sys.executable, "-u", __file__, *sys.argv[1:], "--worker", kind],
            cwd=args.upstream if kind == "upstream" else root,
            env={
                **os.environ,
                "PYTHONPATH": str(root),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            check=True,
        )
        if not expected.is_file():
            raise RuntimeError(kind + " worker did not produce the requested result")
        if args.phase == "tracking" and kind == "upstream":
            import shutil

            shutil.copy2(expected, args.output_dir / "tracking_upstream_first.pt")
            expected.unlink()
            subprocess.run(
                [sys.executable, "-u", __file__, *sys.argv[1:], "--worker", kind],
                cwd=args.upstream,
                env={
                    **os.environ,
                    "PYTHONPATH": str(root),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
                check=True,
            )
            if not expected.is_file():
                raise RuntimeError("Original self-repeat did not produce fresh results")
            shutil.copy2(expected, args.output_dir / "tracking_upstream_repeat.pt")
    import torch

    torch.set_num_threads(4)
    try:
        source = torch.load(
            args.output_dir / (args.phase + "_upstream.pt"), weights_only=True
        )
        native = torch.load(
            args.output_dir / (args.phase + "_native.pt"), weights_only=True
        )
        if args.phase == "tracking":
            compare_tracking(source, native, args, report)
        else:
            compare_full(source, native, "", report["tests"])
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Original and native G1 "
        + args.phase
        + " functional data exact; see report for sensor repeatability"
    )


if __name__ == "__main__":
    main()
