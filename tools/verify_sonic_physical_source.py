"""Independent original SONIC G1 environment and native deployment on real physics."""

from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device="cuda:0", kit_args="--allow-root")
    try:
        run_physics(args)
    except BaseException:
        import traceback

        (args.output_dir / (args.worker + "_error.txt")).write_text(
            traceback.format_exc()
        )
        raise
    finally:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        launcher.app.close()


def run_physics(args):
    import torch
    import numpy as np
    import random
    from omegaconf import OmegaConf
    from isaaclab.envs import ManagerBasedRLEnv
    from verify_prompthmr_video_parity import tensors
    from verify_motionstreamer_source import hashes

    assert os.uname().nodename == "zeyuling-g2-depth11-eval60k-4g-1020-worker-0"
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(712)
    np.random.seed(712)
    torch.manual_seed(712)
    configuration = args.artifact / "sonic_release/config.yaml"
    original = args.worker != "native"
    if original:
        sys.path.insert(0, str(args.upstream))
        from gear_sonic.trl.utils.common import custom_instantiate
        from gear_sonic.envs.wrapper.manager_env_wrapper import ManagerEnvWrapper
        from motius.configuration.sonic.environment import resolved_g1_urdf

        text = configuration.read_text().replace("groot.rl.envs.", "gear_sonic.envs.")
        source = OmegaConf.create(text)
        source.num_envs, source.experiment_dir = 4, str(
            args.output_dir / "source-runtime"
        )
        manager = OmegaConf.to_container(source.manager_env, resolve=True)
        manager["config"].update(headless=True, terrain_type="plane")
        library = manager["commands"]["motion"]["motion_lib_cfg"]
        library["asset"]["assetRoot"] = str(args.assets / "robot_description/mjcf")
        library.update(
            motion_file=str(args.fixture / "robot.pkl"),
            smpl_motion_file=str(args.fixture / "smpl.pkl"),
        )
        config = custom_instantiate(OmegaConf.create(manager))
        config.scene.robot.spawn.asset_path = str(resolved_g1_urdf(args.assets))
        config.seed = 712
        environment = ManagerEnvWrapper(ManagerBasedRLEnv(config), manager["config"])
        observations = environment.reset(flatten_dict_obs=False)
        named = observations["tokenizer"]
        order = environment.env.observation_manager._group_obs_term_names["tokenizer"]
        schema = {name: tuple(named[name].shape[1:]) for name in order}
        observations["tokenizer"] = torch.cat(
            [named[name].reshape(4, -1) for name in order], -1
        )
        from gear_sonic.trl.modules.actor_critic_modules import Actor

        text = configuration.read_text()
        for before, after in [
            ("groot.rl.trl.", "gear_sonic.trl."),
            ("groot.rl.agents.modules.modules.", "gear_sonic.trl.modules.base_module."),
        ]:
            text = text.replace(before, after)
        algo = OmegaConf.create(text).algo.config
        backbone = algo.actor.backbone.copy()
        backbone.aux_loss_func, backbone.aux_loss_coef = {}, {}
        metadata = NS(
            obs=NS(
                group_obs_dims={"tokenizer": schema},
                group_obs_names={"tokenizer": list(schema)},
            ),
            robot=NS(
                actions_dim=29, algo_obs_dim_dict={"actor_obs": 930, "critic_obs": 1645}
            ),
        )
        policy = Actor(
            metadata, algo, backbone, input_obs_dict=True, has_aux_loss=False
        )
        state = torch.load(
            args.fixture / "neural.pt", map_location="cpu", weights_only=True
        )
        policy.load_state_dict(state["policy_state_dict"], strict=True)
        policy.cuda().eval()
        neural = {
            name.removeprefix("actor_module."): value
            for name, value in hashes(policy).items()
        }
    else:
        from motius.simulators.sonic.environment import build_sonic_environment
        from motius.pipelines.sonic import SONICPipeline

        environment = build_sonic_environment(
            configuration,
            num_envs=4,
            terrain_type="plane",
            assets=args.assets,
            motion_file=args.fixture / "robot.pkl",
            smpl_motion_file=args.fixture / "smpl.pkl",
            work_dir=args.output_dir / "source-runtime",
            seed=712,
        )
        observations = environment.reset_all()
        schema = environment.schema
        pipeline = SONICPipeline.from_original_checkpoint(
            args.artifact / "sonic_release/last.pt", device="cuda"
        )
        pipeline.bundle.eval()
        neural = hashes(pipeline.bundle.policy)
        assert not any(
            name == "gear_sonic" or name.startswith("gear_sonic.")
            for name in sys.modules
        )

    # Both constructors initialize temporary parameters before strict loading.
    # Start the inference/physical trajectory from the same runtime RNG state.
    torch.manual_seed(915)
    traces = []
    for step in range(20):
        flat = observations["tokenizer" if original else "tokenizer_obs"].clone()
        mode = torch.full((4,), step % 3, device="cuda", dtype=torch.long)
        flat[:, :3] = torch.nn.functional.one_hot(mode, 3).float()
        if original:
            inputs = {
                "actor_obs": observations["actor_obs"][:, None],
                "tokenizer": flat[:, None],
            }
            with torch.no_grad():
                actions = policy.act_pure_inference(inputs).squeeze(1)
        else:
            encoder = torch.cat([mode[:, None], flat], -1)
            actions = torch.from_numpy(
                pipeline(encoder, observations["actor_obs"])["action"]
            ).cuda()
        before = tensors(
            {
                ("tokenizer_obs" if name == "tokenizer" else name): value
                for name, value in observations.items()
            }
        )
        observations, rewards, dones, infos = environment.step({"actions": actions})
        robot = environment.env.scene["robot"]
        traces.append(
            tensors(
                dict(
                    observation=before,
                    action=actions,
                    reward=rewards,
                    done=dones,
                    timeout=infos["time_outs"],
                    root=robot.data.root_state_w,
                    joints=robot.data.joint_pos,
                    joint_velocity=robot.data.joint_vel,
                    body_position=robot.data.body_pos_w,
                    body_orientation=robot.data.body_quat_w,
                    contacts=environment.env.scene["contact_forces"].data.net_forces_w,
                )
            )
        )
    torch.save(
        dict(
            neural=neural,
            schema=dict(schema),
            traces=traces,
            rng=torch.cuda.get_rng_state(),
        ),
        args.output_dir / (args.worker + ".pt"),
    )
    print(
        args.worker, "SONIC original/native G1 physical inference complete", flush=True
    )
    environment.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "fixture", "assets", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "repeat", "native"])
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    for name in ["upstream", "artifact", "fixture", "assets", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        tests=[],
        scope="Original released SONIC G1 manager "
        "environment and Actor versus selected native environment and public "
        "SONICPipeline, 20 physical steps, four environments, three modality "
        "branches, complete observation/action/state/reward/reset/RNG traces. "
        "Isaac Lab 2.3.2/Sim5.1 FP32. Synthetic short paired robot/SMPL-coordinate "
        "reference; plane terrain. Full real datasets, rough terrain, PPO and "
        "long-horizon behavior excluded.",
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in [] if args.reuse_results else ["upstream", "repeat", "native"]:
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in ["upstream", "artifact", "fixture", "assets", "output_dir"]:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(command + ["--worker", kind], cwd=ROOT, check=True)
        assert (
            args.output_dir / (kind + ".pt")
        ).is_file(), "Worker exited before saving its result"
    left = torch.load(args.output_dir / "upstream.pt", weights_only=True)
    repeat = torch.load(args.output_dir / "repeat.pt", weights_only=True)
    right = torch.load(args.output_dir / "native.pt", weights_only=True)
    bounds = []
    for a, b, c in zip(left["traces"], repeat["traces"], right["traces"]):
        source_contacts, repeated_contacts, native_contacts = (
            a.pop("contacts"),
            b.pop("contacts"),
            c.pop("contacts"),
        )
        bound = (source_contacts - repeated_contacts).abs().max().item()
        error = (source_contacts - native_contacts).abs().max().item()
        assert error <= bound, (
            "contact diagnostic exceeds original self-repeat bound",
            error,
            bound,
        )
        bounds.append(
            dict(source_self_absolute_bound=bound, native_absolute_error=error)
        )
    report["contact_diagnostics"] = bounds
    try:
        compare_full(left, right, "", report["tests"])
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("SONIC original physical environment and native public Pipeline accepted")


if __name__ == "__main__":
    main()
