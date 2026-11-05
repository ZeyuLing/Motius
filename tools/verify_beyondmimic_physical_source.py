"""Compare real Isaac Lab G1 FK conversion and tracking with original source."""

from __future__ import annotations
import argparse, ast, json, os, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
from verify_prompthmr_video_parity import tensors, compare_full


def original_csv_conversion(args, launcher):
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.assets import ArticulationCfg, AssetBaseCfg
    from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
    from isaaclab.sim import SimulationContext
    from isaaclab.utils import configclass
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
    from isaaclab.utils.math import (
        axis_angle_from_quat,
        quat_conjugate,
        quat_mul,
        quat_slerp,
    )
    from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

    G1_CYLINDER_CFG.spawn.asset_path = str(
        args.assets / 'unitree_description/urdf/g1/main.urdf'
    )
    file = args.upstream / 'scripts/csv_to_npz.py'
    tree = ast.parse(file.read_text())
    selected = [
        node
        for node in tree.body
        if (
            isinstance(node, ast.ClassDef)
            and node.name in {'ReplayMotionsSceneCfg', 'MotionLoader'}
        )
        or (isinstance(node, ast.FunctionDef) and node.name == 'run_simulator')
    ]
    # Run the original computation through its authoritative NPZ save, then
    # return instead of uploading data to W&B and looping forever.
    for node in selected:
        if not isinstance(node, ast.FunctionDef):
            continue
        for branch in ast.walk(node):
            if not isinstance(branch, ast.If):
                continue
            for index, statement in enumerate(branch.body):
                if isinstance(statement, ast.Expr) and isinstance(
                    statement.value, ast.Call
                ):
                    call = statement.value
                    if (
                        isinstance(call.func, ast.Attribute)
                        and isinstance(call.func.value, ast.Name)
                        and call.func.value.id == 'np'
                        and call.func.attr == 'savez'
                    ):
                        call.args[0] = ast.Constant(
                            str(args.output_dir / 'upstream_motion.npz')
                        )
                        branch.body = branch.body[: index + 1] + [
                            ast.Return(value=None)
                        ]
                        break
    namespace = dict(
        np=np,
        torch=torch,
        sim_utils=sim_utils,
        ArticulationCfg=ArticulationCfg,
        AssetBaseCfg=AssetBaseCfg,
        InteractiveSceneCfg=InteractiveSceneCfg,
        InteractiveScene=InteractiveScene,
        configclass=configclass,
        G1_CYLINDER_CFG=G1_CYLINDER_CFG,
        ISAAC_NUCLEUS_DIR=ISAAC_NUCLEUS_DIR,
        axis_angle_from_quat=axis_angle_from_quat,
        quat_conjugate=quat_conjugate,
        quat_mul=quat_mul,
        quat_slerp=quat_slerp,
        simulation_app=launcher.app,
        args_cli=SimpleNamespace(
            input_file=str(args.csv), input_fps=30, output_fps=50, frame_range=None
        ),
    )
    exec(
        compile(
            ast.fix_missing_locations(ast.Module(body=selected, type_ignores=[])),
            str(file),
            'exec',
        ),
        namespace,
    )
    config = sim_utils.SimulationCfg(device='cuda:0')
    config.dt = 1 / 50
    simulation = SimulationContext(config)
    scene_config = namespace['ReplayMotionsSceneCfg'](num_envs=1, env_spacing=2.0)
    scene_config.robot.spawn.usd_dir = str(args.output_dir / 'upstream-assets')
    scene = InteractiveScene(scene_config)
    simulation.reset()
    joints = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / 'motius/configuration/beyondmimic/csv_joint_order.json'
        ).read_text()
    )
    namespace['run_simulator'](simulation, scene, joints)


def tracking(args):
    import numpy as np
    import torch

    motion = args.motion or args.output_dir / 'upstream_motion.npz'
    if args.checkpoint is None:
        raise ValueError('Tracking comparison requires --checkpoint')
    if args.worker == 'upstream':
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import (
            G1FlatEnvCfg,
        )
        from rsl_rl.modules import ActorCritic, EmpiricalNormalization

        config = G1FlatEnvCfg()
        config.scene.num_envs = 4
        config.sim.device = 'cuda:0'
        config.seed = 421
        config.commands.motion.motion_file = str(motion)
        config.commands.motion.debug_vis = False
        config.scene.contact_forces.debug_vis = False
        config.scene.robot.spawn.asset_path = str(
            args.assets / 'unitree_description/urdf/g1/main.urdf'
        )
        config.scene.robot.spawn.usd_dir = str(
            args.output_dir / 'upstream-tracking-assets'
        )
        environment = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=config))
        base = environment.unwrapped
        state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
        policy = (
            ActorCritic(
                160,
                286,
                29,
                actor_hidden_dims=[512, 256, 128],
                critic_hidden_dims=[512, 256, 128],
                activation='elu',
            )
            .cuda()
            .eval()
        )
        policy.load_state_dict(state['model_state_dict'], strict=True)
        normalizer = EmpiricalNormalization([160], until=1e8).cuda().eval()
        normalizer.load_state_dict(state['obs_norm_state_dict'], strict=True)
    else:
        from motius.simulators.isaaclab.beyondmimic import (
            create_beyondmimic_environment,
        )
        from motius.models.beyondmimic import BeyondMimicPolicyBundle
        from motius.pipelines.beyondmimic import BeyondMimicPipeline

        environment = create_beyondmimic_environment(
            motion,
            args.assets,
            num_envs=4,
            device='cuda:0',
            seed=421,
            output_dir=args.output_dir / 'native-tracking',
            debug_vis=False,
        )
        base = environment.environment
        bundle = BeyondMimicPolicyBundle.from_original_checkpoint(
            args.checkpoint, device='cuda'
        ).eval()
        reference = {
            name: getattr(base.command_manager.get_term('motion').motion, name)
            .cpu()
            .numpy()
            for name in [
                'joint_pos',
                'joint_vel',
                'body_pos_w',
                'body_quat_w',
                'body_lin_vel_w',
                'body_ang_vel_w',
            ]
        }
        pipeline = BeyondMimicPipeline(bundle, reference_motion=reference)
    try:
        observations, extras = environment.get_observations()
        output = {
            'initial': tensors(
                dict(
                    observations=observations,
                    critic=extras['observations']['critic'],
                    root=base.scene['robot'].data.root_state_w,
                    joint_pos=base.scene['robot'].data.joint_pos,
                    joint_vel=base.scene['robot'].data.joint_vel,
                    time_steps=base.command_manager.get_term('motion').time_steps,
                )
            )
        }
        for step in range(args.steps):
            with torch.no_grad():
                if args.worker == 'upstream':
                    # Use the actual original deterministic inference method.
                    actions = (
                        policy.act_inference(normalizer(observations)).cpu().numpy()
                    )
                else:
                    times = (
                        base.command_manager.get_term('motion').time_steps.cpu().numpy()
                    )
                    actions = pipeline(observations, times)['actions']
            observations, rewards, dones, extras = environment.step(
                torch.as_tensor(actions, device=environment.device)
            )
            robot = base.scene['robot']
            command = base.command_manager.get_term('motion')
            output[str(step)] = tensors(
                dict(
                    actions=actions,
                    observations=observations,
                    critic=extras['observations']['critic'],
                    rewards=rewards,
                    dones=dones,
                    root=robot.data.root_state_w,
                    joint_pos=robot.data.joint_pos,
                    joint_vel=robot.data.joint_vel,
                    body_pos=robot.data.body_pos_w,
                    body_rot=robot.data.body_quat_w,
                    time_steps=command.time_steps,
                    cuda_rng=torch.cuda.get_rng_state(),
                )
            )
        torch.save(output, args.output_dir / ('tracking_' + args.worker + '.pt'))
    finally:
        environment.close()


def training(args):
    """Run the actual original and native online trainers in the real G1 env."""
    import torch
    from verify_beyondmimic_ppo_source import published_options

    motion = args.motion or args.output_dir / 'upstream_motion.npz'
    if args.worker == 'upstream':
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
        from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import (
            G1FlatEnvCfg,
        )

        config = G1FlatEnvCfg()
        config.scene.num_envs = 4
        config.sim.device = 'cuda:0'
        config.seed = 421
        config.commands.motion.motion_file = str(motion)
        config.commands.motion.debug_vis = False
        config.scene.contact_forces.debug_vis = False
        config.scene.robot.spawn.asset_path = str(
            args.assets / 'unitree_description/urdf/g1/main.urdf'
        )
        config.scene.robot.spawn.usd_dir = str(
            args.output_dir / 'upstream-training-assets'
        )
        environment = RslRlVecEnvWrapper(ManagerBasedRLEnv(cfg=config))
        base = environment.unwrapped
    else:
        from motius.simulators.isaaclab.beyondmimic import (
            create_beyondmimic_environment,
        )

        environment = create_beyondmimic_environment(
            motion,
            args.assets,
            num_envs=4,
            device='cuda:0',
            seed=421,
            output_dir=args.output_dir / 'native-training',
            debug_vis=False,
        )
        base = environment.environment
    trace = []
    snapshots = []
    step = environment.step

    def traced(actions):
        obs, rewards, dones, extras = step(actions)
        trace.append(
            tensors(
                dict(
                    actions=actions,
                    observations=obs,
                    rewards=rewards,
                    dones=dones,
                    critic=extras['observations']['critic'],
                    time_outs=extras.get('time_outs'),
                    root=base.scene['robot'].data.root_state_w,
                    joint_pos=base.scene['robot'].data.joint_pos,
                    joint_vel=base.scene['robot'].data.joint_vel,
                    time_steps=base.command_manager.get_term('motion').time_steps,
                )
            )
        )
        return obs, rewards, dones, extras

    environment.step = traced

    def snapshot(policy, actor_norm, critic_norm, optimizer, learning_rate, rollout):
        snapshots.append(
            tensors(
                dict(
                    state=policy.state_dict(),
                    actor_norm=actor_norm.state_dict(),
                    critic_norm=critic_norm.state_dict(),
                    optimizer=optimizer.state_dict(),
                    learning_rate=learning_rate,
                    gradients={
                        key: value.grad for key, value in policy.named_parameters()
                    },
                    rollout=rollout,
                    cuda_rng=torch.cuda.get_rng_state(),
                    cpu_rng=torch.get_rng_state(),
                )
            )
        )

    neural, algorithm = published_options(args.upstream)
    try:
        if args.worker == 'upstream':
            from rsl_rl.runners import OnPolicyRunner

            options = dict(
                policy={**neural, 'class_name': 'ActorCritic'},
                algorithm={**algorithm, 'class_name': 'PPO'},
                num_steps_per_env=24,
                save_interval=500,
                empirical_normalization=True,
            )
            runner = OnPolicyRunner(environment, options, device='cuda')
            runner.disable_logs = True
            state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            runner.alg.policy.load_state_dict(state['model_state_dict'], strict=True)
            runner.obs_normalizer.load_state_dict(
                state['obs_norm_state_dict'], strict=True
            )
            runner.privileged_obs_normalizer.load_state_dict(
                state['privileged_obs_norm_state_dict'], strict=True
            )
            update = runner.alg.update

            def capture():
                result = update()
                storage = runner.alg.storage
                rollout = {
                    name: getattr(
                        storage,
                        (
                            'privileged_observations'
                            if name == 'critic_observations'
                            else name
                        ),
                    )
                    for name in [
                        'observations',
                        'critic_observations',
                        'actions',
                        'values',
                        'actions_log_prob',
                        'mu',
                        'sigma',
                        'returns',
                        'advantages',
                    ]
                }
                snapshot(
                    runner.alg.policy,
                    runner.obs_normalizer,
                    runner.privileged_obs_normalizer,
                    runner.alg.optimizer,
                    runner.alg.learning_rate,
                    rollout,
                )
                return result

            runner.alg.update = capture
            torch.manual_seed(1234)
            runner.learn(2, init_at_random_ep_len=True)
        else:
            from accelerate import Accelerator
            from motius.runner.online_tracking import (
                build_beyondmimic_runner,
                beyondmimic_training_options,
            )

            options = beyondmimic_training_options()
            options['policy'] = neural
            options['algorithm'] = algorithm
            runner = build_beyondmimic_runner(
                environment,
                args.output_dir / 'native_physical_runner',
                accelerator=Accelerator(mixed_precision='no'),
                options=options,
                max_iterations=2,
                checkpoint=args.checkpoint,
                random_episode_lengths=True,
            )
            update = runner.trainer.train_step

            def capture(rollout):
                result = update(rollout)
                snapshot(
                    runner.bundle.network,
                    runner.bundle.actor_normalizer,
                    runner.bundle.critic_normalizer,
                    runner.optimizers['policy'],
                    runner.trainer.learning_rate,
                    rollout,
                )
                return result

            runner.trainer.train_step = capture
            torch.manual_seed(1234)
            runner.train()
            assert runner.global_step == 2
            assert runner.bundle.actor_normalizer.count.item() == 192
        torch.save(
            dict(trace=trace, snapshots=snapshots),
            args.output_dir / ('training_' + args.worker + '.pt'),
        )
    finally:
        environment.close()


def worker(args):
    import resource

    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    if args.phase == 'training' and args.worker == 'native':
        import accelerate  # Resolve its installed dependencies before Kit's pip archive.
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True, device='cuda:0', kit_args='--allow-root')
    try:
        import torch

        torch.set_num_threads(4)
        torch.manual_seed(421)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if args.worker == 'upstream':
            sys.path.insert(0, str(args.upstream / 'source/whole_body_tracking'))
        if args.phase == 'conversion':
            if args.worker == 'upstream':
                original_csv_conversion(args, launcher)
            else:
                from motius.simulators.isaaclab.beyondmimic.prepare import (
                    prepare_csv_motion,
                )

                prepare_csv_motion(
                    args.csv,
                    args.output_dir / 'native_motion.npz',
                    args.assets,
                    input_fps=30,
                    output_fps=50,
                )
        elif args.phase == 'tracking':
            tracking(args)
        else:
            training(args)
    except BaseException:
        import traceback

        error = traceback.format_exc()
        (args.output_dir / (args.phase + '_' + args.worker + '_error.txt')).write_text(
            error
        )
        print(error, file=sys.stderr, flush=True)
        raise
    finally:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['upstream', 'assets', 'csv', 'output-dir']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument(
        '--phase', choices=['conversion', 'tracking', 'training'], default='conversion'
    )
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--motion', type=Path)
    parser.add_argument('--steps', type=int, default=60)
    parser.add_argument('--worker', choices=['upstream', 'native'])
    args = parser.parse_args()
    for name in ['upstream', 'assets', 'csv', 'output_dir']:
        setattr(args, name, getattr(args, name).resolve())
    for name in ['checkpoint', 'motion']:
        if getattr(args, name) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scope = 'Real G1 CSV-to-NPZ conversion with Isaac Sim 4.5 / Isaac Lab 2.1 on H200; original FK/velocity computation stops after the authoritative NPZ save and before W&B upload. Synthetic nonzero joint/reference motion; not trained-policy tracking.'
    if args.phase == 'tracking':
        scope = 'Actual original G1 environment/RSL deterministic inference versus native environment/Bundle/Pipeline; same seeded original neural checkpoint, real source-pinned robot and FK-derived reference, 4 physical environments. Not pretrained tracking quality or full training protocol.'
    if args.phase == 'training':
        scope = 'Actual original RSL OnPolicyRunner.learn versus native AccelerateRunner, two 24x4 real G1 physical rollouts with random initial episode lengths, GAE, 40 PPO minibatch updates, normalizers, gradients and Adam state. Same seeded checkpoint and synthetic FK-derived reference; not full training or pretrained-policy quality.'
    report = dict(
        complete=False,
        scope=scope,
        upstream_commit=subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True
        ).strip(),
        tests=[],
    )
    path = args.output_dir / (
        args.phase + '_source_parity.json'
        if args.phase != 'conversion'
        else 'source_parity.json'
    )
    path.write_text(json.dumps(report, indent=2) + '\n')
    for kind in ['upstream', 'native']:
        expected = args.output_dir / (
            kind + '_motion.npz'
            if args.phase == 'conversion'
            else args.phase + '_' + kind + '.pt'
        )
        expected.unlink(missing_ok=True)
        (args.output_dir / (args.phase + '_' + kind + '_error.txt')).unlink(
            missing_ok=True
        )
        env = {
            **os.environ,
            'PYTHONPATH': str(root),
            'OMNI_KIT_ACCEPT_EULA': 'YES',
            'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        }
        subprocess.run(
            [sys.executable, '-u', __file__, *sys.argv[1:], '--worker', kind],
            cwd=args.upstream if kind == 'upstream' else root,
            env=env,
            check=True,
        )
        if not expected.is_file():
            raise RuntimeError(
                kind + ' worker did not produce its requested physical result'
            )
    import torch, numpy as np

    torch.set_num_threads(4)
    if args.phase == 'conversion':
        with np.load(args.output_dir / 'upstream_motion.npz') as data:
            original = {name: data[name] for name in data.files}
        with np.load(args.output_dir / 'native_motion.npz') as data:
            native = {name: data[name] for name in data.files}
    else:
        original = torch.load(
            args.output_dir / (args.phase + '_upstream.pt'), weights_only=True
        )
        native = torch.load(
            args.output_dir / (args.phase + '_native.pt'), weights_only=True
        )
    try:
        compare_full(tensors(original), tensors(native), '', report['tests'])
    finally:
        path.write_text(json.dumps(report, indent=2) + '\n')
    report.update(complete=True, torch_version=torch.__version__)
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(
        'Original and native real G1 physical ' + args.phase + ' comparisons are exact'
    )


if __name__ == '__main__':
    main()
