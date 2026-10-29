"""Instantiate native G1 tracking terms after Isaac Lab AppLauncher startup."""
from pathlib import Path


class BeyondMimicVectorEnvironment:
    """The published vector step/timeout contract, without an RSL runner."""
    def __init__(self,environment,clip_actions=None):
        self.environment=environment;self.num_envs=environment.num_envs;self.device=environment.device
        self.num_actions=environment.action_manager.total_action_dim;self.clip_actions=clip_actions
        self.environment.reset()

    def get_observations(self):
        observations=self.environment.observation_manager.compute()
        return observations['policy'],{'observations':observations}

    def step(self,actions):
        import torch
        if self.clip_actions is not None:actions=torch.clamp(actions,-self.clip_actions,self.clip_actions)
        observations,rewards,terminated,truncated,extras=self.environment.step(actions)
        extras['observations']=observations
        if not self.environment.cfg.is_finite_horizon:extras['time_outs']=truncated
        return observations['policy'],rewards,(terminated|truncated).to(dtype=torch.long),extras

    def close(self):self.environment.close()


def create_beyondmimic_environment(motion_file,asset_root,*,num_envs=4096,device='cuda:0',seed=42,
                                  variant='flat',output_dir=None,debug_vis=None):
    from isaaclab.envs import ManagerBasedRLEnv
    from .g1_config import G1FlatEnvCfg,G1FlatWoStateEstimationEnvCfg,G1FlatLowFreqEnvCfg
    classes=dict(flat=G1FlatEnvCfg,without_state_estimation=G1FlatWoStateEstimationEnvCfg,low_frequency=G1FlatLowFreqEnvCfg)
    if variant not in classes:raise ValueError('Unknown G1 tracking variant: '+variant)
    motion_file=Path(motion_file).expanduser().resolve()
    robot=Path(asset_root).expanduser().resolve()/'unitree_description/urdf/g1/main.urdf'
    for path in [motion_file,robot]:
        if not path.is_file():raise FileNotFoundError(path)
    cfg=classes[variant]();cfg.scene.num_envs=int(num_envs);cfg.sim.device=str(device);cfg.seed=int(seed)
    cfg.scene.robot.spawn.asset_path=str(robot)
    if output_dir is not None:
        directory=Path(output_dir).expanduser().resolve()/'isaac_assets';directory.mkdir(parents=True,exist_ok=True)
        cfg.scene.robot.spawn.usd_dir=str(directory)
    cfg.commands.motion.motion_file=str(motion_file)
    if debug_vis is not None:
        cfg.commands.motion.debug_vis=bool(debug_vis)
        cfg.scene.contact_forces.debug_vis=bool(debug_vis)
    return BeyondMimicVectorEnvironment(ManagerBasedRLEnv(cfg=cfg))


def tracking_metadata(environment):
    """Read the source deployment contract from the actual simulator managers."""
    env=getattr(environment,'environment',environment)
    observation_names=env.observation_manager.active_terms['policy']
    group=env.observation_manager.cfg.policy
    if group.history_length is not None:
        history_lengths=[group.history_length]*len(observation_names)
    else:
        history_lengths=[group.to_dict()[name]['history_length'] or 1 for name in observation_names]
    robot=env.scene['robot'];command=env.command_manager.get_term('motion')
    return dict(joint_names=robot.data.joint_names,
        joint_stiffness=robot.data.joint_stiffness[0].cpu().tolist(),
        joint_damping=robot.data.joint_damping[0].cpu().tolist(),
        default_joint_pos=robot.data.default_joint_pos_nominal.cpu().tolist(),
        command_names=env.command_manager.active_terms,observation_names=observation_names,
        observation_history_lengths=history_lengths,
        action_scale=env.action_manager.get_term('joint_pos')._scale[0].cpu().tolist(),
        anchor_body_name=command.cfg.anchor_body_name,body_names=command.cfg.body_names)
