"""Native assembly of MotionBricks networks, primitive controller and robot geometry."""
from pathlib import Path
from types import SimpleNamespace
import random
import numpy as np
import torch
from motius.configuration.motionbricks import checkpoint_root,source_config,build_representation
from .inference import motion_inference
from .full_agent import full_navigation_agent
from .controllers import WASD_controller,random_controller


def _seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def build_navigation(bundle,*,controller='wasd',clips='G1',seed=None,**overrides):
    import mujoco
    from omegaconf import OmegaConf
    root=checkpoint_root(bundle.artifact_root)
    cfg_pose=source_config(root,'pose');cfg_root=source_config(root,'root')
    precision=cfg_root.get('matmul_precision')
    if precision is not None:torch.set_float32_matmul_precision(precision)
    representation=build_representation(root)
    inference=motion_inference(bundle,representation,cfg_pose.model.args,cfg_root.model.args)
    assets=Path(__file__).resolve().parents[2]/'configuration/motionbricks_assets'
    options=dict(controller=controller,clips=clips,generate_dt=2.,has_viewer=0,
        humanoid_scene_xml=str(assets/'skeletons/g1/scene_29dof.xml'),skeleton_xml=str(assets/'skeletons/g1/g1.xml'),
        target_root_realignment=1,source_root_realignment=1,force_canonicalization=1,
        skip_ending_target_cond=0,pre_filter_qpos=1,lookat_movement_direction=0,
        random_speed_scale=0,speed_scale=[1.,1.],clips_ckpt=str(root/'G1-clip.ckpt'),reprocess_clips=0,allowed_mode=None)
    options.update(overrides);args=SimpleNamespace(**options)
    if args.has_viewer:raise ValueError('Use the visualization API for an interactive viewer')
    if args.reprocess_clips:raise ValueError('Clip preparation belongs in the dataset stage; supply the released G1-clip.ckpt')
    if not Path(args.clips_ckpt).is_file():raise FileNotFoundError(args.clips_ckpt)
    if seed is not None:_seed(seed)
    full_agent=full_navigation_agent(inference,None,device=str(bundle.device),
        speed_scale=args.speed_scale if args.random_speed_scale else [1.,1.],
        target_root_realignment=args.target_root_realignment,source_root_realignment=args.source_root_realignment,
        force_canonicalization=args.force_canonicalization,skeleton_xml=args.skeleton_xml,
        skip_ending_target_cond=args.skip_ending_target_cond,filter_qpos=args.pre_filter_qpos,
        clips=args.clips,ckpt_path=args.clips_ckpt,reprocess_clips=False).to(bundle.device)
    common=dict(lookat_movement_direction=args.lookat_movement_direction,clips=args.clips,
        min_token=inference._args['min_tokens'],max_token=inference._args['max_tokens'])
    if args.controller=='wasd':control=WASD_controller(**common)
    elif args.controller=='random':control=random_controller(disable_running=getattr(args,'disable_running',True),
        new_control_dt=getattr(args,'new_control_dt',2.),max_angle_change_between_controls=getattr(args,'max_angle_change_between_controls',.5*np.pi),**common)
    else:raise ValueError(args.controller)
    model=mujoco.MjModel.from_xml_path(args.humanoid_scene_xml);model.opt.timestep=1/representation.fps
    data=mujoco.MjData(model)
    return SimpleNamespace(args=args,full_agent=full_agent,controller=control,mj_model=model,mj_data=data,inferencer=inference)


@torch.inference_mode()
def rollout_navigation(agent,steps=120):
    agent.full_agent.reset();values=[];steps=int(steps)
    if steps<1:raise ValueError('steps must be positive')
    idle_tail=min(100,max(0,steps//5))
    for step in range(steps):
        qpos=agent.full_agent.get_next_frame();values.append(np.asarray(qpos,dtype=np.float32).copy())
        context=agent.full_agent.get_context_mujoco_qpos();agent.mj_data.qpos[:]=qpos
        control=agent.controller.generate_control_signals(None,agent.mj_model,agent.mj_data,visualize=False,
            control_info=dict(force_idle=idle_tail>0 and step>=steps-idle_tail,allowed_mode=agent.args.allowed_mode))
        control['context_mujoco_qpos']=context
        agent.full_agent.generate_new_frames(control,agent.controller.get_controller_dt()*agent.args.generate_dt)
    return dict(qpos=np.stack(values,axis=0),fps=30,representation='motionbricks_g1_414')
