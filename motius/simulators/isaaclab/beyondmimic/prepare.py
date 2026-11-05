"""Finite reference-data conversion using the published Isaac Lab FK graph."""
from pathlib import Path
import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg,AssetBaseCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from .robot import G1_CYLINDER_CFG
from motius.motion.representation.beyondmimic import BeyondMimicCSVReference

@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")



def prepare_csv_motion(input_file,output_file,asset_root,*,input_fps=30,output_fps=50,device='cuda:0',frame_range=None):
    import json
    config=sim_utils.SimulationCfg(device=str(device));config.dt=1.0/output_fps
    simulation=SimulationContext(config)
    scene_cfg=ReplayMotionsSceneCfg(num_envs=1,env_spacing=2.0)
    asset=Path(asset_root).expanduser().resolve()/'unitree_description/urdf/g1/main.urdf'
    if not asset.is_file():raise FileNotFoundError(asset)
    scene_cfg.robot.spawn.asset_path=str(asset)
    output=Path(output_file).expanduser().resolve();output.parent.mkdir(parents=True,exist_ok=True)
    scene_cfg.robot.spawn.usd_dir=str(output.parent/'isaac_assets')
    scene=InteractiveScene(scene_cfg);simulation.reset()
    motion=BeyondMimicCSVReference(str(input_file),input_fps,output_fps,simulation.device,frame_range)
    order=Path(__file__).resolve().parents[3]/'configuration/beyondmimic/csv_joint_order.json'
    joints=json.loads(order.read_text());robot=scene['robot'];indexes=robot.find_joints(joints,preserve_order=True)[0]
    fields=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
    log={name:[] for name in fields}
    for _ in range(motion.output_frames):
        (position,rotation,linear_velocity,angular_velocity,joint_positions,joint_velocities),_=motion.get_next_state()
        root=robot.data.default_root_state.clone();root[:,:3]=position;root[:,:2]+=scene.env_origins[:,:2]
        root[:,3:7]=rotation;root[:,7:10]=linear_velocity;root[:,10:]=angular_velocity
        robot.write_root_state_to_sim(root)
        positions=robot.data.default_joint_pos.clone();velocities=robot.data.default_joint_vel.clone()
        positions[:,indexes]=joint_positions;velocities[:,indexes]=joint_velocities
        robot.write_joint_state_to_sim(positions,velocities)
        simulation.render();scene.update(simulation.get_physics_dt())
        for name in fields:log[name].append(getattr(robot.data,name)[0,:].cpu().numpy().copy())
    log={name:np.stack(value,axis=0) for name,value in log.items()};log['fps']=[output_fps]
    np.savez(output,**log)
    return output
