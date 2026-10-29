"""Isaac Lab scene and physical settings for the released SONIC G1 controller."""

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import isaaclab.sim as sim_utils
from .robot import (
    G1_CYLINDER_MODEL_12_DEX_CFG,
    G1_MODEL_12_ACTION_SCALE,
    G1_ISAACLAB_TO_MUJOCO_MAPPING,
)


@configclass
class SonicG1SceneCfg(InteractiveSceneCfg):
    def __init__(self, config):
        super().__init__()
        self.num_envs = config.get("num_envs", 4096)
        self.env_spacing = config.get("env_spacing", 2.5)
        self.replicate_physics = config.get("replicate_physics", True)
        terrain_type = config.get("terrain_type", "plane")
        terrain = dict(
            prim_path="/World/ground", terrain_type="plane", collision_group=-1
        )
        if terrain_type == "trimesh":
            from .mdp.terrain import ROUGH_TERRAINS_CFG

            terrain.update(
                terrain_type="generator",
                terrain_generator=ROUGH_TERRAINS_CFG,
                max_init_terrain_level=10,
                collision_group=1,
                debug_vis=False,
            )
        elif terrain_type != "plane":
            raise ValueError("Unsupported SONIC terrain: " + terrain_type)
        self.terrain = TerrainImporterCfg(
            **terrain,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
            visual_material=sim_utils.MdlFileCfg(
                mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
                project_uvw=True,
            ),
        )
        self.robot = G1_CYLINDER_MODEL_12_DEX_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )
        self.light = AssetBaseCfg(
            prim_path="/World/light",
            spawn=sim_utils.DistantLightCfg(
                color=(0.75, 0.75, 0.75),
                intensity=3000.0,
            ),
        )
        self.sky_light = AssetBaseCfg(
            prim_path="/World/skyLight",
            spawn=sim_utils.DomeLightCfg(
                color=(0.13, 0.13, 0.13),
                intensity=1000.0,
            ),
        )
        self.contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/.*",
            history_length=3,
            track_air_time=True,
            force_threshold=10.0,
            debug_vis=False,
        )


@configclass
class SonicG1EnvCfg(ManagerBasedRLEnvCfg):
    def __init__(
        self,
        config,
        *,
        actions,
        observations,
        rewards,
        terminations,
        commands,
        events,
        curriculum,
        recorders,
    ):
        super().__init__()
        self.config = config
        self.scene = SonicG1SceneCfg(config)
        self.actions, self.observations = actions, observations
        self.rewards, self.terminations, self.commands = rewards, terminations, commands
        self.events, self.curriculum, self.recorders = events, curriculum, recorders
        self.decimation = config.get("decimation", 4)
        self.episode_length_s = config.get("episode_length_s", 10.0)
        self.sim.dt = config.get("sim_dt", 0.005)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.physx.gpu_collision_stack_size = 2 ** config.get(
            "gpu_collision_stack_size_exp", 26
        )
        viewer = config.get("viewer", {})
        self.viewer = ViewerCfg(
            eye=viewer.get("eye", [4.5, 0.0, 4.0]),
            lookat=viewer.get("lookat", [0.0, 0.0, 0.0]),
        )
        self.actions.joint_pos.scale = G1_MODEL_12_ACTION_SCALE
        self.isaaclab_to_mujoco_mapping = G1_ISAACLAB_TO_MUJOCO_MAPPING
