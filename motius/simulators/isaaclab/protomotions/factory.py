# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Utility functions for building environment components.

This module provides helper functions to create terrain, scene_lib, motion_lib,
and simulator objects from their configs, reducing boilerplate in entry scripts.
"""
from typing import Optional, Dict
import torch
from motius.simulators.isaaclab.protomotions.support.construction import get_class


def build_terrain_from_config(terrain_config, num_envs: int, device: torch.device):
    """Build Terrain from config.

    Args:
        terrain_config: TerrainConfig or None (exception: can be None for no terrain)
        num_envs: Number of environments
        device: PyTorch device

    Returns:
        Terrain instance or None if config is None
    """
    if terrain_config is None:
        return None
    from motius.simulators.terrains.protomotions.terrain import Terrain

    return Terrain(config=terrain_config, num_envs=num_envs, device=device)


def build_scene_lib_from_config(
    scene_lib_config,
    num_envs: int,
    device: torch.device,
    terrain,
    scene_weights: Optional[list] = None,
):
    """Build SceneLib from config.

    Always returns a SceneLib instance. If config.scene_file is None,
    creates an empty SceneLib (Null Object pattern).

    Args:
        scene_lib_config: SceneLibConfig (required, scene_file can be None for empty)
        num_envs: Number of environments
        device: PyTorch device
        terrain: Terrain instance (required by SceneLib)
        scene_weights: Optional scene weights for curriculum learning

    Returns:
        SceneLib instance (empty if scene_file is None)
    """
    from motius.simulators.scenes.protomotions import SceneLib

    return SceneLib(
        config=scene_lib_config,
        num_envs=num_envs,
        scenes=scene_lib_config.inline_scenes,
        device=device,
        terrain=terrain,
        scene_weights=scene_weights,
    )


def build_motion_lib_from_config(motion_lib_config, device: torch.device):
    """Build MotionLib from config.

    Always returns a MotionLib instance. If config.motion_file is None,
    creates an empty MotionLib (Null Object pattern).

    Note: Contact smoothing is NOT applied here - it's the Env's responsibility
    to modify the motion_lib during initialization based on its config.

    Args:
        motion_lib_config: MotionLibConfig (required, motion_file can be None for empty)
        device: PyTorch device

    Returns:
        MotionLib instance (empty if motion_file is None)
    """
    from motius.datasets.protomotions_motion import MotionLib

    return MotionLib(config=motion_lib_config, device=device)


def build_simulator_from_config(
    simulator_config,
    robot_config,
    terrain,
    scene_lib,
    device: torch.device,
    **simulator_extra_params,
):
    """Build Simulator from config.

    Creates simulator shell (deferred initialization - will be finalized by Env).

    Args:
        simulator_config: SimulatorConfig
        robot_config: RobotConfig
        terrain: Terrain instance
        scene_lib: SceneLib instance or None
        device: PyTorch device
        **simulator_extra_params: Simulator-specific params (e.g., simulation_app for IsaacLab)

    Returns:
        Simulator instance (shell, not yet initialized)
    """
    SimulatorClass = get_class(simulator_config._target_)
    return SimulatorClass(
        config=simulator_config,
        robot_config=robot_config,
        terrain=terrain,
        scene_lib=scene_lib,
        device=device,
        **simulator_extra_params,
    )


def build_all_components(
    terrain_config,
    scene_lib_config,
    motion_lib_config,
    simulator_config,
    robot_config,
    device: torch.device,
    save_dir: Optional[str] = None,
    **simulator_extra_params,
) -> Dict:
    """Build all environment components from configs.

    Convenience function that builds terrain, scene_lib, motion_lib, and simulator.

    Args:
        terrain_config: TerrainConfig (or None for no terrain - exception)
        scene_lib_config: SceneLibConfig (always provided, scene_file can be None for empty)
        motion_lib_config: MotionLibConfig (always provided, motion_file can be None for empty)
        simulator_config: SimulatorConfig
        robot_config: RobotConfig
        device: PyTorch device
        save_dir: Optional save directory for loading motion weights as scene weights
        **simulator_extra_params: Simulator-specific params (e.g., simulation_app for IsaacLab)

    Returns:
        Dict with keys: terrain, scene_lib, motion_lib, simulator
    """
    terrain = build_terrain_from_config(
        terrain_config, simulator_config.num_envs, device
    )
    scene_weights = None
    if save_dir and motion_lib_config.motion_file:
        from motius.simulators.isaaclab.protomotions.mdp.base_env.env import BaseEnv

        scene_weights = BaseEnv.apply_motion_weights_to_scene_weights(
            save_dir=save_dir, motion_file=motion_lib_config.motion_file, device=device
        )
    scene_lib = build_scene_lib_from_config(
        scene_lib_config, simulator_config.num_envs, device, terrain, scene_weights
    )
    motion_lib = build_motion_lib_from_config(motion_lib_config, device)
    simulator = build_simulator_from_config(
        simulator_config,
        robot_config,
        terrain,
        scene_lib,
        device,
        **simulator_extra_params,
    )
    return {
        "terrain": terrain,
        "scene_lib": scene_lib,
        "motion_lib": motion_lib,
        "simulator": simulator,
    }


def create_protomotions_environment(
    configuration,
    motion_file,
    asset_root,
    *,
    simulation_app,
    device="cuda:0",
    num_envs=4,
    seed=42,
    headless=True,
    output_dir=None,
):
    """Construct the source-pinned G1 environment using native runtime classes.

    The caller starts Isaac Lab. The returned environment supplies the common
    vector reset/step contract and a separate reference observation provider.
    No original trainer, neural package, checkout or import alias is used.
    """
    import random
    from pathlib import Path
    import numpy as np
    from motius.configuration.protomotions.environment import (
        load_environment_configuration,
    )
    from motius.simulators.isaaclab.protomotions.physics.utils import (
        convert_friction_for_simulator,
    )
    from motius.simulators.isaaclab.protomotions.mdp.base_env.env import BaseEnv
    from motius.datasets.protomotions_expert import ProtoMotionsExpertObservations

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    configs = load_environment_configuration(configuration)
    robot = configs["robot"]
    simulator = configs["simulator"]
    environment_config = configs["env"]
    robot.asset.asset_root = str(Path(asset_root).expanduser().resolve())
    simulator.num_envs = int(num_envs)
    simulator.headless = bool(headless)
    import os

    motion_file = str(motion_file).replace("{rank}", os.environ.get("RANK", "0"))
    configs["motion_lib"].motion_file = str(Path(motion_file).expanduser().resolve())
    if output_dir is not None:
        environment_config.save_dir = str(Path(output_dir).expanduser().resolve())
    terrain, simulator = convert_friction_for_simulator(configs["terrain"], simulator)
    components = build_all_components(
        terrain,
        configs["scene_lib"],
        configs["motion_lib"],
        simulator,
        robot,
        torch.device(device),
        simulation_app=simulation_app,
    )
    environment = BaseEnv(
        config=environment_config,
        robot_config=robot,
        device=torch.device(device),
        **components,
    )
    environment.expert_observations = ProtoMotionsExpertObservations(
        environment, configs["agent"].reference_obs_components
    )
    return environment
