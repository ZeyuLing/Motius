# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
"""PhysFlow G1 tracker with global displacement target observations.

This experiment intentionally stays close to the released G1 BeyondMimic deploy
configuration, but exposes reference-vs-current anchor XY drift and reference
anchor velocity to the actor.
The released deploy policy uses ``include_xy_offset=False`` and therefore cannot
be expected to follow the global displacement of a prompt such as "walk forward".
"""

import os
from pathlib import Path

_artifact_root = os.environ.get(
    "MOTIUS_GENTRACK_PROTOMOTIONS_ARTIFACT"
) or os.environ.get("PHYSFLOW_PROTOMOTIONS_ARTIFACT")
_BASE_PATH = (
    Path(_artifact_root).expanduser() / "experiment_config.py"
    if _artifact_root
    else (
        Path(__file__).resolve().parents[3]
        / "data"
        / "pretrained_models"
        / "motion_tracker"
        / "g1-bones-deploy"
        / "experiment_config.py"
    )
)
if not _BASE_PATH.is_file():
    raise FileNotFoundError(
        "GenTrack requires the G1 tracker experiment_config.py; set "
        "MOTIUS_GENTRACK_PROTOMOTIONS_ARTIFACT to the directory containing it "
        f"(resolved path: {_BASE_PATH})"
    )

_ns = {}
exec(compile(_BASE_PATH.read_text(), str(_BASE_PATH), "exec"), _ns)

terrain_config = _ns["terrain_config"]
scene_lib_config = _ns["scene_lib_config"]
motion_lib_config = _ns["motion_lib_config"]
agent_config = _ns["agent_config"]
configure_robot_and_simulator = _ns["configure_robot_and_simulator"]


def _fit_debug_batch_sizes(cfg, args):
    if hasattr(cfg, "amp_parameters"):
        cfg.amp_parameters.discriminator_batch_size = min(
            cfg.amp_parameters.discriminator_batch_size,
            args.batch_size,
        )


def env_config(robot_cfg, args):
    cfg = _ns["env_config"](robot_cfg, args)
    target_obs = cfg.observation_components["noisy_mimic_reduced_coords_target_poses"]
    clean_target_obs = cfg.observation_components["clean_mimic_reduced_coords_target_poses"]
    target_obs.static_params["include_xy_offset"] = True
    target_obs.static_params["include_anchor_vel"] = True
    clean_target_obs.static_params["include_xy_offset"] = True
    clean_target_obs.static_params["include_anchor_vel"] = True
    return cfg


def agent_config(env_cfg, robot_cfg, args):
    cfg = _ns["agent_config"](env_cfg, robot_cfg, args)
    _fit_debug_batch_sizes(cfg, args)
    return cfg


def apply_inference_overrides(
    robot_cfg,
    simulator_cfg,
    env_cfg,
    agent_cfg,
    terrain_cfg,
    motion_lib_cfg,
    scene_lib_cfg,
    args,
):
    _ns["apply_inference_overrides"](
        robot_cfg,
        simulator_cfg,
        env_cfg,
        agent_cfg,
        terrain_cfg,
        motion_lib_cfg,
        scene_lib_cfg,
        args,
    )
    target_obs = env_cfg.observation_components["noisy_mimic_reduced_coords_target_poses"]
    target_obs.static_params["include_xy_offset"] = True
    target_obs.static_params["include_anchor_vel"] = True
    target_obs.static_params["zero_xy_offset"] = False
