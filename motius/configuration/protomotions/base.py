# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Base robot configuration classes.

This module defines the configuration dataclasses for robot morphology, control
parameters, and simulator-specific settings. All robot configurations (SMPL, G1, etc.)
inherit from RobotConfig and specify their robot-specific parameters.

Key Classes:
    - RobotConfig: Main robot configuration
    - RobotAssetConfig: Robot asset and physics properties
    - RobotControlConfig: PD control parameters
    - ControlType: Enum for control modes

Key Features:
    - Multi-simulator support
    - Configurable PD gains and action scaling
    - Asset file management
    - Body name mappings for observations
    - Kinematic structure extraction
"""
from dataclasses import dataclass, field
from pathlib import Path
from motius.motion.kinematics.protomotions import ControlInfo
from motius.motion.kinematics.protomotions import KinematicInfo
from motius.simulators.isaaclab.protomotions.physics.config import RobotNoiseConfig
from motius.configuration.protomotions.compatibility_isaacgym import IsaacGymSimParams
from motius.simulators.isaaclab.protomotions.backend.config import IsaacLabSimParams
from motius.configuration.protomotions.compatibility_genesis import GenesisSimParams
from motius.configuration.protomotions.compatibility_newton import NewtonSimParams
from motius.configuration.protomotions.compatibility_mujoco import MujocoSimParams
from typing import List, Optional, Dict, Union
from enum import Enum
import os
import re
import torch


@dataclass
class SimulatorParams:
    isaacgym: IsaacGymSimParams = field(default_factory=IsaacGymSimParams)
    isaaclab: IsaacLabSimParams = field(default_factory=IsaacLabSimParams)
    genesis: GenesisSimParams = field(default_factory=GenesisSimParams)
    newton: NewtonSimParams = field(default_factory=NewtonSimParams)
    mujoco: MujocoSimParams = field(default_factory=MujocoSimParams)


@dataclass
class InitState:
    """Configuration for robot initial state."""

    pos: Optional[List[float]]


class ControlType(Enum):
    """Enum defining the available control types for the robot.

    BUILT_IN_PD: Built-in PD controller (e.g. Isaac Gym's internal PD controller)
    VELOCITY: Velocity control using custom PD controller
    TORQUE: Direct torque control
    PROPORTIONAL: Proportional control using custom PD controller
    """

    BUILT_IN_PD = "built_in_pd"
    TORQUE = "torque"
    PROPORTIONAL = "proportional"

    @classmethod
    def from_str(cls, value: str) -> "ControlType":
        """Create enum from string, case-insensitive."""
        try:
            return next(
                (member for member in cls if member.value.lower() == value.lower())
            )
        except StopIteration:
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. Valid values are: {[e.value for e in cls]}"
            )


@dataclass
class RobotAssetConfig:
    """Configuration for robot asset properties."""

    asset_root: str = field(
        default_factory=lambda: str(
            Path(__file__).resolve().parents[3]
            / "checkpoints/robots/protomotions"
        )
    )
    self_collisions: bool = True
    asset_file_name: str = None
    usd_asset_file_name: str = None
    usd_bodies_root_prim_path: str = None
    replace_cylinder_with_capsule: Optional[bool] = None
    thickness: Optional[float] = None
    max_angular_velocity: Optional[float] = None
    max_linear_velocity: Optional[float] = None
    density: Optional[float] = None
    angular_damping: Optional[float] = None
    linear_damping: Optional[float] = None
    disable_gravity: Optional[bool] = None
    fix_base_link: Optional[bool] = None

    def __post_init__(self):
        """Validate that asset_file_name is set."""
        if not self.asset_file_name or not self.asset_file_name.endswith(".xml"):
            raise ValueError(
                f"RobotAssetConfig.asset_file_name ('{self.asset_file_name}') must be a valid path to an .xml MJCF file to extract kinematic info. if you are using URDF, convert it to MJCF first"
            )


@dataclass
class ControlConfig:
    """Configuration for robot control parameters."""

    override_control_info: Optional[Dict[str, ControlInfo]] = None
    control_type: ControlType = ControlType.BUILT_IN_PD
    soft_pos_limit: float = 0.9
    control_info: Dict[str, ControlInfo] = field(init=False)

    def __post_init__(self):
        """Validate that override_control_info is a dictionary."""
        if self.override_control_info is not None:
            override_control_info = {}
            for key, value in self.override_control_info.items():
                if isinstance(value, dict):
                    override_control_info[key] = ControlInfo.from_dict(value)
                else:
                    override_control_info[key] = value
            self.override_control_info = override_control_info

    def initialize_control_info(self, asset: RobotAssetConfig):
        """Initialize control info from asset configuration."""
        if not hasattr(self, "control_info"):
            from motius.motion.kinematics.protomotions import extract_control_info

            self.control_info = extract_control_info(
                mjcf_path=os.path.join(asset.asset_root, asset.asset_file_name),
                override_control_info=self.override_control_info,
            )


@dataclass
class RobotConfig:
    """Configuration for robot morphology and control parameters.

    Defines all robot-specific parameters including asset files, body names,
    control settings, and physical properties. Each robot (SMPL, G1, etc.)
    should subclass this and provide robot-specific values.

    Key configuration areas:

    - **Asset**: Robot mesh/URDF files and physical properties
    - **Body Names**: Named body parts for tracking and observations
    - **Control**: PD gains, action scales, and control modes
    - **Kinematic Info**: Joint structure extracted from MJCF/URDF
    - **Simulator Params**: Simulator-specific overrides

    Example::

        config = RobotConfig(
            asset=RobotAssetConfig(asset_file_name="robot.xml"),
            control=ControlConfig(control_type=ControlType.BUILT_IN_PD)
        )
    """

    asset: RobotAssetConfig
    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {}
    )
    default_root_height: float = 1
    default_dof_pos: Optional[Union[List[float], torch.Tensor, Dict[str, float]]] = None
    contact_bodies: Optional[Union[List[str], str]] = None
    trackable_bodies_subset: Union[List[str], str] = "all"
    non_termination_contact_bodies: Union[List[str], str] = "all"
    init_state: Optional[InitState] = None
    reset_noise: Optional[RobotNoiseConfig] = None
    mimic_small_marker_bodies: Optional[List[str]] = None
    anchor_body_name: Optional[str] = None
    contact_pairs_multiplier: int = 16
    control: ControlConfig = field(default_factory=ControlConfig)
    kinematic_info: KinematicInfo = field(init=False)
    number_of_actions: int = field(init=False)
    anchor_body_index: int = field(init=False)
    simulation_params: SimulatorParams = field(default_factory=SimulatorParams)

    def __post_init__(self):
        """Compute derived fields after initialization."""
        from motius.motion.kinematics.protomotions import extract_kinematic_info

        self.kinematic_info = extract_kinematic_info(
            os.path.join(self.asset.asset_root, self.asset.asset_file_name)
        )
        if self.anchor_body_name is None:
            self.anchor_body_index = 0
        else:
            if self.anchor_body_name not in self.kinematic_info.body_names:
                raise ValueError(
                    f"anchor_body_name '{self.anchor_body_name}' not found in body_names: {self.kinematic_info.body_names}"
                )
            self.anchor_body_index = self.kinematic_info.body_names.index(
                self.anchor_body_name
            )
        self.control.initialize_control_info(self.asset)
        self.number_of_actions = self.kinematic_info.num_dofs
        if self.default_dof_pos is None:
            self.default_dof_pos = torch.zeros(
                self.kinematic_info.num_dofs, dtype=torch.float32
            )
        elif isinstance(self.default_dof_pos, dict):
            dof_pos = torch.zeros(self.kinematic_info.num_dofs, dtype=torch.float32)
            for pattern, value in self.default_dof_pos.items():
                for i, name in enumerate(self.kinematic_info.dof_names):
                    if re.fullmatch(pattern, name):
                        dof_pos[i] = value
            self.default_dof_pos = dof_pos
        else:
            if not isinstance(self.default_dof_pos, torch.Tensor):
                self.default_dof_pos = torch.tensor(
                    self.default_dof_pos, dtype=torch.float32
                )
            assert (
                len(self.default_dof_pos) == self.kinematic_info.num_dofs
            ), f"default_dof_pos length {len(self.default_dof_pos)} != num_dofs {self.kinematic_info.num_dofs}"
        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self.contact_bodies = abstract_names_to_body_names(self.contact_bodies, self)
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )
        required_abstract_names = [
            "all_left_foot_bodies",
            "all_right_foot_bodies",
            "all_left_hand_bodies",
            "all_right_hand_bodies",
            "head_body_name",
            "torso_body_name",
        ]
        for name in required_abstract_names:
            assert (
                name in self.common_naming_to_robot_body_names.keys()
            ), f"RobotConfig.common_naming_to_robot_body_names must contain {name}"

    def update_fields(self, **kwargs):
        """Update robot config fields and reprocess derived fields.

        This is useful when you need to modify fields after initialization
        without manually calling __post_init__.

        Args:
            **kwargs: Fields to update on the robot config
        """
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise ValueError(f"RobotConfig has no field '{key}'")
            setattr(self, key, value)
        self.mimic_small_marker_bodies = abstract_names_to_body_names(
            self.mimic_small_marker_bodies, self
        )
        self.contact_bodies = abstract_names_to_body_names(self.contact_bodies, self)
        self.trackable_bodies_subset = abstract_names_to_body_names(
            self.trackable_bodies_subset, self
        )


def abstract_names_to_body_names(
    names: Union[List[str], str], robot_config: RobotConfig
):
    if names is None:
        return None
    if names == "all":
        return robot_config.kinematic_info.body_names
    if isinstance(names, list):
        parsed_names = []
        for name in names:
            parsed_names.extend(abstract_names_to_body_names(name, robot_config))
        return parsed_names
    if names == "root":
        return [robot_config.kinematic_info.body_names[0]]
    elif names in robot_config.common_naming_to_robot_body_names.keys():
        return robot_config.common_naming_to_robot_body_names[names]
    else:
        return [names]
