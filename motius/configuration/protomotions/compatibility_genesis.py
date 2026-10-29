# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Configuration classes for Genesis simulator."""
from dataclasses import dataclass, field
from motius.simulators.isaaclab.protomotions.physics.config import SimParams
from motius.simulators.isaaclab.protomotions.physics.config import SimulatorConfig


@dataclass
class GenesisSimParams(SimParams):
    """Genesis-specific simulation parameters."""

    substeps: int = field(
        default=2, metadata={"help": "Physics substeps per simulation step.", "min": 1}
    )


@dataclass
class GenesisSimulatorConfig(SimulatorConfig):
    """Configuration specific to Genesis simulator."""

    _target_: str = "protomotions.simulator.genesis.simulator.GenesisSimulator"
    w_last: bool = field(
        default=False,
        metadata={"help": "Quaternion format: False for wxyz (Genesis convention)."},
    )
    sim: GenesisSimParams = field(
        default_factory=GenesisSimParams,
        metadata={"help": "Genesis-specific simulation parameters."},
    )
