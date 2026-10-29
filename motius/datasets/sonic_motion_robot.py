# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 sonic_motion_robot components; see licenses/sonic."""
from motius.datasets.sonic_motion import MotionLibBase
from motius.motion.kinematics.sonic import Humanoid_Batch


class MotionLibRobot(MotionLibBase):

    def __init__(self, motion_lib_cfg, num_envs, device):
        super().__init__(
            motion_lib_cfg=motion_lib_cfg, num_envs=num_envs, device=device
        )
        self.mesh_parsers = Humanoid_Batch(motion_lib_cfg)
        return
