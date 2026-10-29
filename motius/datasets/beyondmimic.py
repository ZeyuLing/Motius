"""Published joint/body reference-motion arrays for BeyondMimic tracking."""
import os
import numpy as np
import torch
from collections.abc import Sequence

class BeyondMimicMotion:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu", *, body_names=None, joint_names=None):
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        if body_names is not None and joint_names is not None:
            from .beyondmimic_motion_contract import load_motion_npz
            data,body_indexes=load_motion_npz(motion_file,robot_body_indexes=body_indexes,
                target_body_names=body_names,target_joint_names=joint_names)
        else:
            with np.load(motion_file,allow_pickle=False) as source:
                data={name:source[name] for name in source.files}
        self.fps = data["fps"]
        self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
        self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
        self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
        self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
        self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
        self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
        self._body_indexes = body_indexes
        self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    def __len__(self):return self.time_step_total
    def __getitem__(self,index):
        return {name:getattr(self,name)[index] for name in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
