"""Published temporal normalization and inverse representation; no body assets."""
import json
from pathlib import Path
import torch
from .video_rotation import rotation_6d_to_matrix,matrix_to_axis_angle

class PromptHMRVideoCodec:
    def __init__(self,device):
        stats=json.loads((Path(__file__).resolve().parents[3]/'configuration/prompthmr/stats.json').read_text())['motion']
        self.mean=torch.tensor(stats['mean'],device=device).float()
        self.std=torch.tensor(stats['std'],device=device).float()

    def decode(self, x_norm):
        """x_norm: (B, L, C)"""
        B, L, C = x_norm.shape
        x = x_norm * self.std + self.mean
        body_pose_r6d = x[:, :, :126]
        betas = x[:, :, 126:136]
        global_orient_r6d = x[:, :, 136:142]
        global_orient_gv_r6d = x[:, :, 142:148]
        local_transl_vel = x[:, :, 148:151]
        body_pose = matrix_to_axis_angle(rotation_6d_to_matrix(body_pose_r6d.reshape(B, L, -1, 6)))
        body_pose = body_pose.flatten(-2)
        global_orient_c = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_r6d))
        global_orient_gv = matrix_to_axis_angle(rotation_6d_to_matrix(global_orient_gv_r6d))
        output = {'body_pose': body_pose, 'betas': betas, 'global_orient': global_orient_c, 'global_orient_gv': global_orient_gv, 'local_transl_vel': local_transl_vel}
        return output
