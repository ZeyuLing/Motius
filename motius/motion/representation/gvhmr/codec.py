import torch
import torch.nn as nn
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d
import motius.motion.representation.gvhmr.matrix as matrix
from motius.motion.representation.gvhmr.global_motion import get_local_transl_vel
from motius.motion.body_models.gvhmr.smplx import SmplxLiteV437Coco17
from pathlib import Path
import json

class EnDecoder(nn.Module):

    def __init__(self, body_model_path):
        super().__init__()
        stats = json.loads((Path(__file__).resolve().parents[3] / 'configuration/gvhmr/stats.json').read_text())['motion']
        self.register_buffer('mean', torch.tensor(stats['mean']).float(), False)
        self.register_buffer('std', torch.tensor(stats['std']).float(), False)
        self.smplx_model = SmplxLiteV437Coco17(model_path=body_model_path)
        parents = self.smplx_model.parents[:22]
        self.register_buffer('parents_tensor', parents, False)
        self.parents = parents.tolist()

    def normalize_body_pose_r6d(self, body_pose_r6d):
        """body_pose_r6d: (B, L, {J*6}/{J, 6}) ->  (B, L, J*6)"""
        B, L = body_pose_r6d.shape[:2]
        body_pose_r6d = body_pose_r6d.reshape(B, L, -1)
        if self.mean.shape[-1] == 1:
            return body_pose_r6d
        body_pose_r6d = (body_pose_r6d - self.mean[:126]) / self.std[:126]
        return body_pose_r6d

    def fk_v2(self, body_pose, betas, global_orient=None, transl=None, get_intermediate=False):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
        Returns:
            joints: (B, L, 22, 3)
        """
        B, L = body_pose.shape[:2]
        if global_orient is None:
            global_orient = torch.zeros((B, L, 3), device=body_pose.device)
        aa = torch.cat([global_orient, body_pose], dim=-1).reshape(B, L, -1, 3)
        rotmat = axis_angle_to_matrix(aa)
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        if transl is not None:
            local_skeleton[..., 0, :] += transl
        mat = matrix.get_TRS(rotmat, local_skeleton)
        fk_mat = matrix.forward_kinematics(mat, self.parents)
        joints = matrix.get_position(fk_mat)
        if not get_intermediate:
            return joints
        else:
            return (joints, mat, fk_mat)

    def get_local_pos(self, betas):
        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)
        return local_skeleton

    def encode(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        B, L = inputs['smpl_params_c']['body_pose'].shape[:2]
        smpl_params_c = inputs['smpl_params_c']
        body_pose = smpl_params_c['body_pose'].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        betas = smpl_params_c['betas']
        global_orient_R = axis_angle_to_matrix(smpl_params_c['global_orient'])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)
        R_c2gv = inputs['R_c2gv']
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)
        smpl_params_w = inputs['smpl_params_w']
        local_transl_vel = get_local_transl_vel(smpl_params_w['transl'], smpl_params_w['global_orient'])
        x = torch.cat([body_pose_r6d, betas, global_orient_r6d, global_orient_gv_r6d, local_transl_vel], dim=-1)
        x_norm = (x - self.mean) / self.std
        return x_norm

    def encode_translw(self, inputs):
        """
        definition: {
                body_pose_r6d,  # (B, L, (J-1)*6) -> 0:126
                betas, # (B, L, 10) -> 126:136
                global_orient_r6d,  # (B, L, 6) -> 136:142  incam
                global_orient_gv_r6d: # (B, L, 6) -> 142:148  gv
                local_transl_vel,  # (B, L, 3) -> 148:151, smpl-coord
            }
        """
        smpl_params_w = inputs['smpl_params_w']
        local_transl_vel = get_local_transl_vel(smpl_params_w['transl'], smpl_params_w['global_orient'])
        x = local_transl_vel
        x_norm = (x - self.mean[-3:]) / self.std[-3:]
        return x_norm

    def decode_translw(self, x_norm):
        return x_norm * self.std[-3:] + self.mean[-3:]

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
