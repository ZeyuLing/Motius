"""Licensed body geometry applied after PromptHMR's neural heads."""
from pathlib import Path
import pickle
import torch
from torch import nn
from torch.amp import autocast
from .smpl_wrapper import SMPL,SMPLX
from motius.motion.representation.prompthmr.image_rotation import rotation_6d_to_matrix

class PromptHMRBodyGeometry(nn.Module):
    def __init__(self,body_models_root,*,mapping_path,joint_mapping_path,transl_type='root'):
        super().__init__()
        root=Path(body_models_root)
        self.smplx=SMPLX(str(root/'smplx'),joint_mapping_path=joint_mapping_path)
        self.smpl=SMPL(str(root/'smpl'))
        with open(mapping_path,'rb') as stream:mapping=pickle.load(stream)['matrix']
        self.register_buffer('smplx2smpl',torch.from_numpy(mapping).float())
        self.transl_type=transl_type

    def process_output(self, output, use_mean_hands):
        K = output['cam_int']
        transl = output['transl'].reshape(-1, 3)
        shape = output['betas'].reshape(-1, 10)
        pose = output['pose'].reshape(-1, 22, 6)
        rotmat = rotation_6d_to_matrix(pose)
        B = len(rotmat)
        with autocast('cuda', enabled=False):
            if self.transl_type == 'root':
                root = self.smplx(betas=shape).joints[:, 0]
                transl = transl - root.detach()
                output['transl'] = transl
            if use_mean_hands:
                smplx_out = self.smplx(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:], betas=shape, transl=transl, left_hand_pose=self.smplx.hands_mean[:, :15].repeat(B, 1, 1, 1), right_hand_pose=self.smplx.hands_mean[:, 15:].repeat(B, 1, 1, 1))
            else:
                smplx_out = self.smplx(global_orient=rotmat[:, :1], body_pose=rotmat[:, 1:], betas=shape, transl=transl)
            vertices = smplx_out.vertices
            body_joints = smplx_out.body_joints
            output['rotmat'] = rotmat.reshape(B, -1, 3, 3)
            output['vertices'] = vertices.reshape(B, -1, 3)
            output['body_joints'] = body_joints.reshape(B, -1, 3)
            pred_joints = output['body_joints']
            pred_joints = pred_joints / (pred_joints[..., [-1]] + 1e-05)
            pred_joints = torch.einsum('bij, bvj->bvi', K.reshape(-1, 3, 3), pred_joints)[..., :2]
            output['body_joints2d'] = pred_joints.clamp(-2000.0, 2000.0)
            smpl_verts = self.smplx2smpl @ vertices
            smpl_joints = self.smpl.joints_from_vertices(smpl_verts)
            smpl_joints2d = smpl_joints
            smpl_joints2d = smpl_joints2d / (smpl_joints2d[..., [-1]] + 1e-05)
            smpl_joints2d = torch.einsum('bij, bvj->bvi', K.reshape(-1, 3, 3), smpl_joints2d)[..., :2]
            output['smpl_vertices'] = smpl_verts
            output['smpl_joints'] = smpl_joints
            output['smpl_joints2d'] = smpl_joints2d.clamp(-2000.0, 2000.0)
            output['smpl_j3d'] = self.smpl.J_regressor @ smpl_verts
        return output
