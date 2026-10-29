ASSET_ROOT = __import__('pathlib').Path(__file__).resolve().parents[3] / 'configuration/gvhmr/body'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from pytorch3d.transforms import axis_angle_to_matrix, rotation_6d_to_matrix
from smplx.utils import Struct, to_np, to_tensor
from einops import einsum, rearrange

class SmplxLite(nn.Module):

    def __init__(self, model_path, gender='neutral', num_betas=10):
        super().__init__()
        model_path = Path(model_path)
        if model_path.is_dir():
            smplx_path = Path(model_path) / f'SMPLX_{gender.upper()}.npz'
        else:
            smplx_path = model_path
        assert smplx_path.exists()
        model_data = np.load(smplx_path, allow_pickle=True)
        data_struct = Struct(**model_data)
        self.faces = data_struct.f
        self.register_smpl_buffers(data_struct, num_betas)
        self.register_fast_skeleton_computing_buffers()
        other_default_pose = torch.cat([torch.zeros(9), to_tensor(data_struct.hands_meanl).float(), to_tensor(data_struct.hands_meanr).float()])
        self.register_buffer('other_default_pose', other_default_pose, False)

    def register_smpl_buffers(self, data_struct, num_betas):
        shapedirs = to_tensor(to_np(data_struct.shapedirs[:, :, :num_betas])).float()
        self.register_buffer('shapedirs', shapedirs, False)
        v_template = to_tensor(to_np(data_struct.v_template)).float()
        self.register_buffer('v_template', v_template, False)
        J_regressor = to_tensor(to_np(data_struct.J_regressor)).float()
        self.register_buffer('J_regressor', J_regressor, False)
        posedirs = to_tensor(to_np(data_struct.posedirs)).float()
        posedirs = rearrange(posedirs, 'v c n -> n v c')
        self.register_buffer('posedirs', posedirs, False)
        lbs_weights = to_tensor(to_np(data_struct.weights)).float()
        self.register_buffer('lbs_weights', lbs_weights, False)
        parents = to_tensor(to_np(data_struct.kintree_table[0])).long()
        parents[0] = -1
        self.register_buffer('parents', parents, False)

    def register_smplh_buffers(self, data_struct, num_pca_comps, flat_hand_mean):
        left_hand_components = to_tensor(data_struct.hands_componentsl[:num_pca_comps]).float()
        right_hand_components = to_tensor(data_struct.hands_componentsr[:num_pca_comps]).float()
        self.register_buffer('left_hand_components', left_hand_components, False)
        self.register_buffer('right_hand_components', right_hand_components, False)
        left_hand_mean = to_tensor(data_struct.hands_meanl).float()
        right_hand_mean = to_tensor(data_struct.hands_meanr).float()
        if not flat_hand_mean:
            left_hand_mean = torch.zeros_like(left_hand_mean)
            right_hand_mean = torch.zeros_like(right_hand_mean)
        self.register_buffer('left_hand_mean', left_hand_mean, False)
        self.register_buffer('right_hand_mean', right_hand_mean, False)

    def register_smplx_buffers(self, data_struct):
        expr_dirs = to_tensor(to_np(data_struct.shapedirs[:, :, 300:310])).float()
        self.register_buffer('expr_dirs', expr_dirs, False)

    def register_fast_skeleton_computing_buffers(self):
        J_template = self.J_regressor @ self.v_template
        J_shapedirs = torch.einsum('jv, vcd -> jcd', self.J_regressor, self.shapedirs)
        self.register_buffer('J_template', J_template, False)
        self.register_buffer('J_shapedirs', J_shapedirs, False)

    def get_skeleton(self, betas):
        return self.J_template + einsum(betas, self.J_shapedirs, '... k, j c k -> ... j c')

    def forward(self, body_pose, betas, global_orient, transl=None, rotation_type='aa'):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
        Returns:
            vertices: (B, L, V, 3)
        """
        other_default_pose = self.other_default_pose
        if rotation_type == 'aa':
            other_default_pose = other_default_pose.expand(*body_pose.shape[:-1], -1)
            full_pose = torch.cat([global_orient, body_pose, other_default_pose], dim=-1)
            rot_mats = axis_angle_to_matrix(full_pose.reshape(*full_pose.shape[:-1], 55, 3))
            del full_pose, other_default_pose
        else:
            assert rotation_type == 'r6d'
            other_default_pose = axis_angle_to_matrix(other_default_pose.view(33, 3))
            part_full_pose = torch.cat([global_orient, body_pose], dim=-1)
            rot_mats = rotation_6d_to_matrix(part_full_pose.view(*part_full_pose.shape[:-1], 22, 6))
            other_default_pose = other_default_pose.expand(*rot_mats.shape[:-3], -1, -1, -1)
            rot_mats = torch.cat([rot_mats, other_default_pose], dim=-3)
            del part_full_pose, other_default_pose
        J = self.get_skeleton(betas)
        A = batch_rigid_transform_v2(rot_mats, J, self.parents)[1]
        pose_feature = rot_mats[..., 1:, :, :] - rot_mats.new([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pose_feature = pose_feature.view(*pose_feature.shape[:-3], -1)
        v_posed = self.v_template + einsum(betas, self.shapedirs, '... k, v c k -> ... v c') + einsum(pose_feature, self.posedirs, '... k, k v c -> ... v c')
        del pose_feature, rot_mats
        T = einsum(self.lbs_weights, A, 'v j, ... j c d -> ... v c d')
        verts = einsum(T[..., :3, :3], v_posed, '... v c d, ... v d -> ... v c') + T[..., :3, 3]
        if transl is not None:
            verts = verts + transl[..., None, :]
        return verts

class SmplxLiteV437Coco17(SmplxLite):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        smplx2smpl = torch.load(ASSET_ROOT / 'smplx2smpl_sparse.pt')
        COCO17_regressor = torch.load(ASSET_ROOT / 'smpl_coco17_J_regressor.pt')
        smplx2coco17 = torch.matmul(COCO17_regressor, smplx2smpl.to_dense())
        jids, smplx_vids = torch.where(smplx2coco17 != 0)
        smplx2coco17_interestd = torch.zeros([len(smplx_vids), 17])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            smplx2coco17_interestd[idx, jid] = smplx2coco17[jid, smplx_vid]
        self.register_buffer('smplx2coco17_interestd', smplx2coco17_interestd, False)
        assert len(smplx_vids) == 132
        smplx_vids2 = torch.load(ASSET_ROOT / 'smplx_verts437.pt')
        smplx_vids = torch.cat([smplx_vids, smplx_vids2])
        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, body_pose, betas, global_orient, transl):
        """
        Returns:
            verts_437: (*, 437, 3)
            joints (*, 17, 3). (B, L) or  (B,) are both supported.
        """
        verts = super().forward(body_pose, betas, global_orient, transl)
        verts_437 = verts[..., 132:, :].clone()
        joints = einsum(self.smplx2coco17_interestd, verts[..., :132, :], 'v j, ... v c -> ... j c')
        return (verts_437, joints)

class SmplxLiteSmplN24(SmplxLite):
    """Output SMPL(not smplx)-Neutral 24 joints (Faster, but cannot output vertices)"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        smplx2smpl = torch.load(ASSET_ROOT / 'smplx2smpl_sparse.pt')
        smpl2joints = torch.load(ASSET_ROOT / 'smpl_neutral_J_regressor.pt')
        smplx2joints = torch.matmul(smpl2joints, smplx2smpl.to_dense())
        jids, smplx_vids = torch.where(smplx2joints != 0)
        smplx2joints_interested = torch.zeros([len(smplx_vids), smplx2joints.size(0)])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            smplx2joints_interested[idx, jid] = smplx2joints[jid, smplx_vid]
        self.register_buffer('smplx2joints_interested', smplx2joints_interested, False)
        self.v_template = self.v_template[smplx_vids].clone()
        self.shapedirs = self.shapedirs[smplx_vids].clone()
        self.posedirs = self.posedirs[:, smplx_vids].clone()
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()

    def forward(self, body_pose, betas, global_orient, transl):
        """Returns: joints (*, J, 3). (B, L) or  (B,) are both supported."""
        verts = super().forward(body_pose, betas, global_orient, transl)
        joints = einsum(self.smplx2joints_interested, verts, 'v j, ... v c -> ... j c')
        return joints

def batch_rigid_transform_v2(rot_mats, joints, parents):
    """
    Args:
        rot_mats: (*, J, 3, 3)
        joints: (*, J, 3)
    """
    rot_mats_shape_prefix = rot_mats.shape[:-3]
    if rot_mats_shape_prefix != joints.shape[:-2]:
        joints = joints.expand(*rot_mats_shape_prefix, -1, -1)
    rel_joints = joints.clone()
    rel_joints[..., 1:, :] -= joints[..., parents[1:], :]
    transforms_mat = torch.cat([rot_mats, rel_joints[..., :, None]], dim=-1)
    transforms_mat = F.pad(transforms_mat, [0, 0, 0, 1], value=0.0)
    transforms_mat[..., 3, 3] = 1.0
    transform_chain = [transforms_mat[..., 0, :, :]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[..., i, :, :])
        transform_chain.append(curr_res)
    transforms = torch.stack(transform_chain, dim=-3)
    posed_joints = transforms[..., :3, 3].clone()
    rel_transforms = transforms.clone()
    rel_transforms[..., :3, 3] -= einsum(transforms[..., :3, :3], joints, '... j c d, ... j d -> ... j c')
    return (posed_joints, rel_transforms)
