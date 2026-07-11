import os
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import scipy.sparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange
from torch import Tensor

from .geometry import (
    angle_axis_to_rotation_matrix,
    rot6d_to_rotation_matrix,
    rotation_matrix_to_angle_axis,
    rotation_matrix_to_rot6d,
)

# fmt: off
SMPLX_JOINT2NUM = {
    "Pelvis": 0,
    "L_Hip": 1,
    "R_Hip": 2,
    "Spine1": 3,
    "L_Knee": 4,
    "R_Knee": 5,
    "Spine2": 6,
    "L_Ankle": 7,
    "R_Ankle": 8,
    "Spine3": 9,
    "L_Foot": 10,
    "R_Foot": 11,
    "Neck": 12,
    "L_Collar": 13,
    "R_Collar": 14,
    "Head": 15,
    "L_Shoulder": 16,
    "R_Shoulder": 17,
    "L_Elbow": 18,
    "R_Elbow": 19,
    "L_Wrist": 20,
    "R_Wrist": 21,
    "Jaw": 22,
    "L_Eye": 23,
    "R_Eye": 24,
    "L_Index1": 25,
    "L_Index2": 26,
    "L_Index3": 27,
    "L_Middle1": 28,
    "L_Middle2": 29,
    "L_Middle3": 30,
    "L_Pinky1": 31,
    "L_Pinky2": 32,
    "L_Pinky3": 33,
    "L_Ring1": 34,
    "L_Ring2": 35,
    "L_Ring3": 36,
    "L_Thumb1": 37,
    "L_Thumb2": 38,
    "L_Thumb3": 39,
    "R_Index1": 40,
    "R_Index2": 41,
    "R_Index3": 42,
    "R_Middle1": 43,
    "R_Middle2": 44,
    "R_Middle3": 45,
    "R_Pinky1": 46,
    "R_Pinky2": 47,
    "R_Pinky3": 48,
    "R_Ring1": 49,
    "R_Ring2": 50,
    "R_Ring3": 51,
    "R_Thumb1": 52,
    "R_Thumb2": 53,
    "R_Thumb3": 54,
}
SMPLX_NUM2JOINT = {value: key for key, value in SMPLX_JOINT2NUM.items()}
SMPLX_JOINTS = list(SMPLX_JOINT2NUM.keys())
SMPLH_JOINTS = SMPLX_JOINTS[:22] + SMPLX_JOINTS[25:]

LEFT_HAND_MEAN_AA = [ 0.1117,  0.0429, -0.4164,  0.1088, -0.0660, -0.7562, -0.0964, -0.0909,
        -0.1885, -0.1181,  0.0509, -0.5296, -0.1437,  0.0552, -0.7049, -0.0192,
        -0.0923, -0.3379, -0.4570, -0.1963, -0.6255, -0.2147, -0.0660, -0.5069,
        -0.3697, -0.0603, -0.0795, -0.1419, -0.0859, -0.6355, -0.3033, -0.0579,
        -0.6314, -0.1761, -0.1321, -0.3734,  0.8510,  0.2769, -0.0915, -0.4998,
         0.0266,  0.0529,  0.5356,  0.0460, -0.2774]
RIGHT_HAND_MEAN_AA = [ 0.1117, -0.0429,  0.4164,  0.1088,  0.0660,  0.7562, -0.0964,  0.0909,
         0.1885, -0.1181, -0.0509,  0.5296, -0.1437, -0.0552,  0.7049, -0.0192,
         0.0923,  0.3379, -0.4570,  0.1963,  0.6255, -0.2147,  0.0660,  0.5069,
        -0.3697,  0.0603,  0.0795, -0.1419,  0.0859,  0.6355, -0.3033,  0.0579,
         0.6314, -0.1761,  0.1321,  0.3734,  0.8510, -0.2769,  0.0915, -0.4998,
        -0.0266, -0.0529,  0.5356, -0.0460,  0.2774]
# fmt: on


def batch_rigid_transform_v2(rot_mats: Tensor, joints: Tensor, parents: Tensor) -> Tuple[Tensor, Tensor]:
    """
    Args:
        rot_mats: (*, J, 3, 3)
        joints: (*, J, 3)
    """
    # check shape, since sometimes beta has shape=1
    rot_mats_shape_prefix = rot_mats.shape[:-3]
    if rot_mats_shape_prefix != joints.shape[:-2]:
        joints = joints.expand(*rot_mats_shape_prefix, -1, -1)

    rel_joints = joints.clone()
    rel_joints[..., 1:, :] -= joints[..., parents[1:], :]
    transforms_mat = torch.cat([rot_mats, rel_joints[..., :, None]], dim=-1)  # (*, J, 3, 4)
    transforms_mat = F.pad(transforms_mat, [0, 0, 0, 1], value=0.0)
    transforms_mat[..., 3, 3] = 1.0  # (*, J, 4, 4)

    transform_chain = [transforms_mat[..., 0, :, :]]
    for i in range(1, parents.shape[0]):
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[..., i, :, :])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=-3)  # (*, J, 4, 4)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[..., :3, 3].clone()
    rel_transforms = transforms.clone()
    rel_transforms[..., :3, 3] -= einsum(transforms[..., :3, :3], joints, "... j c d, ... j d -> ... j c")
    return posed_joints, rel_transforms


def to_np(array: scipy.sparse.spmatrix, dtype=np.float32) -> np.ndarray:
    if "scipy.sparse" in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


def to_tensor(array: Union[Tensor, np.ndarray], dtype=torch.float32) -> Tensor:
    if torch.is_tensor(array):
        return array
    else:
        return torch.tensor(array, dtype=dtype)


class SmplLite(nn.Module):
    def __init__(
        self,
        model_path: str = "assets/body_models/smplh",
        gender: str = "neutral",
        num_betas: int = 16,
    ) -> None:
        super().__init__()

        # Load the model
        npz_name = os.path.join(model_path, f"{gender}", "model.npz")
        data_struct = dict(np.load(npz_name))
        self.faces = data_struct["f"]  # (F, 3)
        self.register_smpl_buffers(data_struct, num_betas)
        self.register_fast_skeleton_computing_buffers()

    def register_smpl_buffers(self, data_struct: dict, num_betas: int) -> None:
        # shapedirs, (V, 3, N_betas), V=10475 for SMPLX
        shapedirs = to_tensor(to_np(data_struct["shapedirs"][:, :, :num_betas])).float()
        self.register_buffer("shapedirs", shapedirs, False)

        # v_template, (V, 3)
        v_template = to_tensor(to_np(data_struct["v_template"])).float()
        self.register_buffer("v_template", v_template, False)

        # J_regressor, (J, V), J=55 for SMPLX
        J_regressor = to_tensor(to_np(data_struct["J_regressor"])).float()
        self.register_buffer("J_regressor", J_regressor, False)

        # posedirs, (54*9, V, 3), note that the first global_orient is not included
        posedirs = to_tensor(to_np(data_struct["posedirs"])).float()  # (V, 3, 54*9)
        posedirs = rearrange(posedirs, "v c n -> n v c")
        self.register_buffer("posedirs", posedirs, False)

        # lbs_weights, (V, J), J=55
        lbs_weights = to_tensor(to_np(data_struct["weights"])).float()
        self.register_buffer("lbs_weights", lbs_weights, False)

        # parents, (J), long
        parents = to_tensor(to_np(data_struct["kintree_table"][0])).long()
        parents[0] = -1
        self.register_buffer("parents", parents, False)

    def register_fast_skeleton_computing_buffers(self) -> None:
        # For fast computing of skeleton under beta
        J_template = self.J_regressor @ self.v_template  # (J, 3)
        J_shapedirs = torch.einsum("jv, vcd -> jcd", self.J_regressor, self.shapedirs)  # (J, 3, 16)
        self.register_buffer("J_template", J_template, False)
        self.register_buffer("J_shapedirs", J_shapedirs, False)

    def get_skeleton(self, betas: Tensor) -> Tensor:
        return self.J_template + einsum(betas, self.J_shapedirs, "... k, j c k -> ... j c")

    def forward(
        self,
        body_pose: Tensor,
        betas: Tensor,
        global_orient: Tensor,
        transl: Tensor,
        rotation_mode: str = "rot6d",
    ):
        """
        Args:
            body_pose: (B, L, 63)
            betas: (B, L, 10)
            global_orient: (B, L, 3)
            transl: (B, L, 3)
        Returns:
            vertices: (B, L, V, 3)
        """
        if rotation_mode == "rot6d":
            full_pose = torch.cat([global_orient, body_pose], dim=-2)
            rot_mats = rot6d_to_rotation_matrix(full_pose)
        elif rotation_mode == "aa":
            # 1. Convert [global_orient, body_pose] to rot_mats
            full_pose = torch.cat([global_orient, body_pose], dim=-1)
            rot_mats = angle_axis_to_rotation_matrix(
                full_pose.reshape(*full_pose.shape[:-1], full_pose.shape[-1] // 3, 3)
            )
        else:
            raise ValueError(f"Unsupported rotation_mode: {rotation_mode}. Supported modes are 'rot6d' and 'aa'.")

        # 2. Forward Kinematics
        J = self.get_skeleton(betas)  # (*, 55, 3)
        A = batch_rigid_transform_v2(rot_mats, J, self.parents)[1]

        # 3. Canonical v_posed = v_template + shaped_offsets + pose_offsets
        pose_feature = rot_mats[..., 1:, :, :] - rot_mats.new([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        pose_feature = pose_feature.view(*pose_feature.shape[:-3], -1)  # (*, 55*3*3)
        v_posed = (
            self.v_template
            + einsum(betas, self.shapedirs, "... k, v c k -> ... v c")
            + einsum(pose_feature, self.posedirs, "... k, k v c -> ... v c")
        )
        del pose_feature, rot_mats, full_pose

        # 4. Skinning
        T = einsum(self.lbs_weights, A, "v j, ... j c d -> ... v c d")
        verts = einsum(T[..., :3, :3], v_posed, "... v c d, ... v d -> ... v c") + T[..., :3, 3]

        # 5. Translation
        verts = verts + transl[..., None, :]
        return verts


class SmplxLiteJ24(SmplLite):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        # Compute mapping
        smpl2j24 = self.J_regressor  # (24, 6890)
        # print(f"[{self.__class__.__name__}] smpl2j24 shape: {smpl2j24.shape}")

        jids, smplx_vids = torch.where(smpl2j24 != 0)
        interestd = torch.zeros([len(smplx_vids), smpl2j24.shape[0]])
        for idx, (jid, smplx_vid) in enumerate(zip(jids, smplx_vids)):
            interestd[idx, jid] = smpl2j24[jid, smplx_vid]
        self.register_buffer("interestd", interestd, False)  # (236, 24)

        # Update to vertices of interest
        self.v_template = self.v_template[smplx_vids].clone()  # (V', 3)
        self.shapedirs = self.shapedirs[smplx_vids].clone()  # (V', 3, K)
        self.posedirs = self.posedirs[:, smplx_vids].clone()  # (K, V', 3)
        self.lbs_weights = self.lbs_weights[smplx_vids].clone()  # (V', J)

    def forward(
        self,
        body_pose: Tensor,
        betas: Tensor,
        global_orient: Tensor,
        transl: Tensor,
        left_hand_pose: Optional[Tensor] = None,
        right_hand_pose: Optional[Tensor] = None,
        rotation_mode: str = "rot6d",
    ):
        """Returns: joints (*, J, 3). (B, L) or  (B,) are both supported."""
        # Use super class's forward to get verts
        if left_hand_pose is None and right_hand_pose is None:
            if rotation_mode == "rot6d":
                eye = torch.eye(3, device=body_pose.device, dtype=body_pose.dtype)[None].repeat(15, 1, 1)
                rot6d = rotation_matrix_to_rot6d(eye)
                rot6d = rot6d[None].repeat(body_pose.shape[0], 1, 1)
                left_hand_pose = rot6d.clone()
                right_hand_pose = rot6d.clone()
            elif rotation_mode == "aa":
                left_hand_pose = torch.zeros(
                    body_pose.shape[0],
                    15 * 3,
                    device=body_pose.device,
                    dtype=body_pose.dtype,
                )
                right_hand_pose = torch.zeros(
                    body_pose.shape[0],
                    15 * 3,
                    device=body_pose.device,
                    dtype=body_pose.dtype,
                )
            else:
                raise ValueError(f"Unsupported rotation_mode: {rotation_mode}. Supported modes are 'rot6d' and 'aa'.")

        assert left_hand_pose is not None and right_hand_pose is not None
        if rotation_mode == "aa":
            if body_pose.shape[-1] == 63:
                body_pose = torch.cat([body_pose, left_hand_pose, right_hand_pose], dim=-1)
        else:
            body_pose = torch.cat([body_pose, left_hand_pose, right_hand_pose], dim=-2)
        verts = super().forward(body_pose, betas, global_orient, transl, rotation_mode=rotation_mode)  # (*, 236, 3)
        joints = einsum(self.interestd, verts, "v j, ... v c -> ... j c")
        return joints


def construct_smpl_data_dict(
    rot6d: Tensor,
    transl: Tensor,
    betas: Optional[Tensor] = None,
    gender: str = "neutral",
    use_default_hand_mean_pose: bool = False,
) -> dict:
    rotation_matrix = rot6d_to_rotation_matrix(rot6d)
    angle_axis = rotation_matrix_to_angle_axis(rotation_matrix)
    left_hand_mean_pose = (
        torch.tensor(
            LEFT_HAND_MEAN_AA,
            device=angle_axis.device,
            dtype=angle_axis.dtype,
        )
        .unsqueeze(0)
        .repeat(angle_axis.shape[0], 1)
        .reshape(angle_axis.shape[0], -1, 3)
    )
    right_hand_mean_pose = (
        torch.tensor(
            RIGHT_HAND_MEAN_AA,
            device=angle_axis.device,
            dtype=angle_axis.dtype,
        )
        .unsqueeze(0)
        .repeat(angle_axis.shape[0], 1)
        .reshape(angle_axis.shape[0], -1, 3)
    )
    if angle_axis.shape[1] == 22:
        angle_axis = torch.cat(
            [
                angle_axis,
                left_hand_mean_pose,
                right_hand_mean_pose,
            ],
            dim=1,
        )
    elif angle_axis.shape[1] == 52:
        if use_default_hand_mean_pose:
            angle_axis = torch.cat(
                [
                    angle_axis[:, :22],
                    left_hand_mean_pose,
                    right_hand_mean_pose,
                ],
                dim=1,
            )
        else:
            angle_axis = angle_axis

    assert angle_axis.shape[1] == 52, f"angle_axis should be 52, but got {angle_axis.shape[1]}"
    dump = {
        "betas": betas.cpu().numpy() if betas is not None else np.zeros((1, 16)),
        "gender": gender,
        "poses": angle_axis.cpu().numpy().reshape(angle_axis.shape[0], -1),
        "trans": transl.cpu().numpy(),
        "mocap_framerate": 30,
        "num_frames": angle_axis.shape[0],
        "Rh": angle_axis.cpu().numpy().reshape(angle_axis.shape[0], -1)[:, :3],
    }
    return dump


if __name__ == "__main__":
    # python -m hymotion.pipeline.smpl_lite
    model = SmplxLiteJ24()
    batch_size = 128
    body_pose = torch.randn(batch_size, 51, 6)
    betas = torch.randn(batch_size, 16)
    global_orient = torch.randn(batch_size, 3)
    transl = torch.randn(batch_size, 3)
    joints = model(body_pose, betas, global_orient, transl)
    print(joints.shape)
