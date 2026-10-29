"""Temporal target representation and FK for PromptHMR training."""
import torch
from .video_codec import PromptHMRVideoCodec
from .video_rotation import matrix_to_rotation_6d,axis_angle_to_matrix
from .video_motion import get_local_transl_vel
from motius.motion.representation.gvhmr import matrix
from motius.motion.body_models.prompthmr.temporal import SmplxLiteV437Coco17

class PromptHMRTrainingCodec(PromptHMRVideoCodec):
    def __init__(self,device,body_model_path):
        super().__init__(device)
        self.smplx_model=SmplxLiteV437Coco17(smplx_path=body_model_path).to(device).requires_grad_(False)
        self.parents_tensor=self.smplx_model.parents[:22]
        self.parents=self.parents_tensor.tolist()

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
        rotmat = axis_angle_to_matrix(aa)  # (B, L, 22, 3, 3)

        skeleton = self.smplx_model.get_skeleton(betas)[..., :22, :]  # (B, L, 22, 3)
        local_skeleton = skeleton - skeleton[:, :, self.parents_tensor]
        local_skeleton = torch.cat([skeleton[:, :, :1], local_skeleton[:, :, 1:]], dim=2)

        if transl is not None:
            local_skeleton[..., 0, :] += transl  # B, L, 22, 3

        mat = matrix.get_TRS(rotmat, local_skeleton)  # B, L, 22, 4, 4
        fk_mat = matrix.forward_kinematics(mat, self.parents)  # B, L, 22, 4, 4
        joints = matrix.get_position(fk_mat)  # B, L, 22, 3
        if not get_intermediate:
            return joints
        else:
            return joints, mat, fk_mat


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
        B, L = inputs["smpl_params_c"]["body_pose"].shape[:2]
        # cam
        smpl_params_c = inputs["smpl_params_c"]
        body_pose = smpl_params_c["body_pose"].reshape(B, L, 21, 3)
        body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(body_pose)).flatten(-2)
        betas = smpl_params_c["betas"]
        global_orient_R = axis_angle_to_matrix(smpl_params_c["global_orient"])
        global_orient_r6d = matrix_to_rotation_6d(global_orient_R)

        # global
        R_c2gv = inputs["R_c2gv"]  # (B, L, 3, 3)
        global_orient_gv_r6d = matrix_to_rotation_6d(R_c2gv @ global_orient_R)

        # local_transl_vel
        smpl_params_w = inputs["smpl_params_w"]
        local_transl_vel = get_local_transl_vel(smpl_params_w["transl"], smpl_params_w["global_orient"])
        if False:  # debug
            transl_recover = rollout_local_transl_vel(
                local_transl_vel, smpl_params_w["global_orient"], smpl_params_w["transl"][:, [0]]
            )
            print((transl_recover - smpl_params_w["transl"]).abs().max())

        # returns
        x = torch.cat([body_pose_r6d, betas, global_orient_r6d, global_orient_gv_r6d, local_transl_vel], dim=-1)
        x_norm = (x - self.mean) / self.std
        return x_norm
