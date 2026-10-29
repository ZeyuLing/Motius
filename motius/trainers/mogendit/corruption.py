"""Training-only corruption of Occam-201 motions."""
import torch
from motius.motion.representation.mogendit.articulate.math.angular import (
    r6d_to_rotation_matrix, rotation_matrix_to_r6d)
from .degradation import GlobalMotionDegradation


class MoGenDiTCorruption:
    def __init__(self, representation):
        self.n_joint = representation.n_joint
        self.pose_mask = representation.pose_mask
        self.joint_mask = representation.joint_mask
        self.trans_mask = representation.trans_mask
        self.global_pose = representation.global_pose
        self.degradation = GlobalMotionDegradation()

    @torch.no_grad()
    def motion_degradation_batch(
        self,
        motion: torch.Tensor,
        keyframe_mask=None,
        length=None,
        bool_length_mask=None,
    ):
        b = motion.shape[0]
        seq_len = motion.shape[1]
        motion = motion.clone()
        motion_origin = motion.clone()

        # 提提取并处理pose
        pose = r6d_to_rotation_matrix(
            motion[..., self.pose_mask.to(motion.device)]
        ).reshape(-1, self.n_joint, 3, 3)

        pose = pose.reshape(b, -1, self.n_joint, 3, 3)

        # 提提取并处理joint vel trans
        joint = motion[..., self.joint_mask.to(motion.device)].reshape(
            b, -1, self.n_joint, 3
        )

        trans = motion[..., self.trans_mask.to(motion.device)].reshape(b, -1, 3)

        pose, joint, trans = self.degradation.apply_random_degradations(
            pose, joint, trans, global_pose=self.global_pose
        )

        pose = rotation_matrix_to_r6d(pose).reshape(b, seq_len, -1)

        motion[..., self.pose_mask] = pose.flatten(2)
        motion[..., self.joint_mask] = joint.flatten(2)
        motion[..., self.trans_mask] = trans.flatten(2)

        if keyframe_mask is not None:
            keyframe_mask_bool = keyframe_mask == 1
            motion[keyframe_mask_bool] = motion_origin[keyframe_mask_bool]

        if bool_length_mask is not None:
            motion[~bool_length_mask] *= 0.0

        return motion
