"""Occam-201 rigidity loss; this is not part of the network forward."""
import torch
from motius.motion.representation.mogendit.articulate.math.angular import r6d_to_rotation_matrix


class MoGenDiTKinematicLoss:
    def __init__(self, representation):
        self.n_joint = representation.n_joint
        self.global_pose = representation.global_pose
        self.body_model = representation.body_model

    def __call__(
        self,
        R6d,
        joint,
        length=None,
        l1_weight=0.0,
        l2_weight=1.0,
    ):
        b = R6d.shape[0]
        n_j = self.n_joint

        R = r6d_to_rotation_matrix(R6d.clone()).reshape(-1, n_j, 3, 3)
        joint = joint.clone().reshape(b, -1, n_j, 3)

        offsets_from_motion = self.body_model.get_skeleton_offsets(
            pose=R, joint=joint, global_pose=self.global_pose, require_grad=True
        ).reshape(b, -1, n_j, 3)
        init_skeleton_offsets = offsets_from_motion[:, [0]]
        loss_rigid_body = (
            torch.nn.functional.mse_loss(
                offsets_from_motion,
                init_skeleton_offsets.expand_as(offsets_from_motion),
                reduction="none",
            )
            * l2_weight
        )

        # L1 penalty
        if l1_weight > 0.0:
            loss_rigid_body += l1_weight * torch.nn.functional.l1_loss(
                offsets_from_motion,
                init_skeleton_offsets.expand_as(offsets_from_motion),
                reduction="none",
            )
        if length is not None:
            mask = torch.zeros_like(loss_rigid_body[:, :, :1, :1])
            for i in range(b):
                mask[i, : length[i]] = 1.0
            mask_rigid_body = mask.expand_as(loss_rigid_body)
            loss_rigid_body = (
                loss_rigid_body * mask_rigid_body
            ).sum() / mask_rigid_body.sum()

        else:
            loss_rigid_body = loss_rigid_body.mean()

        return loss_rigid_body
