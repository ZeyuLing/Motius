"""FlowHMR vertex supervision, kept outside the model package."""
import torch
from torch import Tensor
from motius.motion.body_models.flowhmr import SMPLMesh
from motius.motion.body_models.flowhmr.fk_utils import get_joints_from_smpl_params

class VertexLoss(torch.nn.Module):
    def __init__(
        self,
        body_model: SMPLMesh,
        start_step=10000,
        overlap_step=10000,
        name="SmoothL1Loss",
        weight=1.0,
        num_sample_points=-1,
    ):
        super().__init__()
        self.body_model = body_model
        self.start_step = start_step
        self.overlap_step = overlap_step
        self.num_sample_points = num_sample_points
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")

    def calculate_smpl_mesh_vertices(self, batch, sample_indices=None):
        rot6d = batch.get("vertex_rot6d", batch["rot6d"])
        transl = batch["trans"]
        shapes = batch["shapes"]
        if shapes.dim() == 3 and shapes.shape[1] != rot6d.shape[1]:
            shapes = shapes.expand(-1, rot6d.shape[1], -1)
        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        transl_flat = torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)

        params = {
            "rot6d": rot6d_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out_vertices = self.body_model(params, sample_indices=sample_indices)
        out_vertices = out_vertices["vertices_wotrans"]
        out_vertices = out_vertices.reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)
        return out_vertices

    def forward(self, pred, gt, global_step):
        if global_step <= self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        if self.num_sample_points > 0:
            sample_indices = torch.randint(0, self.body_model.v_template.shape[0], (self.num_sample_points,))
        else:
            sample_indices = None
        pred_vertices = self.calculate_smpl_mesh_vertices(pred, sample_indices)
        gt_vertices = self.calculate_smpl_mesh_vertices(gt, sample_indices)

        loss = self.loss_func(pred_vertices, gt_vertices, reduction="none").sum(dim=-1).mean(dim=-1)
        loss = loss * weight
        return loss
