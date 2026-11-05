"""Native FlowHMR objective executed by the shared Motius runner."""
from typing import Any, Dict, Optional
import torch
from torch import Tensor
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.representation.flowhmr import FlowHMRMotionCodec
from motius.motion.representation.flowhmr.motion import _run_fk
from motius.motion.representation.flowhmr.geometry import rotation_matrix_to_rot6d
from motius.diffusion.flowhmr_utils import length_to_mask
from .losses import VertexLoss


def to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: to_device(v, device) for k, v in value.items()}
    return value


@TRAINERS.register_module()
class FlowHMRTrainer(BaseTrainer):
    def __init__(self, bundle, body_model_path=None, **kwargs):
        super().__init__(bundle, **kwargs)
        self.codec = FlowHMRMotionCodec.from_bundle(bundle, body_model_path)
        args = bundle.config["train_pipeline_args"]
        self.train_cfg = args["train_cfg"]
        self.losses_cfg = args["losses_cfg"]
        self.pred_type = args.get("pred_type", "velocity")
        self.validation_steps = args["infer_noise_scheduler_cfg"]["validation_steps"]
        self.cond_mask_prob = self.train_cfg.get("cond_mask_prob", 0.0)
        self._component_split_sizes = self.codec._component_split_sizes
        self._component_names = self.codec._component_names
        self.DEFAULT_LOSS_WEIGHTS = self.codec.DEFAULT_LOSS_WEIGHTS
        self.global_iteration = -1
        if "vertex" in self.losses_cfg:
            self.vertex_loss = VertexLoss(self.codec.mesh_model, **self.losses_cfg["vertex"])

    def train_step(self, batch):
        device = next(self.bundle.parameters()).device
        self.codec.to(device)
        batch = to_device(batch, device)
        lengths = batch["length"]
        frames = batch["inputs"]["feature"]["feature"].shape[1]
        if lengths.ndim != 1 or (lengths <= 0).any() or (lengths > frames).any():
            raise ValueError("length must contain positive frame counts within the padded batch")
        result = self.forward_batch(batch, global_step=self.get_global_step())
        return {"loss": result["loss"],
                **{f"loss_{k}": v.detach() for k, v in result["loss_dict"].items()}}

    @torch.no_grad()
    def val_step(self, batch):
        return self.train_step(batch)

    def forward_batch(self, batch: Dict[str, Any], global_step: int = 0) -> Dict[str, Any]:
        self.codec.synchronize_statistics(self.bundle)
        self.global_iteration = global_step

        gt_motion = self.codec.encode_motion(batch["target"])
        device = gt_motion.device
        length = batch["length"]
        max_length = length.max().item()

        ctxt_input = batch["inputs"]["feature"]

        gt_motion = gt_motion[:, :max_length]
        if isinstance(ctxt_input, dict):
            for key in ctxt_input:
                ctxt_input[key] = ctxt_input[key][:, :max_length]
        else:
            ctxt_input = ctxt_input[:, :max_length]

        vtxt_input = torch.zeros(
            (gt_motion.shape[0], 1, self.bundle.network.vtxt_input_dim), device=device
        )

        x0 = torch.randn(gt_motion.shape).to(device)
        x1 = gt_motion

        if "timestep_sample_method" in self.train_cfg:
            if self.train_cfg["timestep_sample_method"] == "logit_normal":
                timesteps = (
                    torch.randn(gt_motion.shape[0]) * self.train_cfg["t_sample_P_std"]
                    + self.train_cfg["t_sample_P_mean"]
                )
                timesteps = torch.sigmoid(timesteps)
            else:
                raise NotImplementedError(
                    f"unsupported timestep_sample_method: {self.train_cfg['timestep_sample_method']}"
                )
            timesteps = timesteps.to(device)
        else:
            timesteps = torch.rand(
                (gt_motion.shape[0],),
                dtype=gt_motion.dtype,
            ).to(device)

        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        # temporal mask
        x_mask_temporal = length_to_mask(length, gt_motion.shape[1])

        pred = self.bundle.predict_motion(
            x=phi,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=x_mask_temporal,
            cond_mask_prob=self.cond_mask_prob,
        )

        pred_decode = None
        gt_decode = None

        if self.pred_type == "velocity":
            pass
        elif self.pred_type == "x1":
            pred_decode = self.codec.decode_motion(pred)
            gt_decode = self.codec.decode_motion(x1)
            t_eps = 1 / self.validation_steps
            flow = (x1 - phi) / (1 - t).clamp_min(t_eps)
            pred = (pred - phi) / (1 - t).clamp_min(t_eps)
        elif self.pred_type == "x1raw":
            pred_decode = self.codec.decode_motion(pred)
            gt_decode = self.codec.decode_motion(x1)
            pred = pred - x0
        else:
            raise NotImplementedError(f"unsupported pred_type: {self.pred_type}")

        loss_dict, loss_weight = self.compute_loss(
            pred,
            flow,
            pred_decode=pred_decode,
            gt_decode=gt_decode,
            data_mask_temporal=x_mask_temporal,
        )

        loss = sum(loss_dict[k] * loss_weight[k] for k in loss_dict.keys())

        return {
            "latent": gt_motion,
            "model_output": pred,
            "loss": loss,
            "loss_dict": loss_dict,
            "tensor_results": {
                "index": batch.get("index", None),
            },
        }

    def compute_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_decode: Optional[Dict[str, Tensor]] = None,
        gt_decode: Optional[Dict[str, Tensor]] = None,
        data_mask_temporal: Optional[Tensor] = None,
    ) -> tuple:
        loss_fn = torch.nn.functional.smooth_l1_loss
        loss_dict = {}

        pred_parts = torch.split(pred, self._component_split_sizes, dim=-1)
        gt_parts = torch.split(gt, self._component_split_sizes, dim=-1)

        for i, name in enumerate(self._component_names):
            loss_dict[name] = loss_fn(
                pred_parts[i], gt_parts[i], reduction="none"
            ).mean(dim=-1)

        need_fk_rotation = self.losses_cfg.get("fk_rotation", {}).get("weight", 0.0) > 0
        need_fk_consistency = self.losses_cfg.get("fk_consistency", {}).get("weight", 0.0) > 0
        if (
            pred_decode is not None
            and gt_decode is not None
            and (need_fk_rotation or need_fk_consistency)
        ):
            # FK on predicted rotations
            pred_shapes_expanded = pred_decode["shapes"].expand(-1, pred_decode["root_rot6d"].shape[1], -1)
            gt_shapes_expanded = gt_decode["shapes"].expand(-1, gt_decode["root_rot6d"].shape[1], -1)

            pred_kp3d, pred_transforms = _run_fk(
                self.codec.body_model,
                pred_decode["root_rot6d"], pred_decode["body_rot6d"],
                pred_shapes_expanded, pred_decode["trans"],
            )
            gt_kp3d, gt_transforms = _run_fk(
                self.codec.body_model,
                gt_decode["root_rot6d"], gt_decode["body_rot6d"],
                gt_shapes_expanded, gt_decode["trans"],
            )

            B_fk, T_fk = pred_kp3d.shape[:2]
            if need_fk_rotation:
                pred_global_rot6d = rotation_matrix_to_rot6d(pred_transforms[:, :, :, :3, :3])
                gt_global_rot6d = rotation_matrix_to_rot6d(gt_transforms[:, :, :, :3, :3])
                fk_rot_loss = loss_fn(
                    pred_global_rot6d, gt_global_rot6d, reduction="none"
                )
                loss_dict["fk_rotation"] = fk_rot_loss.reshape(B_fk, T_fk, -1).mean(dim=-1)

            if need_fk_consistency:
                fk_pos_loss = loss_fn(pred_kp3d, gt_kp3d, reduction="none")
                loss_dict["fk_consistency"] = fk_pos_loss.reshape(B_fk, T_fk, -1).mean(dim=-1)

        if "vertex" in self.losses_cfg and pred_decode is not None and gt_decode is not None:
            loss_dict["vertex"] = self.vertex_loss(
                pred_decode, gt_decode, self.global_iteration
            )

        if "transroll" in self.losses_cfg and pred_decode is not None and gt_decode is not None:
            transroll_cfg = self.losses_cfg["transroll"]
            start_step = int(transroll_cfg.get("start_step", 0))
            overlap_step = int(transroll_cfg.get("overlap_step", 1))
            if self.global_iteration < start_step:
                warmup_scale = 0.0
            elif overlap_step <= 0 or self.global_iteration > start_step + overlap_step:
                warmup_scale = 1.0
            else:
                warmup_scale = (self.global_iteration - start_step) / overlap_step
            transroll_loss = loss_fn(
                pred_decode["trans"], gt_decode["trans"], reduction="none"
            ).mean(dim=-1)
            loss_dict["transroll"] = transroll_loss * warmup_scale

        loss_weight = {}
        for key in loss_dict:
            if key in self.losses_cfg:
                loss_weight[key] = self.losses_cfg[key].get("weight", self.DEFAULT_LOSS_WEIGHTS.get(key, 1.0))
            else:
                loss_weight[key] = self.DEFAULT_LOSS_WEIGHTS.get(key, 1.0)

        # temporal mask + mean
        loss_dict_mean = {}
        for key, val in loss_dict.items():
            if data_mask_temporal is not None:
                loss_dict_mean[key] = (val * data_mask_temporal).sum() / data_mask_temporal.sum()
            else:
                loss_dict_mean[key] = val.mean()

        return loss_dict_mean, loss_weight
