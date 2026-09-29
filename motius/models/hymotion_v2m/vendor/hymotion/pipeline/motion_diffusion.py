from __future__ import annotations
import os
import os.path as osp
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.signal import savgol_filter
from torch import Tensor
from torch.nn import functional as F
from torchdiffeq import odeint

from ..datasets.geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from ..evaluation.metrics import (
    calculate_jerk,
    calculate_motion_diversity,
    calculate_motion_similarity,
    calculate_translation_error,
    calculate_velocity_o6d,
)
from ..utils.loaders import load_object, read_yaml
from ..utils.motion_process import correct_translation_with_contact, recover_from_ric, recover_root_kpts
from ..utils.rotation_converter import matrix_to_quaternion, quaternion_fix_continuity, quaternion_to_matrix
from ..utils.type_converter import get_module_device
from .body_model import WoodenMesh
from .motion_flowmatching import length_to_mask, randn_tensor, rollout_local_transl_vel


def start_end_frame_to_mask(start_frame: Tensor, end_frame: Tensor, max_len: int) -> Tensor:
    # 生成一个 (B, max_len) 的mask，只有在[start_frame, end_frame]区间内为True，其余为False
    assert (start_frame >= 0).all() and (end_frame >= 0).all(), f"start_frame={start_frame}, end_frame={end_frame}"
    lengths = end_frame - start_frame + 1
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    batch_size = start_frame.shape[0]
    arange_ids = torch.arange(max_len, device=start_frame.device).unsqueeze(0).expand(batch_size, max_len)
    mask = (arange_ids >= start_frame.unsqueeze(1)) & (arange_ids <= end_frame.unsqueeze(1))
    return mask


class MotionGeneration(torch.nn.Module):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        text_encoder_module: str,
        text_encoder_cfg: dict,
        losses_cfg: Optional[dict],
        mean_std_dir: str,
        motion_type="auto",
        **kwargs,
    ):
        super().__init__()
        # build models and parameters
        self._network_module_args = deepcopy(network_module_args)
        self.motion_transformer = load_object(network_module, network_module_args)
        self._text_encoder_module = text_encoder_module
        self._text_encoder_cfg = deepcopy(text_encoder_cfg)
        self.motion_type = motion_type
        if motion_type.startswith("vae"):
            assert "vae_cfg" in kwargs, "vae_cfg is required for vae motion type"
            vae_config = read_yaml(kwargs["vae_cfg"]["config"])
            vae = load_object(
                vae_config["train_pipeline"],
                vae_config["train_pipeline_args"],
                network_module=vae_config["network_module"],
                network_module_args=vae_config["network_module_args"],
            )
            vae_ckpt = kwargs["vae_cfg"]["ckpt"]
            assert os.path.exists(vae_ckpt), f"Checkpoint {vae_ckpt} not found"
            state_dict = torch.load(vae_ckpt, map_location="cpu")
            state_dict = state_dict["model_state_dict"]
            vae.load_state_dict(state_dict)
            vae.eval()
            self.vae = vae
            # freeze vae parameters
            for param in vae.parameters():
                param.requires_grad = False

        self.null_vtxt_feat = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("vtxt_input_dim", 768))
        )
        self.null_ctxt_input = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("ctxt_input_dim", 4096))
        )
        self.special_game_vtxt_feat = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("vtxt_input_dim", 768))
        )
        self.special_game_ctxt_feat = torch.nn.Parameter(
            torch.randn(1, 1, self._network_module_args.get("ctxt_input_dim", 4096))
        )
        # build losses
        self.losses_cfg: dict[str, Any] = (
            losses_cfg if losses_cfg else {"recons": {"name": "SmoothL1Loss", "weight": 1.0}}
        )
        self._parse_losses_cfg()
        # build buffer
        self.mean_std_dir = mean_std_dir
        self._parse_buffer(self.motion_type)

        self.output_mesh_fps = kwargs.get("output_mesh_fps", 30)
        self.train_frames = kwargs.get("train_frames", 360)
        self.uncondition_mode = kwargs.get("uncondition_mode", False)
        self.enable_ctxt_null_feat = kwargs.get("enable_ctxt_null_feat", False)
        self.enable_special_game_feat = kwargs.get("enable_special_game_feat", False)

    def _parse_losses_cfg(self) -> None:
        for loss_name, loss_conf in list(self.losses_cfg.items()):
            if loss_name == "recons":
                # 1) 数值：仅作为权重，loss 默认使用 SmoothL1Loss(reduction="none")
                # 2) 字符串：loss 名称（如 "MSELoss" 或 "torch.nn.SmoothL1Loss"），默认权重 1.0
                # 3) 字典：{"name": <类名或全路径>, "args"/"loss_cfg": {...}, "weight": <float>}
                if isinstance(loss_conf, (int, float)):
                    self.reconstruction_loss_fn = torch.nn.SmoothL1Loss(reduction="none")
                elif isinstance(loss_conf, str):
                    cls = loss_conf
                    module_name = cls if "." in cls else f"torch.nn.{cls}"
                    self.reconstruction_loss_fn = load_object(module_name, {"reduction": "none"})
                    self.losses_cfg[loss_name] = 1.0
                elif isinstance(loss_conf, dict):
                    assert "name" in loss_conf, f"Missing 'name' in losses_cfg['{loss_name}']"
                    cls = loss_conf["name"]
                    module_name = cls if "." in cls else f"torch.nn.{cls}"
                    args = loss_conf.get("args", loss_conf.get("loss_cfg", {})) or {}
                    args = {"reduction": "none", **args}
                    self.reconstruction_loss_fn = load_object(module_name, args)
                    self.losses_cfg[loss_name] = float(loss_conf.get("weight", 1.0))
                else:
                    raise TypeError(f"Unsupported type for losses_cfg['{loss_name}']: {type(loss_conf)}")
            else:
                raise ValueError(f"Unsupported loss: {loss_name}")

    def compute_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        data_mask_temporal: Tensor,
        has_finger: Optional[Tensor] = None,
    ) -> dict[str, Tensor]:
        # constrcut the data mask matrix
        # temporal mask: (B, L, 1)
        data_mask_temporal = data_mask_temporal.float().unsqueeze(2)
        # spatial mask: (B, 1, D)
        data_mask_spatial = self._get_data_mask(gt, has_finger).float()
        # final mask: (B, L, D)
        data_mask = data_mask_temporal * data_mask_spatial
        weight = data_mask.expand_as(pred)
        # TODO: maybe add denorm here later

        if not torch.isfinite(gt).all():
            raise RuntimeError("NaN/Inf in gt_motion")
        if not torch.isfinite(pred).all():
            print("NaN/Inf in pred_motion")
            # raise RuntimeError("NaN/Inf in pred")

        losses = {}
        for loss_name, loss_weight in self.losses_cfg.items():
            if loss_name == "recons":
                # element-wise loss: (B, L, D)
                elem = self.reconstruction_loss_fn(pred, gt)
                avg_factor = weight.sum().clamp_min(1.0)
                loss = (elem * weight).sum() / avg_factor
                # per sample loss: (B,)
                sample_factor = weight.view(weight.shape[0], -1).sum(dim=1).clamp_min(1.0)
                per_sample_loss = (elem * weight).reshape(elem.shape[0], -1).sum(dim=1) / sample_factor
            else:
                raise ValueError(f"Unsupported loss: {loss_name}")
            losses[loss_name] = loss_weight * loss
            losses["per_sample_loss"] = per_sample_loss.detach()
        losses["loss"] = sum(v for k, v in losses.items() if k != "per_sample_loss")
        return losses

    def _parse_buffer(self, mode: str) -> None:
        self.body_model = WoodenMesh()
        self._find_motion_type(mode=mode)
        self._load_mean_std()

    def _load_mean_std(self, mean_std_name: Optional[str] = None) -> None:
        mean_std_name = self.mean_std_dir if mean_std_name is None else mean_std_name
        if mean_std_name is not None and osp.isdir(mean_std_name):
            if osp.exists(osp.join(mean_std_name, "Mean.npy")):
                mean_path = osp.join(mean_std_name, "Mean.npy")
            else:
                mean_path = osp.join(mean_std_name, "mean.npy")
            if osp.exists(osp.join(mean_std_name, "Std.npy")):
                std_path = osp.join(mean_std_name, "Std.npy")
            else:
                std_path = osp.join(mean_std_name, "std.npy")
            mean = torch.from_numpy(np.load(mean_path)).float()
            std = torch.from_numpy(np.load(std_path)).float()
            self._assert_motion_dimension(mean.unsqueeze(0), std.unsqueeze(0))
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        elif mean_std_name is not None and osp.isfile(mean_std_name):
            print(f"[{self.__class__.__name__}] Loading mean_std from {mean_std_name}")
            from .motion_vae import DebugRot6dDataset

            mean, std = DebugRot6dDataset.load_mean_std(mean_std_name)
            self.register_buffer("mean", mean[None])
            self.register_buffer("std", std[None])
        else:
            print(
                f"[{self.__class__.__name__}] No mean_std found, using blank mean_std, self.mean_std_dir={self.mean_std_dir}"
            )
            self.register_buffer("mean", torch.zeros(1))
            self.register_buffer("std", torch.ones(1))

    # TODO: remove this function later
    def _assert_motion_dimension(self, mean: Tensor, std: Tensor) -> None:
        # motion: (L, D)
        assert mean.shape == std.shape, f"mean.shape={mean.shape} != std.shape={std.shape}"
        assert mean.ndim == 2, f"mean.ndim={mean.ndim} != 2"

        if self.motion_type == "o6dp":
            assert mean.shape == (1, 272), f"mean.shape={mean.shape} != (1, 272)"
        elif self.motion_type == "o6dp_with_finger":
            assert mean.shape == (1, 632), f"mean.shape={mean.shape} != (1, 632)"
        elif self.motion_type == "o6dp_1103" or self.motion_type == "o6dp_1103_rel":
            assert mean.shape == (1, 201), f"mean.shape={mean.shape} != (1, 201)"
        elif self.motion_type == "vae":
            assert mean.shape == (1, 256), f"mean.shape={mean.shape} != (1, 256)"
        else:
            raise ValueError(f"Unsupported motion type: {self.motion_type}")

    def _find_motion_type(self, mode: str) -> None:
        assert mode in [
            "auto",
            "o6dp",
            "o6dp_with_finger",
            "o6dp_1103",
            "o6dp_1103_rel",
            "vae",
        ] or mode.startswith(
            "vae"
        ), f"mode must be one of auto, hml3d, o6d, o6d_with_finger, o6dp, o6dp_with_finger, vae, but got {mode}"
        if mode == "auto":
            if self.motion_transformer.motion_input_dim == 272:
                self.motion_type = "o6dp"
            elif self.motion_transformer.motion_input_dim == 632:
                self.motion_type = "o6dp_with_finger"
            elif self.motion_transformer.motion_input_dim == 201:
                self.motion_type = "o6dp_1103"
            elif self.motion_transformer.motion_input_dim == 135:
                self.motion_type = "o6dp_1103_rel"
            elif self.motion_transformer.motion_input_dim == 256:
                self.motion_type = "vae"
            else:
                raise ValueError(f"Unsupported motion type: {self.motion_transformer.motion_input_dim}")
        else:
            self.motion_type = mode

    def set_epoch(self, epoch) -> None:
        self.current_epoch = epoch

    def load_in_demo(
        self,
        ckpt_name: str,
        build_text_encoder: bool = True,
        allow_empty_ckpt: bool = False,
    ) -> None:
        if not allow_empty_ckpt:
            assert os.path.exists(ckpt_name), f"{ckpt_name} not found"
            checkpoint = torch.load(ckpt_name, map_location="cpu", weights_only=False)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                # FIXME: 这里是一个临时修复，后续需要修改
                checkpoint["model_state_dict"]["mean"] = (
                    checkpoint["model_state_dict"]["mean"].squeeze(0)
                    if checkpoint["model_state_dict"]["mean"].ndim == 2
                    else checkpoint["model_state_dict"]["mean"]
                )
                checkpoint["model_state_dict"]["std"] = (
                    checkpoint["model_state_dict"]["std"].squeeze(0)
                    if checkpoint["model_state_dict"]["std"].ndim == 2
                    else checkpoint["model_state_dict"]["std"]
                )
                self.load_state_dict(checkpoint["model_state_dict"], strict=False)
            else:
                self.load_state_dict(checkpoint, strict=False)
        self.motion_transformer.eval()
        if build_text_encoder and not self.uncondition_mode:
            self.text_encoder = load_object(self._text_encoder_module, self._text_encoder_cfg)
            self.text_encoder.to(get_module_device(self))

    @torch.no_grad()
    def encode_text(self, text: Dict[str, List[str]]) -> Dict[str, Tensor]:
        if not hasattr(self, "text_encoder"):
            self.text_encoder = load_object(self._text_encoder_module, self._text_encoder_cfg)
            self.text_encoder.to(get_module_device(self))
        text = text["text"]
        vtxt_input, ctxt_input, ctxt_length = self.text_encoder.encode(text=text)
        return {
            "text_vec_raw": vtxt_input,
            "text_ctxt_raw": ctxt_input,
            "text_ctxt_raw_length": ctxt_length,
        }

    def decode_motion_from_latent(self, latent: Tensor, should_apply_smooothing: bool = True) -> Dict[str, Tensor]:
        device = get_module_device(self)

        std_zero = self.std < 1e-3
        std = torch.where(std_zero, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean
        if self.motion_type == "o6dp":
            return self._decode_o6dp(
                latent_denorm,
                num_joints=22,
                should_apply_smooothing=should_apply_smooothing,
            )
        elif self.motion_type == "o6dp_with_finger":
            return self._decode_o6dp(
                latent_denorm,
                num_joints=52,
                should_apply_smooothing=should_apply_smooothing,
            )
        elif self.motion_type == "o6dp_1103":
            return self._decode_o6dp_1103(
                latent_denorm,
                num_joints=22,
                rel_trans=False,
                should_apply_smooothing=should_apply_smooothing,
            )
        elif self.motion_type == "o6dp_1103_rel":
            return self._decode_o6dp_1103(
                latent_denorm,
                num_joints=22,
                rel_trans=True,
                should_apply_smooothing=should_apply_smooothing,
            )
        elif self.motion_type.startswith("vae"):
            return self._decode_vae(latent_denorm, should_apply_smooothing=should_apply_smooothing)
        else:
            raise ValueError(f"Unsupported motion type: {self.motion_type}")

    def _forward_smpl_batch(
        self,
        root_rot6d: Tensor,  # (B, L, 1, 6)
        body_rot6d: Tensor,  # (B, L, 21, 6) 或 (B, L, 51, 6) 的前 21
        transl: Tensor,  # (B, L, 3)
        left_hand_pose: Optional[Tensor] = None,  # (B, L, 15, 6)
        right_hand_pose: Optional[Tensor] = None,  # (B, L, 16, 6)
    ) -> Tensor:
        device = transl.device
        bsz, L = transl.shape[:2]
        k3d_all = []
        tmp_betas = torch.zeros(1, 16, device=device)
        # FIXME: 这里是一个临时修复，后续需要修改
        if not hasattr(self, "smpl_body_model"):
            return torch.zeros(bsz, L, 22, 3, device=device)
        for bs in range(bsz):
            out = self.smpl_body_model(
                body_rot6d[bs],
                tmp_betas,
                root_rot6d[bs],
                transl[bs],
                left_hand_pose=(left_hand_pose[bs] if left_hand_pose is not None else None),
                right_hand_pose=(right_hand_pose[bs] if right_hand_pose is not None else None),
            )
            k3d_all.append(out.detach().cpu())
        return torch.stack(k3d_all, dim=0)  # (B, L, J, 3)

    def _decode_o6dp(
        self,
        latent_denorm: Tensor,
        num_joints: int,
        should_correct_foot_contact: bool = False,
        should_apply_smooothing: bool = True,
    ) -> dict:
        device = get_module_device(self)
        B, L = latent_denorm.shape[:2]
        nj = num_joints
        body_n = nj - 1

        root_ry_vel_world = latent_denorm[..., 0:1].clone()
        root_xz_vel_body = latent_denorm[..., 1:3].clone()
        root_y_transl = latent_denorm[..., 3:4].clone()
        root_rot6d = latent_denorm[..., 4:10].clone().reshape(B, L, 1, 6)

        body6d_start = 10
        body6d_end = body6d_start + body_n * 6
        body_rot6d_full = latent_denorm[..., body6d_start:body6d_end].clone().reshape(B, L, body_n, 6)

        ric_end = body6d_end + nj * 3
        ric_joints_locations = latent_denorm[..., body6d_end:ric_end].clone().reshape(B, L, nj, 3)

        vel_end = ric_end + nj * 3
        joints_vel_body = latent_denorm[..., ric_end:vel_end].clone().reshape(B, L, nj, 3)

        foot_end = vel_end + 4
        foot_detect = latent_denorm[..., vel_end:foot_end].clone()
        foot_prob = torch.clamp(torch.sigmoid(foot_detect), 0.0, 1.0).detach()  # (B, L, 4)

        pred_rotmat = rot6d_to_rotation_matrix(root_rot6d.reshape(B, L, 6))

        transl = recover_root_kpts(
            root_ry_vel_world, root_xz_vel_body, root_y_transl, full_root_rotations_mat=pred_rotmat
        ).to(device)

        # 52 关节需要拆出手部
        left_hand_pose = right_hand_pose = None
        if nj == 52:
            body_rot6d = body_rot6d_full[:, :, :21, :].clone()
            left_hand_pose = body_rot6d_full[:, :, 21:36, :].clone()
            right_hand_pose = body_rot6d_full[:, :, 36:51, :].clone()
        else:
            body_rot6d = body_rot6d_full

        # 打包 rot6d（含手时拼回 51）
        if left_hand_pose is not None and right_hand_pose is not None:
            body_full = torch.cat([body_rot6d, left_hand_pose, right_hand_pose], dim=2)
        else:
            body_full = body_rot6d
        rot6d = torch.cat([root_rot6d, body_full], dim=2)  # (B, L, nj, 6)
        if should_apply_smooothing:
            rot6d_smooth = self.smooth_with_slerp(rot6d, sigma=1.0)
        else:
            rot6d_smooth = rot6d
        root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])  # (B, L, 3, 3)

        if should_correct_foot_contact:
            k3d = self._forward_smpl_batch(
                root_rot6d.to(device),
                body_rot6d.to(device),
                transl,
                left_hand_pose=(left_hand_pose.to(device) if left_hand_pose is not None else None),
                right_hand_pose=(right_hand_pose.to(device) if right_hand_pose is not None else None),
            )
            transl_fixed = correct_translation_with_contact(
                k3d,
                transl,
                foot_prob,
                joint_ids=[7, 10, 8, 11],
                on_thr=0.50,
                off_thr=0.30,
                morph_min_len=3,
                morph_max_gap=2,
            )
        else:
            transl_fixed = transl.detach()
        if should_apply_smooothing:
            transl_smooth = self.smooth_with_savgol(transl_fixed.detach(), window_length=11, polyorder=5)
        else:
            transl_smooth = transl_fixed

        with torch.no_grad():
            vertices_all = []
            k3d_all = []
            for bs in range(rot6d_smooth.shape[0]):
                out = self.body_model.forward({"rot6d": rot6d_smooth[bs], "trans": transl_smooth[bs]})
                vertices_all.append(out["vertices"])
                k3d_all.append(out["keypoints3d"])
            vertices = torch.stack(vertices_all, dim=0)
            k3d = torch.stack(k3d_all, dim=0)
            # 地面对齐
            min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
            # print(f"{self.__class__.__name__} min_y: {min_y}")
            k3d = k3d.clone()
            k3d[..., 1] -= min_y  # (B, L, J) - (B, 1, 1)
            transl_smooth = transl_smooth.clone()
            transl_smooth[..., 1] -= min_y.squeeze(-1).to(device)  # (B, L) - (B, 1)

        return dict(
            latent_denorm=latent_denorm,  # (B, L, 272/632)
            keypoints3d=k3d,  # (B, L, J, 3)
            rot6d=rot6d_smooth,  # (B, L, J, 6)
            transl=transl_smooth,  # (B, L, 3)
            ric3d=ric_joints_locations,  # (B, L, J, 3)
            vel_body=joints_vel_body,  # (B, L, J, 3)
            foot_prob=foot_prob,  # (B, L, 4)
            root_rotations_mat=root_rotmat_smooth,  # (B, L, 3, 3)
        )

    def _decode_o6dp_1103(
        self,
        latent_denorm: torch.Tensor,
        num_joints: int,
        rel_trans: bool = False,
        should_apply_smooothing: bool = True,
    ) -> dict:
        device = get_module_device(self)
        B, L = latent_denorm.shape[:2]
        nj = num_joints
        body_n = nj - 1

        if not rel_trans:
            transl = latent_denorm[..., 0:3].clone()
        else:
            transl = torch.cumsum(latent_denorm[..., 0:3].clone(), dim=1) / self.output_mesh_fps
        root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()

        body6d_start = 9
        body6d_end = body6d_start + body_n * 6
        body_rot6d_full = latent_denorm[..., body6d_start:body6d_end].clone().reshape(B, L, body_n, 6)

        # 52 关节需要拆出手部
        left_hand_pose = right_hand_pose = None
        if nj == 52:
            body_rot6d = body_rot6d_full[:, :, :21, :].clone()
            left_hand_pose = body_rot6d_full[:, :, 21:36, :].clone()
            right_hand_pose = body_rot6d_full[:, :, 36:51, :].clone()
        else:
            body_rot6d = body_rot6d_full

        if left_hand_pose is not None and right_hand_pose is not None:
            body_full = torch.cat([body_rot6d, left_hand_pose, right_hand_pose], dim=2)
        else:
            body_full = body_rot6d
        rot6d = torch.cat([root_rot6d, body_full], dim=2)  # (B, L, nj, 6)
        if should_apply_smooothing:
            # 只对前22个关节（非手指部分）应用slerp平滑
            rot6d_body = rot6d[:, :, :22, :]  # (B, L, 22, 6)
            rot6d_fingers = rot6d[:, :, 22:, :]  # (B, L, J-22, 6)
            rot6d_body_smooth = self.smooth_with_slerp(rot6d_body, sigma=1.0)
            rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)
        else:
            rot6d_smooth = rot6d
        root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])  # (B, L, 3, 3)

        transl_fixed = transl.detach()
        if should_apply_smooothing:
            transl_smooth = self.smooth_with_savgol(transl_fixed.detach(), window_length=11, polyorder=5)
        else:
            transl_smooth = transl_fixed

        with torch.no_grad():
            vertices_all = []
            k3d_all = []
            for bs in range(rot6d_smooth.shape[0]):
                out = self.body_model.forward({"rot6d": rot6d_smooth[bs], "trans": transl_smooth[bs]})
                vertices_all.append(out["vertices"])
                k3d_all.append(out["keypoints3d"])
            vertices = torch.stack(vertices_all, dim=0)
            k3d = torch.stack(k3d_all, dim=0)
            # 地面对齐
            min_y = vertices[..., 1].amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
            # print(f"{self.__class__.__name__} min_y: {min_y}")
            k3d = k3d.clone()
            k3d[..., 1] -= min_y  # (B, L, J) - (B, 1, 1)
            transl_smooth = transl_smooth.clone()
            transl_smooth[..., 1] -= min_y.squeeze(-1).to(device)  # (B, L) - (B, 1)

        return dict(
            latent_denorm=latent_denorm,  # (B, L, 201)
            keypoints3d=k3d,  # (B, L, J, 3)
            rot6d=rot6d_smooth,  # (B, L, J, 6)
            transl=transl_smooth,  # (B, L, 3)
            root_rotations_mat=root_rotmat_smooth,  # (B, L, 3, 3)
        )

    def _decode_vae(self, latent_denorm: Tensor, should_apply_smooothing: bool = False) -> dict:
        device = get_module_device(self)
        B, L = latent_denorm.shape[:2]

        decode_pred = self.vae.decode(latent_denorm)
        fk_pred = self.vae.forward_body(decode_pred)["params_decode"]

        k3d = fk_pred["keypoints3d"].detach().cpu()
        transl = fk_pred["trans"]
        rot6d = fk_pred["rot6d"]

        if should_apply_smooothing:
            # 只对前22个关节（非手指部分）应用slerp平滑
            rot6d_body = rot6d[:, :, :22, :]  # (B, L, 22, 6)
            rot6d_fingers = rot6d[:, :, 22:, :]  # (B, L, J-22, 6)
            rot6d_body_smooth = self.smooth_with_slerp(rot6d_body, sigma=1.0)
            rot6d_smooth = torch.cat([rot6d_body_smooth, rot6d_fingers], dim=2)
        else:
            rot6d_smooth = rot6d
        root_rotmat_smooth = rot6d_to_rotation_matrix(rot6d_smooth[:, :, 0, :])  # (B, L, 3, 3)

        transl_fixed = transl.detach()
        if should_apply_smooothing:
            transl_smooth = self.smooth_with_savgol(transl_fixed.detach(), window_length=11, polyorder=5)
        else:
            transl_smooth = transl_fixed

        # 地面对齐
        for bs in range(B):
            min_y = k3d[..., 1].amin(dim=(1, 2), keepdim=True)  # (B, 1, 1)
            k3d = k3d.clone()
            k3d[..., 1] -= min_y  # (B, L, J) - (B, 1, 1)
            transl_smooth = transl_smooth.clone()
            transl_smooth[..., 1] -= min_y.squeeze(-1).to(device)  # (B, L) - (B, 1)

        ret = dict(
            latent_denorm=latent_denorm,  # (B, L, 256)
            keypoints3d=k3d,  # (B, L, J, 3)
            keypoints3d_woshapes_wotrans=fk_pred["keypoints3d_woshapes_wotrans"],  # (B, L, J, 3)
            rot6d=rot6d_smooth,  # (B, L, J, 6)
            transl=transl_smooth,  # (B, L, 3)
            root_rotations_mat=root_rotmat_smooth,  # (B, L, 3, 3)
            shapes=fk_pred["shapes"],
        )

        return ret

    @staticmethod
    def smooth_with_savgol(input: torch.Tensor, window_length: int = 9, polyorder: int = 5) -> torch.Tensor:
        if len(input.shape) == 2:
            is_batch = False
            input = input.unsqueeze(0)
        else:
            is_batch = True
        input_np = input.cpu().numpy()
        input_smooth_np = np.empty_like(input_np, dtype=np.float32)
        for b in range(input_np.shape[0]):
            for j in range(input_np.shape[2]):
                input_smooth_np[b, :, j] = savgol_filter(input_np[b, :, j], window_length, polyorder)
        input_smooth = torch.from_numpy(input_smooth_np).to(input)
        if not is_batch:
            input_smooth = input_smooth.squeeze(0)
        return input_smooth

    @staticmethod
    def smooth_with_slerp(input: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
        from ..utils.motion_process import smooth_rotation
        from .smpl_lite import SMPLX_JOINTS

        def fix_time_continuity(q: Tensor, time_dim: int = -3):
            shape = q.shape
            qv = q.moveaxis(time_dim, 0).contiguous().view(shape[time_dim], -1, 4)
            qv = quaternion_fix_continuity(qv)
            return qv.view(shape[time_dim], *shape[:time_dim], *shape[time_dim + 1 :]).moveaxis(0, time_dim)

        num_joints = input.shape[2]
        RR = rot6d_to_rotation_matrix(input)
        qq = matrix_to_quaternion(RR)
        qq_np = fix_time_continuity(qq, time_dim=1).cpu().numpy()
        qq_s_np = smooth_rotation(
            qq_np,
            joint_names=SMPLX_JOINTS[:num_joints],
            smooth_joints=SMPLX_JOINTS[:num_joints],
            sigma=sigma,
        )
        input_smooth = rotation_matrix_to_rot6d(quaternion_to_matrix(torch.from_numpy(qq_s_np)))
        return input_smooth.to(input.device)

    @staticmethod
    def noise_from_seeds(latent: Tensor, seeds: Union[int, List[int]], seed_start: int = 0) -> Tensor:
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        noise_list = []
        B = latent.shape[0]
        shape = (B, *latent.shape[1:])
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed + seed_start)
            noise_sample = randn_tensor(shape, generator=generator, dtype=latent.dtype).to(latent.device)
            noise_list.append(noise_sample)
        return torch.cat(noise_list, dim=0)

    def get_shape_of_noise(self) -> torch.Size:
        return torch.Size([1, self.train_frames, self.motion_transformer.motion_input_dim])

    def _get_data_mask(self, gt: Tensor, has_finger: Optional[Tensor] = None) -> Tensor:
        # gt: (B, L, D)
        B, _, D = gt.shape
        device = gt.device
        mask = torch.ones((B, 1, D), dtype=torch.bool, device=device)

        if self.motion_transformer.motion_input_dim == 632 and D == 632:
            o6d_start = 10
            num_o6d_body = 21 * 6
            num_o6d_full = 51 * 6
            o6d_pad = slice(o6d_start + num_o6d_body, o6d_start + num_o6d_full)

            rifke_start = o6d_start + num_o6d_full
            num_rifke_body = 22 * 3
            num_rifke_full = 52 * 3
            rifke_pad = slice(rifke_start + num_rifke_body, rifke_start + num_rifke_full)

            rifke_vel_start = rifke_start + num_rifke_full
            num_rifke_vel_body = 22 * 3
            num_rifke_vel_full = 52 * 3
            rifke_vel_pad = slice(
                rifke_vel_start + num_rifke_vel_body,
                rifke_vel_start + num_rifke_vel_full,
            )

            if has_finger is not None:
                no_finger = ~has_finger
            else:
                no_finger = (
                    gt[:, :, o6d_start + num_o6d_body : o6d_start + num_o6d_full].abs().sum(dim=(1, 2)) == 0
                )  # (B,)

            if torch.any(no_finger):
                mask[no_finger, :, o6d_pad] = False
                mask[no_finger, :, rifke_pad] = False
                mask[no_finger, :, rifke_vel_pad] = False
        return mask

    def _mask_cond(
        self, cond_vtxt: Tensor, cond_ctxt: Tensor, force_mask: bool = False, cond_mask_prob: float = 0.1
    ) -> Tuple[Tensor, Tensor]:
        bs = cond_vtxt.shape[0]
        if force_mask:
            return self.null_vtxt_feat.expand(*cond_vtxt.shape), self.null_ctxt_input.expand(*cond_ctxt.shape)
        elif self.training and cond_mask_prob > 0.0:
            mask = (
                torch.bernoulli(torch.ones(bs, device=cond_vtxt.device) * cond_mask_prob).view(bs, 1).bool()
            )  # 1-> use null_cond, 0-> use real cond

            mask_vtxt = mask
            while len(mask_vtxt.shape) < len(cond_vtxt.shape):
                mask_vtxt = mask_vtxt[..., None]
            cond_vtxt = torch.where(mask_vtxt, self.null_vtxt_feat.expand(*cond_vtxt.shape), cond_vtxt)

            mask_ctxt = mask
            while len(mask_ctxt.shape) < len(cond_ctxt.shape):
                mask_ctxt = mask_ctxt[..., None]
            cond_ctxt = torch.where(mask_ctxt, self.null_ctxt_input.expand(*cond_ctxt.shape), cond_ctxt)
            return cond_vtxt, cond_ctxt
        else:
            return cond_vtxt, cond_ctxt

    def _maybe_inject_source_token(
        self,
        vtxt_input: Tensor,
        ctxt_input: Tensor,
        ctxt_mask_temporal: Tensor,
        sources: Optional[List[str]],
        trigger_sources: Optional[set] = None,
        prob: float = 0.5,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if (sources is None or trigger_sources is None) or not self.enable_special_game_feat:
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        B, Lc, Dc = ctxt_input.shape
        assert (
            isinstance(sources, (list, tuple)) and len(sources) == B
        ), f"sources长度需等于batch: {len(sources)} vs {B}"

        trig = set(s.lower() for s in trigger_sources)
        src_mask = torch.tensor(
            [str(s).lower() in trig for s in sources], dtype=torch.bool, device=ctxt_input.device
        )  # (B,)
        if not src_mask.any():
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        rand_mask = (
            torch.rand(B, device=ctxt_input.device) < prob
            if self.training
            else torch.BoolTensor(B).fill_(True).to(ctxt_input.device)
        )
        apply_mask = src_mask & rand_mask
        if not apply_mask.any():
            return vtxt_input, ctxt_input, ctxt_mask_temporal

        # vtxt：仅对命中样本做加法混合
        vtxt_token = self.special_game_vtxt_feat.to(vtxt_input).expand(B, 1, -1)
        vtxt_input = vtxt_input + vtxt_token * apply_mask.view(B, 1, 1).to(vtxt_input.dtype)

        # 计算每个样本当前有效长度
        if ctxt_mask_temporal.dtype == torch.bool:
            cur_len = ctxt_mask_temporal.sum(dim=1).long()  # (B,)
        else:
            cur_len = (ctxt_mask_temporal > 0).sum(dim=1).long()

        # 1) 对“未满”的命中样本，原地在 cur_len 位置写入 special token，并把该位mask置True
        can_inplace = apply_mask & (cur_len < Lc)
        b_inplace = torch.nonzero(can_inplace, as_tuple=False).squeeze(1)  # (K,)
        if b_inplace.numel() > 0:
            pos = cur_len[b_inplace]  # (K,)
            token = self.special_game_ctxt_feat.squeeze(0).squeeze(0).to(ctxt_input)  # (Dc,)
            ctxt_input[b_inplace, pos, :] = token.unsqueeze(0).expand(b_inplace.numel(), Dc)
            if ctxt_mask_temporal.dtype == torch.bool:
                ctxt_mask_temporal[b_inplace, pos] = True
            else:
                ctxt_mask_temporal[b_inplace, pos] = 1

        # 2) 若存在“已满”的命中样本，需要统一pad一位：满的样本在新位写入special，其他样本补零且mask=False
        need_expand = (apply_mask & (cur_len >= Lc)).any()
        if need_expand:
            suffix = torch.zeros((B, 1, Dc), dtype=ctxt_input.dtype, device=ctxt_input.device)
            full_hit = apply_mask & (cur_len >= Lc)
            b_full = torch.nonzero(full_hit, as_tuple=False).squeeze(1)
            if b_full.numel() > 0:
                suffix[b_full, 0, :] = (
                    self.special_game_ctxt_feat.expand(b_full.numel(), 1, -1).to(ctxt_input).squeeze(1)
                )
            ctxt_input = torch.cat([ctxt_input, suffix], dim=1)

            if ctxt_mask_temporal.dtype == torch.bool:
                suffix_mask = torch.zeros((B, 1), dtype=torch.bool, device=ctxt_input.device)
                suffix_mask[b_full, 0] = True
            else:
                suffix_mask = torch.zeros((B, 1), dtype=ctxt_mask_temporal.dtype, device=ctxt_input.device)
                suffix_mask[b_full, 0] = 1
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal, suffix_mask], dim=1)

        return vtxt_input, ctxt_input, ctxt_mask_temporal


class MotionDiffusion(MotionGeneration):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        text_encoder_module: str,
        text_encoder_cfg: dict,
        noise_scheduler_module: str,
        noise_scheduler_cfg: dict,
        infer_noise_scheduler_module: str,
        infer_noise_scheduler_cfg: dict,
        mean_std_dir: Optional[str] = None,
        losses_cfg: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            network_module=network_module,
            network_module_args=network_module_args,
            text_encoder_module=text_encoder_module,
            text_encoder_cfg=text_encoder_cfg,
            losses_cfg=losses_cfg,
            mean_std_dir=(mean_std_dir if mean_std_dir is not None else (test_cfg or {})["mean_std_dir"]),
            **kwargs,
        )
        # build scheduler
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg)
        self.noise_scheduler = load_object(noise_scheduler_module, noise_scheduler_cfg)
        self._infer_noise_scheduler_cfg = deepcopy(infer_noise_scheduler_cfg)
        self.infer_noise_scheduler = load_object(infer_noise_scheduler_module, infer_noise_scheduler_cfg)
        # additional cfg
        self.train_cfg = deepcopy(train_cfg) if train_cfg else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg else dict()
        self._parse_train_cfg()
        self._parse_test_cfg()

    def _parse_train_cfg(self) -> None:
        self.cond_mask_prob = self.train_cfg.get("cond_mask_prob", 0.0)

    def _parse_test_cfg(self) -> None:
        self.validation_steps = self.test_cfg.get("num_inference_timesteps", 50)
        self.text_guidance_scale = self.test_cfg.get("text_guidance_scale", 1)

    def forward(
        self,
        noise: Tensor,
        noise_length: Tensor,
        hidden_state_dict: Dict[str, Tensor],
        t: Tensor,
        cfg_scale: Optional[float] = None,
    ):
        # NOTE: this is for single forward without infer_noise_scheduler.step (for reinforcement learning)
        # noise: (B, train_frames, D)
        # noise_length: (B, 1)
        # hidden_state: (B, max_text_len, D)
        # hidden_state_length: (B, 1)
        # t: (B, 1)
        vtxt_input = hidden_state_dict["text_vec_raw"]
        ctxt_input = hidden_state_dict["text_ctxt_raw"]
        ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_mask_temporal = length_to_mask(noise_length, noise.shape[1])

        text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
        do_classifier_free_guidance = text_guidance_scale > 1.0 and not self.uncondition_mode
        if do_classifier_free_guidance is True:
            silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)

            if self.enable_ctxt_null_feat:
                silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
            else:
                silent_ctxt_input = ctxt_input
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)

            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        x_input = torch.cat([noise] * 2, dim=0) if do_classifier_free_guidance else noise
        t = torch.cat([t] * 2, dim=0) if do_classifier_free_guidance else t
        x_pred = self.motion_transformer(
            x=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=t,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )
        if do_classifier_free_guidance:
            x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
            x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)
        return x_pred

    def forward_in_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = get_module_device(self)

        # allow externally provided timesteps (e.g., for fair eval); fallback to random
        fixed_timesteps = batch.get("fixed_train_timesteps", None)

        gt_motion = batch["motion"]
        noise = torch.randn(gt_motion.shape, dtype=gt_motion.dtype).to(device)

        if fixed_timesteps is not None:
            timesteps = fixed_timesteps.to(device=device, dtype=gt_motion.dtype).view(-1)
        else:
            timesteps = torch.rand((gt_motion.shape[0],), dtype=gt_motion.dtype).to(device)

        x_t = self.noise_scheduler.add_noise(gt_motion, noise, timesteps)

        if "text_vec_raw" in batch.keys():
            vtxt_input = batch["text_vec_raw"]
            ctxt_input = batch["text_ctxt_raw"]
            ctxt_length = batch["text_ctxt_raw_length"]
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            vtxt_input, ctxt_input = self._mask_cond(
                vtxt_input,
                ctxt_input,
                force_mask=False,
                cond_mask_prob=self.cond_mask_prob,
            )
            sources = (batch.get("data_meta", {}) or {}).get("data_src", None)
            vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
                vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
            )
        else:
            vtxt_input = self.null_vtxt_feat.expand(gt_motion.shape[0], 1, -1)
            ctxt_input = self.null_ctxt_input.expand(gt_motion.shape[0], 1, -1)
            ctxt_length = torch.tensor([1]).expand(gt_motion.shape[0])
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(gt_motion.shape[0], -1)

        if "motion_start_frame" in batch.keys() and "motion_end_frame" in batch.keys():
            x_mask_temporal = start_end_frame_to_mask(
                batch["motion_start_frame"],
                batch["motion_end_frame"],
                gt_motion.shape[1],
            )
        else:
            x_mask_temporal = length_to_mask(batch["length"], gt_motion.shape[1])

        pred = self.motion_transformer(
            x=x_t,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        with torch.autocast("cuda", enabled=False):
            pred_fp32 = pred.float()
            gt_fp32 = gt_motion.float()
            mask_fp32 = x_mask_temporal.float()
            losses = self.compute_loss(
                pred_fp32, gt_fp32, data_mask_temporal=mask_fp32, has_finger=batch.get("has_finger", None)
            )
        per_sample_loss = losses.pop("per_sample_loss", None)
        return {
            "latent": gt_motion,
            "model_output": pred,
            "loss": losses["loss"],
            "loss_dict": losses,
            "tensor_results": {
                "per_sample_loss": per_sample_loss,
                "index": batch.get("index", None),
            },
        }

    @torch.no_grad()
    def validate(self, batch: Dict[str, Any], seeds: List[int] = [0, 1, 2, 3]) -> Dict[str, Any]:
        device = get_module_device(self)
        if self.motion_type.startswith("vae"):
            gt_motion = self.vae.encode_motion(batch)
        else:
            gt_motion = batch["motion"]
        x = self.noise_from_seeds(gt_motion, seeds)

        self.infer_noise_scheduler.set_timesteps(self.validation_steps)
        timesteps = self.infer_noise_scheduler.timesteps.to(device)

        if "text_vec_raw" in batch.keys():
            vtxt_input = batch["text_vec_raw"]
            ctxt_input = batch["text_ctxt_raw"]
            ctxt_length = batch["text_ctxt_raw_length"]
        else:
            vtxt_input = self.null_vtxt_feat
            ctxt_input = self.null_ctxt_input
            ctxt_length = torch.tensor([1])

        repeat = len(seeds)
        vtxt_input = vtxt_input.repeat((repeat,) + (1,) * (vtxt_input.dim() - 1))
        ctxt_input = ctxt_input.repeat((repeat,) + (1,) * (ctxt_input.dim() - 1))
        ctxt_length = ctxt_length.repeat(repeat)
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        sources = (batch.get("data_meta", {}) or {}).get("data_src", None)
        sources = (batch.get("data_meta", {}) or {}).get("data_src", None)
        if sources is not None:
            sources = sources * repeat
        vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
            vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
        )
        if "motion_start_frame" in batch.keys() and "motion_end_frame" in batch.keys():
            x_mask_temporal = start_end_frame_to_mask(
                batch["motion_start_frame"],
                batch["motion_end_frame"],
                gt_motion.shape[1],
            )
        else:
            x_mask_temporal = length_to_mask(batch["length"], gt_motion.shape[1])
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))

        x = self._denoise_loop(
            x=x,
            x_mask_temporal=x_mask_temporal,
            vtxt_input=vtxt_input,
            ctxt_input=ctxt_input,
            ctxt_mask_temporal=ctxt_mask_temporal,
            timesteps=timesteps,
        )

        with torch.autocast("cuda", enabled=False):
            pred_output_dict = self.decode_motion_from_latent(x, should_apply_smooothing=False)
            gt_output_dict = self.decode_motion_from_latent(gt_motion, should_apply_smooothing=False)

        for kk, vv in pred_output_dict.items():
            pred_output_dict[kk] = vv[:, : batch["length"][0].item(), ...]
        for kk, vv in gt_output_dict.items():
            gt_output_dict[kk] = vv[:, : batch["length"][0].item(), ...]

        pred_keypoints3d = pred_output_dict["keypoints3d"]
        pred_root_rotations_mat = pred_output_dict["root_rotations_mat"]
        gt_keypoints3d = gt_output_dict["keypoints3d"]
        gt_root_rotations_mat = gt_output_dict["root_rotations_mat"]
        pred_transl = pred_output_dict["transl"]
        gt_transl = gt_output_dict["transl"]

        with torch.autocast("cuda", enabled=False):
            jerk_maxJ_meanBT = calculate_jerk(pred_keypoints3d, batch["length"])
            mpjpe_per, mpjpe_best = calculate_motion_similarity(
                pred_keypoints3d,
                gt_keypoints3d,
                use_rifke=True,
                root_rotations_mat_1=pred_root_rotations_mat,
                root_rotations_mat_2=gt_root_rotations_mat,
            )
            diversity = calculate_motion_diversity(
                pred_keypoints3d,
                use_rifke=True,
                root_rotations_mat=pred_root_rotations_mat,
            )
            translation_error = calculate_translation_error(pred_transl, gt_transl, batch["length"])
        model_output = dict(
            latent_denorm=torch.cat(
                [
                    gt_output_dict["latent_denorm"][0:1],
                    pred_output_dict["latent_denorm"],
                ],
                dim=0,
            ).cpu(),
            keypoints3d=torch.cat([gt_keypoints3d[0:1], pred_keypoints3d], dim=0),
            rot6d=torch.cat([gt_output_dict["rot6d"][0:1], pred_output_dict["rot6d"]], dim=0).cpu(),
            transl=torch.cat([gt_output_dict["transl"][0:1], pred_output_dict["transl"]], dim=0).cpu(),
        )
        return {
            "metrics": {
                "mpjpe_per": mpjpe_per,
                "mpjpe_best": mpjpe_best,
                "diversity": diversity,
                "jerk_max": jerk_maxJ_meanBT,
                "translation_err": translation_error,
            },
            "model_output": model_output,
        }

    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        seed_input: List[int],
        duration_slider: int,
        cfg_scale: Optional[float] = None,
        use_special_game_feat: bool = False,
        hidden_state_dict=None,
        debug=False,
        length=None,
    ) -> Dict[str, Any]:
        device = get_module_device(self)
        if length is None:
            length = int(round(duration_slider * self.output_mesh_fps))
        assert (
            0 < length < 5000
        ), f"input duration_slider must be in (0, {5000/self.output_mesh_fps}] due to rope, but got {duration_slider}"
        if length > self.train_frames or length < min(self.train_frames, 20):
            print(f">>> given length is too long or too short, got {length}, will be truncated")
            length = min(length, self.train_frames)
            length = max(length, min(self.train_frames, 20))
        if self.motion_type.startswith("vae"):
            length = int(round(length / self.vae.network.downsample))

        x = self.noise_from_seeds(
            torch.zeros(
                1,
                length,
                self._network_module_args["input_dim"],
                device=device,
            ),
            seed_input,
        )

        self.infer_noise_scheduler.set_timesteps(self.validation_steps)
        timesteps = self.infer_noise_scheduler.timesteps.to(device)

        repeat = len(seed_input)
        if isinstance(text, list):
            assert len(text) == repeat, f"len(text) must equal len(seed_input), got {len(text)} vs {repeat}"
            text_list = text
        elif isinstance(text, str):
            text_list = [text] * repeat
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")
        if not self.uncondition_mode:
            if hidden_state_dict is None:
                hidden_state_dict = self.encode_text({"text": text_list})
            vtxt_input = hidden_state_dict["text_vec_raw"]
            ctxt_input = hidden_state_dict["text_ctxt_raw"]
            ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            sources = None if not use_special_game_feat else ["Game"] * repeat
            vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
                vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
            )
        else:
            vtxt_input = self.null_vtxt_feat.expand(repeat, 1, -1)
            ctxt_input = self.null_ctxt_input.expand(repeat, 1, -1)
            ctxt_length = torch.tensor([1]).expand(repeat)
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(repeat, -1)

        assert len(vtxt_input.shape) == 3, f"vtxt_input.shape: {vtxt_input.shape}, should be (B, 1, D)"
        assert len(ctxt_input.shape) == 3, f"ctxt_input.shape: {ctxt_input.shape}, should be (B, 1, D)"
        assert len(ctxt_length.shape) == 1, f"ctxt_length.shape: {ctxt_length.shape}, should be (B,)"
        x_length = torch.LongTensor([length] * repeat).to(device)
        x_mask_temporal = length_to_mask(x_length, length)

        x = self._denoise_loop(
            x=x,
            x_mask_temporal=x_mask_temporal,
            vtxt_input=vtxt_input,
            ctxt_input=ctxt_input,
            ctxt_mask_temporal=ctxt_mask_temporal,
            timesteps=timesteps,
            cfg_scale=cfg_scale,
        )
        x = x[:, :length, ...].clone()
        output_dict = self.decode_motion_from_latent(x, should_apply_smooothing=True)
        return {
            **output_dict,
            "text": text,
        }

    def _denoise_loop(
        self,
        x: Tensor,
        x_mask_temporal: Tensor,
        vtxt_input: Tensor,
        ctxt_input: Tensor,
        ctxt_mask_temporal: Tensor,
        timesteps: Tensor,
        cfg_scale: Optional[float] = None,
    ) -> Tensor:
        text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
        do_classifier_free_guidance = text_guidance_scale > 1.0
        if do_classifier_free_guidance is True:
            # TODO: add negetive text here later
            silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)

            if self.enable_ctxt_null_feat:
                silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
            else:
                silent_ctxt_input = ctxt_input
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)

            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        for i, t in enumerate(timesteps):
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.motion_transformer(
                x=x_input,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)
            x = self.infer_noise_scheduler.step(x_pred, t, x).prev_sample
        return x


class MotionFlowMatching(MotionGeneration):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        text_encoder_module: str,
        text_encoder_cfg: dict,
        noise_scheduler_cfg: dict = {"method": "euler"},
        infer_noise_scheduler_cfg: dict = {"validation_steps": 50},
        mean_std_dir: Optional[str] = None,
        losses_cfg: Optional[dict] = None,
        train_cfg: Optional[dict] = None,
        test_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(
            network_module=network_module,
            network_module_args=network_module_args,
            text_encoder_module=text_encoder_module,
            text_encoder_cfg=text_encoder_cfg,
            losses_cfg=losses_cfg,
            mean_std_dir=(mean_std_dir if mean_std_dir is not None else (test_cfg or {})["mean_std_dir"]),
            **kwargs,
        )
        # build scheduler
        self._noise_scheduler_cfg = deepcopy(noise_scheduler_cfg)
        self._infer_noise_scheduler_cfg = deepcopy(infer_noise_scheduler_cfg)
        # additional cfg
        self.pred_type = kwargs.get("pred_type", "velocity")
        self.train_cfg = deepcopy(train_cfg) if train_cfg else dict()
        self.test_cfg = deepcopy(test_cfg) if test_cfg else dict()
        self._parse_train_cfg()
        self._parse_test_cfg()

    def _parse_train_cfg(self) -> None:
        self.cond_mask_prob = self.train_cfg.get("cond_mask_prob", 0.0)

    def _parse_test_cfg(self) -> None:
        self.validation_steps = self._infer_noise_scheduler_cfg["validation_steps"]
        self.text_guidance_scale = self.test_cfg.get("text_guidance_scale", 1)

    def forward(
        self,
        noise: Tensor,
        noise_length: Tensor,
        hidden_state_dict: Dict[str, Tensor],
        t: Tensor,
        cfg_scale: Optional[float] = None,
    ):
        # NOTE: this is for single forward without infer_noise_scheduler.step (for reinforcement learning)
        # noise: (B, train_frames, D)
        # noise_length: (B, 1)
        # hidden_state: (B, max_text_len, D)
        # hidden_state_length: (B, 1)
        # t: (B, 1)
        vtxt_input = hidden_state_dict["text_vec_raw"]
        ctxt_input = hidden_state_dict["text_ctxt_raw"]
        ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_mask_temporal = length_to_mask(noise_length, noise.shape[1])

        text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
        do_classifier_free_guidance = text_guidance_scale > 1.0
        if do_classifier_free_guidance is True:
            silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)

            silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)

            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        x_input = torch.cat([noise] * 2, dim=0) if do_classifier_free_guidance else noise
        t = torch.cat([t] * 2, dim=0) if do_classifier_free_guidance else t
        x_pred = self.motion_transformer(
            x=x_input,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=t,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )
        if do_classifier_free_guidance:
            x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
            x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)

        return x_pred

    def forward_in_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        device = get_module_device(self)

        # allow externally provided timesteps (e.g., for fair eval); fallback to random
        fixed_timesteps = batch.get("fixed_train_timesteps", None)

        if self.motion_type.startswith("vae"):
            length = batch["length"]
            length_for_network = length / self.vae.network.downsample
            batch["motion_start_frame"] = (batch["motion_start_frame"] / self.vae.network.downsample).long()
            batch["motion_end_frame"] = (batch["motion_end_frame"] / self.vae.network.downsample).long()
            gt_motion = self.vae.encode(batch).mean
        else:
            gt_motion = batch["motion"]
            length = batch["length"]
            length_for_network = length

        if "text_vec_raw" in batch.keys():
            vtxt_input = batch["text_vec_raw"]
            ctxt_input = batch["text_ctxt_raw"]
            ctxt_length = batch["text_ctxt_raw_length"]
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
            vtxt_input, ctxt_input = self._mask_cond(
                vtxt_input,
                ctxt_input,
                force_mask=False,
                cond_mask_prob=self.cond_mask_prob,
            )
            sources = (batch.get("data_meta", {}) or {}).get("data_src", None)
            vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
                vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
            )
        else:
            vtxt_input = self.null_vtxt_feat.expand(gt_motion.shape[0], 1, -1)
            ctxt_input = self.null_ctxt_input.expand(gt_motion.shape[0], 1, -1)
            ctxt_length = torch.tensor([1]).expand(gt_motion.shape[0])
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(gt_motion.shape[0], -1)

        # x0 is gaussian noise
        x0 = torch.randn(gt_motion.shape, dtype=gt_motion.dtype).to(device)
        x1 = gt_motion
        # Sample a random timestep for each image
        # time step
        # 1000 与MDM兼容
        if fixed_timesteps is not None:
            timesteps = fixed_timesteps.to(device=device, dtype=gt_motion.dtype).view(-1)
        elif "timestep_sample_method" in self.train_cfg:
            if self.train_cfg["timestep_sample_method"] == "logit_normal":
                timesteps = (
                    torch.randn(gt_motion.shape[0], dtype=gt_motion.dtype) * self.train_cfg["t_sample_P_std"]
                    + self.train_cfg["t_sample_P_mean"]
                ).to(device)
                timesteps = torch.sigmoid(timesteps)
            else:
                raise NotImplementedError(
                    f"Unsupported timestep sample method: {self.train_cfg['timestep_sample_method']}"
                )
        else:
            timesteps = torch.rand((gt_motion.shape[0],), dtype=gt_motion.dtype).to(device)
        # sample xt (phi_t(x) in the paper)
        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        if "motion_start_frame" in batch.keys() and "motion_end_frame" in batch.keys():
            x_mask_temporal = start_end_frame_to_mask(
                batch["motion_start_frame"],
                batch["motion_end_frame"],
                gt_motion.shape[1],
            )
        else:
            x_mask_temporal = length_to_mask(length_for_network, gt_motion.shape[1])

        pred = self.motion_transformer(
            x=phi,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=ctxt_mask_temporal,
        )

        if self.pred_type == "velocity":
            pass
        elif self.pred_type == "x1":
            # predict the original x1
            # https://github.com/LTH14/JiT/blob/main/denoiser.py#L56
            t_eps = 0.05
            flow = (x1 - phi) / (1 - t).clamp_min(t_eps)
            pred = (pred - phi) / (1 - t).clamp_min(t_eps)
        else:
            raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

        with torch.autocast("cuda", enabled=False):
            pred_fp32 = pred.float()
            flow_fp32 = flow.float()
            mask_fp32 = x_mask_temporal.float()
            losses = self.compute_loss(
                pred_fp32, flow_fp32, data_mask_temporal=mask_fp32, has_finger=batch.get("has_finger", None)
            )
        per_sample_loss = losses.pop("per_sample_loss", None)
        return {
            "latent": gt_motion,
            "model_output": pred,
            "loss": losses["loss"],
            "loss_dict": losses,
            "tensor_results": {
                "per_sample_loss": per_sample_loss,
                "index": batch.get("index", None),
            },
        }

    @torch.no_grad()
    def validate(self, batch: Dict[str, Any], seeds: List[int] = [0, 1, 2, 3]) -> Dict[str, Any]:
        device = get_module_device(self)
        if self.motion_type.startswith("vae"):
            length = batch["length"]
            length_for_network = length / self.vae.network.downsample
            batch["motion_start_frame"] = (batch["motion_start_frame"] / self.vae.network.downsample).long()
            batch["motion_end_frame"] = (batch["motion_end_frame"] / self.vae.network.downsample).long()
            gt_motion = self.vae.encode(batch).mean
        else:
            length = batch["length"]
            length_for_network = length
            gt_motion = batch["motion"]

        if "text_vec_raw" in batch.keys():
            vtxt_input = batch["text_vec_raw"]
            ctxt_input = batch["text_ctxt_raw"]
            ctxt_length = batch["text_ctxt_raw_length"]
            if len(ctxt_length.shape) == 2:
                ctxt_length = ctxt_length.squeeze(0)
        else:
            vtxt_input = self.null_vtxt_feat
            ctxt_input = self.null_ctxt_input
            ctxt_length = torch.tensor([1])

        repeat = len(seeds)
        vtxt_input = vtxt_input.repeat((repeat,) + (1,) * (vtxt_input.dim() - 1))
        ctxt_input = ctxt_input.repeat((repeat,) + (1,) * (ctxt_input.dim() - 1))
        ctxt_length = ctxt_length.repeat(repeat)
        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        sources = (batch.get("data_meta", {}) or {}).get("data_src", None)
        if sources is not None:
            sources = sources * repeat
        vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
            vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
        )
        if "motion_start_frame" in batch.keys() and "motion_end_frame" in batch.keys():
            x_mask_temporal = start_end_frame_to_mask(
                batch["motion_start_frame"],
                batch["motion_end_frame"],
                gt_motion.shape[1],
            )
        else:
            x_mask_temporal = length_to_mask(length_for_network, gt_motion.shape[1])
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))

        do_classifier_free_guidance = self.text_guidance_scale > 1.0
        if do_classifier_free_guidance is True:
            silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)

            if self.enable_ctxt_null_feat:
                silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
            else:
                silent_ctxt_input = ctxt_input
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)

            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)

        def fn(t: Tensor, x: Tensor) -> Tensor:
            # predict flow
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.motion_transformer(
                x=x_input,
                ctxt_input=ctxt_input,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=ctxt_mask_temporal,
            )
            if self.pred_type == "velocity":
                pass
            elif self.pred_type == "x1":
                t_eps = 0.05
                x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
            else:
                raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + self.text_guidance_scale * (x_pred_text - x_pred_basic)
            return x_pred

        dtype = gt_motion.dtype
        y0 = self.noise_from_seeds(gt_motion, seeds)
        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        with torch.no_grad():
            trajectory = odeint(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]

        if self.motion_type.startswith("vae"):
            decode_pred = self.vae.decode(sampled)
            fk_pred = self.vae.forward_body(decode_pred)["params_decode"]
            pred_output_dict = dict(
                latent_denorm=sampled,
                root_rotations_mat=fk_pred["rotations"][:, :, 0],
                keypoints3d=fk_pred["keypoints3d"],
                rot6d=fk_pred["rot6d"],
                transl=fk_pred["trans"],
                shapes=fk_pred["shapes"],
            )
            gt_encode = self.vae.encode_motion(batch)
            gt_decode = self.vae.decode_motion(gt_encode)
            fk_gt = self.vae.forward_body(gt_decode)["params_decode"]
            gt_output_dict = dict(
                latent_denorm=gt_motion,
                root_rotations_mat=fk_gt["rotations"][:, :, 0],
                keypoints3d=fk_gt["keypoints3d"],
                rot6d=fk_gt["rot6d"],
                transl=fk_gt["trans"],
                shapes=fk_gt["shapes"],
            )
        else:
            with torch.autocast("cuda", enabled=False):
                pred_output_dict = self.decode_motion_from_latent(sampled, should_apply_smooothing=False)
                gt_output_dict = self.decode_motion_from_latent(gt_motion, should_apply_smooothing=False)

        for kk, vv in pred_output_dict.items():
            pred_output_dict[kk] = vv[:, : batch["length"][0].item(), ...]
        for kk, vv in gt_output_dict.items():
            gt_output_dict[kk] = vv[:, : batch["length"][0].item(), ...]

        pred_keypoints3d = pred_output_dict["keypoints3d"]
        pred_root_rotations_mat = pred_output_dict["root_rotations_mat"]
        gt_keypoints3d = gt_output_dict["keypoints3d"]
        gt_root_rotations_mat = gt_output_dict["root_rotations_mat"]
        pred_transl = pred_output_dict["transl"]
        gt_transl = gt_output_dict["transl"]

        with torch.autocast("cuda", enabled=False):
            jerk_maxJ_meanBT = calculate_jerk(pred_keypoints3d, batch["length"])
            mpjpe_per, mpjpe_best = calculate_motion_similarity(
                pred_keypoints3d,
                gt_keypoints3d,
                use_rifke=True,
                root_rotations_mat_1=pred_root_rotations_mat,
                root_rotations_mat_2=gt_root_rotations_mat,
            )
            diversity = calculate_motion_diversity(
                pred_keypoints3d,
                use_rifke=True,
                root_rotations_mat=pred_root_rotations_mat,
            )
            translation_error = calculate_translation_error(pred_transl, gt_transl, batch["length"])
        vel = calculate_velocity_o6d(
            pred_output_dict["rot6d"],  # [B, T, J, 6]
            pred_output_dict["transl"],  # [B, T, 3]
            self.output_mesh_fps,
        )
        model_output = dict(
            latent_denorm=torch.cat(
                [
                    gt_output_dict["latent_denorm"][0:1],
                    pred_output_dict["latent_denorm"],
                ],
                dim=0,
            ).cpu(),
            keypoints3d=torch.cat([gt_keypoints3d[0:1], pred_keypoints3d], dim=0),
            rot6d=torch.cat([gt_output_dict["rot6d"][0:1], pred_output_dict["rot6d"]], dim=0).cpu(),
            transl=torch.cat([gt_output_dict["transl"][0:1], pred_output_dict["transl"]], dim=0).cpu(),
        )
        return {
            "metrics": {
                "jerk_max": jerk_maxJ_meanBT,
                "mpjpe_per": mpjpe_per,
                "mpjpe_best": mpjpe_best,
                "diversity": diversity,
                "translation_err": translation_error,
                **vel,
            },
            "model_output": model_output,
        }

    def prepare_latents(self, batch_size, noise_length, device=None):
        return torch.randn(batch_size, noise_length, self._network_module_args["input_dim"], device=device)

    @torch.no_grad()
    def generate(
        self,
        text: Union[str, List[str]],
        seed_input: List[int],
        duration_slider: int,
        cfg_scale: Optional[float] = None,
        debug: bool = False,
        use_special_game_feat: bool = False,
        hidden_state_dict=None,
        length=None,
    ) -> Dict[str, Any]:
        device = get_module_device(self)
        if length is None:
            length = int(round(duration_slider * self.output_mesh_fps))
        assert (
            0 < length < 5000
        ), f"input duration_slider must be in (0, {5000/self.output_mesh_fps}] due to rope, but got {duration_slider}"
        if length > self.train_frames or length < min(self.train_frames, 20):
            print(f">>> given length is too long or too short, got {length}, will be truncated")
            length = min(length, self.train_frames)
            length = max(length, min(self.train_frames, 20))
        if self.motion_type.startswith("vae"):
            if isinstance(length, int):
                length = int(round(length / self.vae.network.downsample))
            else:
                length = length / self.vae.network.downsample

        repeat = len(seed_input)
        if isinstance(text, list):
            assert len(text) == repeat, f"len(text) must equal len(seed_input), got {len(text)} vs {repeat}"
            text_list = text
        elif isinstance(text, str):
            text_list = [text] * repeat
        else:
            raise TypeError(f"Unsupported text type: {type(text)}")

        if not self.uncondition_mode:
            if hidden_state_dict is None:
                hidden_state_dict = self.encode_text({"text": text_list})
            vtxt_input = hidden_state_dict["text_vec_raw"]
            ctxt_input = hidden_state_dict["text_ctxt_raw"]
            ctxt_length = hidden_state_dict["text_ctxt_raw_length"]
            # 检查shape
            if len(vtxt_input.shape) == 2 and len(ctxt_input.shape) == 2:
                vtxt_input = vtxt_input[None].repeat(repeat, 1, 1)
                ctxt_input = ctxt_input[None].repeat(repeat, 1, 1)
                ctxt_length = ctxt_length.repeat(repeat)
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])

            sources = None if not use_special_game_feat else ["Game"] * repeat
            vtxt_input, ctxt_input, ctxt_mask_temporal = self._maybe_inject_source_token(
                vtxt_input, ctxt_input, ctxt_mask_temporal, sources, trigger_sources={"Taobao", "Game"}
            )
        else:
            vtxt_input = self.null_vtxt_feat.expand(repeat, 1, -1)
            ctxt_input = self.null_ctxt_input.expand(repeat, 1, -1)
            ctxt_length = torch.tensor([1]).expand(repeat)
            ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1]).expand(repeat, -1)
        assert len(vtxt_input.shape) == 3, f"vtxt_input.shape: {vtxt_input.shape}, should be (B, 1, D)"
        assert len(ctxt_input.shape) == 3, f"ctxt_input.shape: {ctxt_input.shape}, should be (B, 1, D)"
        assert len(ctxt_length.shape) == 1, f"ctxt_length.shape: {ctxt_length.shape}, should be (B,)"

        ctxt_mask_temporal = length_to_mask(ctxt_length, ctxt_input.shape[1])
        x_length = torch.LongTensor([length] * repeat).to(device)
        x_mask_temporal = length_to_mask(x_length, self.train_frames)

        text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
        do_classifier_free_guidance = text_guidance_scale > 1.0 and not self.uncondition_mode
        if do_classifier_free_guidance is True:
            silent_text_feat = self.null_vtxt_feat.expand(*vtxt_input.shape)
            vtxt_input = torch.cat([silent_text_feat, vtxt_input], dim=0)

            if self.enable_ctxt_null_feat:
                silent_ctxt_input = self.null_ctxt_input.expand(*ctxt_input.shape)
            else:
                silent_ctxt_input = ctxt_input
            ctxt_input = torch.cat([silent_ctxt_input, ctxt_input], dim=0)

            ctxt_mask_temporal = torch.cat([ctxt_mask_temporal] * 2, dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        def fn(t: Tensor, x: Tensor) -> Tensor:
            # predict flow
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            if not debug:
                x_pred = self.motion_transformer(
                    x=x_input,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t.expand(x_input.shape[0]),
                    x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )
            else:
                from ..utils.visualize_analysis import plot_attn_list

                x_pred, attn_list, motion_len, text_len = self.motion_transformer.forward_with_attn(
                    x=x_input,
                    ctxt_input=ctxt_input,
                    vtxt_input=vtxt_input,
                    timesteps=t.expand(x_input.shape[0]),
                    x_mask_temporal=x_mask_temporal,
                    ctxt_mask_temporal=ctxt_mask_temporal,
                )
                plot_attn_list(
                    attn_list,
                    length,
                    motion_len,
                    ctxt_length[0],
                    save_dir=f"output/attn/attn_t{t.item()}",
                    prefix="mmdit",
                )
            if self.pred_type == "velocity":
                pass
            elif self.pred_type == "x1":
                t_eps = 0.05
                x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
            else:
                raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + text_guidance_scale * (x_pred_text - x_pred_basic)
            return x_pred

        # duplicate test corner for inner time step oberservation
        t = torch.linspace(0, 1, self.validation_steps + 1, device=device)
        y0 = self.noise_from_seeds(
            torch.zeros(
                1,
                self.train_frames,
                self._network_module_args["input_dim"],
                device=device,
            ),
            seed_input,
        )
        with torch.no_grad():
            trajectory = odeint(fn, y0, t, **self._noise_scheduler_cfg)
        sampled = trajectory[-1]
        assert isinstance(sampled, Tensor), f"sampled must be a Tensor, but got {type(sampled)}"
        sampled = sampled[:, :length, ...].clone()

        output_dict = self.decode_motion_from_latent(sampled, should_apply_smooothing=True)

        return {
            **output_dict,
            "text": text,
        }


if __name__ == "__main__":
    # python -m hymotion.pipeline.motion_diffusion
    import time

    import torch

    device = "cuda:0"
    bsz, input_dim = 64, 272
    seq_lens = [90, 180, 360]
    ctxt_seq_lens = 64
    warmup = 5
    repeats = 100

    def make_batch(seq_len: int, device: str = "cuda:0"):
        return dict(
            motion=torch.randn(bsz, seq_len, input_dim, device=device),
            text_ctxt_raw=torch.randn(bsz, ctxt_seq_lens, 4096, device=device),
            text_vec_raw=torch.randn(bsz, 1, 768, device=device),
            text_ctxt_raw_length=torch.LongTensor([ctxt_seq_lens] * bsz).to(device),
            length=torch.LongTensor([seq_len] * bsz).to(device),
        )

    def benchmark_train_step(
        model,
        optimizer,
        batch,
        name: str,
        warmup: int = 3,
        repeats: int = 20,
        step: bool = True,
    ):
        model.train()
        torch.cuda.synchronize()
        for _ in range(warmup):
            optimizer.zero_grad(set_to_none=True)
            out = model.forward_in_training(batch)
            out["loss"].backward()
            if step:
                optimizer.step()
        torch.cuda.synchronize()

        times = []
        for _ in range(repeats):
            optimizer.zero_grad(set_to_none=True)
            t0 = time.perf_counter()
            out = model.forward_in_training(batch)
            out["loss"].backward()
            if step:
                optimizer.step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg = sum(times) / len(times)
        p50 = sorted(times)[len(times) // 2]
        frames = batch["motion"].shape[1] * bsz
        kind = "fwd+bwd+step" if step else "fwd+bwd"
        print(
            f"[{name} | train_{kind}] seq_len={batch['motion'].shape[1]} | "
            f"avg={avg*1000:.1f} ms/iter | p50={p50*1000:.1f} ms | "
            f"throughput={frames/avg:.1f} frames/s"
        )

    network_module = "hymotion/network/hymotion_mmdit.HunyuanMotionMMDiT"
    network_module_args = {
        "input_dim": input_dim,
        "feat_dim": 512,
        "ctxt_input_dim": 4096,
        "vtxt_input_dim": 768,
        "num_layers": 12,
        "num_heads": 4,
        "mlp_ratio": 2.0,
        "dropout": 0.0,
        "mask_mode": "narrowband",
    }
    text_encoder_module = "hymotion/network/text_encoders/text_encoder.HYTextModel"
    text_encoder_cfg = {"llm_type": "qwen3_embedding", "max_length_llm": ctxt_seq_lens}
    mean_std_dir = "/apdcephfs_cq10/share_1467498/datasets/motion_data/HunyuanMotion/_stats/v20250804_h_o6dp/"

    # # ================================ DDPM_MMDiT ================================
    # DDPM_MMDiT = MotionDiffusion(
    #     network_module=network_module,
    #     network_module_args=network_module_args,
    #     text_encoder_module=text_encoder_module,
    #     text_encoder_cfg=text_encoder_cfg,
    #     noise_scheduler_module="diffusers.DDPMScheduler",
    #     noise_scheduler_cfg={
    #         "num_train_timesteps": 1000,
    #         "beta_start": 0.00085,
    #         "beta_end": 0.012,
    #         "beta_schedule": "squaredcos_cap_v2",
    #         "clip_sample": False,
    #         "variance_type": "fixed_small",
    #         "prediction_type": "sample",
    #     },
    #     infer_noise_scheduler_module="diffusers.DDIMScheduler",
    #     infer_noise_scheduler_cfg={
    #         "num_train_timesteps": 1000,
    #         "beta_start": 0.00085,
    #         "beta_end": 0.012,
    #         "beta_schedule": "squaredcos_cap_v2",
    #         "clip_sample": False,
    #         "set_alpha_to_one": False,
    #         "steps_offset": 1,
    #         "prediction_type": "sample",
    #     },
    #     train_cfg={"cond_mask_prob": 0.1},
    #     test_cfg={
    #         "num_inference_timesteps": 50,
    #         "text_guidance_scale": 1.5,
    #         "mean_std_dir": mean_std_dir,
    #     },
    # ).to(device)
    # ================================ FM_MMDiT ================================
    FM_MMDiT = MotionFlowMatching(
        network_module=network_module,
        network_module_args=network_module_args,
        text_encoder_module=text_encoder_module,
        text_encoder_cfg=text_encoder_cfg,
        noise_scheduler_module={"method": "euler"},
        infer_noise_scheduler_cfg={"validation_steps": 50},
        train_cfg={"cond_mask_prob": 0.1},
        test_cfg={
            "text_guidance_scale": 1.5,
            "mean_std_dir": mean_std_dir,
        },
    ).to(device)

    optimizer = torch.optim.AdamW(FM_MMDiT.parameters(), lr=1e-4)
    # 运行基准测试（只测试 forward_in_training）
    for L in seq_lens:
        batch = make_batch(L, device=device)
        benchmark_train_step(
            FM_MMDiT,
            optimizer,
            batch,
            name="FM_MMDiT",
            warmup=warmup,
            repeats=repeats,
            step=True,
        )
