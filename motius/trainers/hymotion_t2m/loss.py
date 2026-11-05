from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import deque


class M2MLoss(nn.Module):
    _MODALITY_MEAN_REDUCTIONS = ("component_mean", "modality_mean")
    _VALID_VELOCITY_LOSS_REDUCTIONS = (
        "element_mean",
        "official_element_mean",
        "component_mean",
        "modality_mean",
    )

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        velocity_weight: float = 1.0,
        x1_weight: float = 1.0,
        keypoints3d_weight: float = 1.0,
        translation_weight: float = 1.0,
        motion_smoothness_weight: float = 0.0,
        fk_loss_start_step: int = 0,
        trans_dim_weight: float = 1.0,
        trans_dims: int = 3,
        velocity_loss_reduction: str = "element_mean",
        fk_consistency_weight: float = 0.0,
        fk_consistency_warmup_steps: int = 1000,
        foot_contact_weight: float = 0.0,
        foot_contact_warmup_steps: int = 0,
        spike_downweight_enabled: bool = True,
        spike_downweight_factor: float = 0.3,
        spike_detection_std_threshold: float = 2.0,
        spike_detection_window: int = 100,
    ):
        super().__init__()
        self.velocity_weight = velocity_weight
        self.x1_weight = x1_weight
        self.keypoints3d_weight = keypoints3d_weight
        self.translation_weight = translation_weight
        self.motion_smoothness_weight = motion_smoothness_weight
        self.fk_loss_start_step = fk_loss_start_step
        self.trans_dim_weight = trans_dim_weight
        self.trans_dims = trans_dims
        self.velocity_loss_reduction = velocity_loss_reduction
        self.fk_consistency_weight = fk_consistency_weight
        self.fk_consistency_warmup_steps = fk_consistency_warmup_steps
        self.foot_contact_weight = foot_contact_weight
        self.foot_contact_warmup_steps = foot_contact_warmup_steps

        # Spike detection parameters (Fix 2, P0)
        self.spike_downweight_enabled = spike_downweight_enabled
        self.spike_downweight_factor = spike_downweight_factor
        self.spike_detection_std_threshold = spike_detection_std_threshold
        self.spike_detection_window = spike_detection_window

        # Rolling statistics for spike detection
        self._trans_loss_history = deque(maxlen=spike_detection_window)
        self._baseline_trans_loss = 0.0
        self._trans_loss_std = 0.0

        if velocity_loss_reduction not in self._VALID_VELOCITY_LOSS_REDUCTIONS:
            raise ValueError(
                "velocity_loss_reduction must be one of "
                f"{self._VALID_VELOCITY_LOSS_REDUCTIONS}, "
                f"got {velocity_loss_reduction!r}"
            )

        if loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type == "mse":
            self.loss_fn = F.mse_loss
        elif loss_type == "l2":
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @staticmethod
    def _motion_components(dim: int):
        if dim >= 198:
            # 198-dim SMPL layout:
            #   trans(0:3), root rot6d(3:9), body rot6d(9:135),
            #   joint positions(135:198).
            return ((0, 3), (3, 9), (9, 135), (135, 198))
        if dim >= 135:
            return ((0, 3), (3, 9), (9, 135))
        if dim == 38:
            # G1-native layout: transl(0:3) + pelvis rot6d(3:9) + 29 joint angles(9:38).
            # Splitting here lets modality/component mean give translation and root
            # rotation their own mean instead of being swamped by the 29 dof.
            return ((0, 3), (3, 9), (9, 38))
        return ((0, dim),)

    @staticmethod
    def _component_names(dim: int):
        """Log-friendly names aligned with :meth:`_motion_components`."""
        if dim >= 198:
            return ('trans', 'root_rot', 'body_rot', 'joint_pos')
        if dim >= 135:
            return ('trans', 'root_rot', 'body_rot')
        if dim == 38:
            return ('trans', 'root_rot', 'joint')
        return ('all',)

    def _uses_modality_mean(self) -> bool:
        return self.velocity_loss_reduction in self._MODALITY_MEAN_REDUCTIONS

    def _update_spike_detection_stats(self, trans_loss_magnitude: float):
        """Update rolling statistics for spike detection.

        Args:
            trans_loss_magnitude: Combined magnitude of loss_velocity_trans + loss_x1_trans
        """
        if not self.spike_downweight_enabled:
            return

        self._trans_loss_history.append(trans_loss_magnitude)

        if len(self._trans_loss_history) >= 10:
            losses = list(self._trans_loss_history)
            self._baseline_trans_loss = sum(losses) / len(losses)
            var = sum((x - self._baseline_trans_loss) ** 2 for x in losses) / len(losses)
            self._trans_loss_std = var ** 0.5 if var > 0 else 1e-6

    def _detect_trans_spike(self, trans_loss_magnitude: float) -> float:
        """Detect if current translation loss is a spike and return downweight factor.

        Args:
            trans_loss_magnitude: Combined magnitude of loss_velocity_trans + loss_x1_trans

        Returns:
            Downweight factor (1.0 if no spike, 0.3 if spike detected)
        """
        if not self.spike_downweight_enabled or len(self._trans_loss_history) < 10:
            return 1.0

        threshold = self._baseline_trans_loss + self._trans_loss_std * self.spike_detection_std_threshold

        if trans_loss_magnitude > threshold:
            return self.spike_downweight_factor

        return 1.0

    def _masked_motion_loss(
        self,
        per_dim: Tensor,
        data_mask_temporal: Tensor,
        generation_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Reduce (B, L, D) losses with optional modality-wise means."""
        data_mask = data_mask_temporal.to(per_dim.device).to(per_dim.dtype)

        if self.velocity_loss_reduction == "element_mean":
            if generation_mask is not None:
                gen_mask = generation_mask.to(per_dim.device).to(per_dim.dtype)
                combined = gen_mask * data_mask.unsqueeze(-1)
                mask_sum = torch.clamp(combined.sum(), min=1.0)
                return (per_dim * combined).sum() / mask_sum

            per_frame = per_dim.mean(dim=-1)
            mask_sum = torch.clamp(data_mask.sum(), min=1.0)
            return (per_frame * data_mask).sum() / mask_sum

        if self.velocity_loss_reduction == "official_element_mean":
            combined = data_mask.unsqueeze(-1).expand_as(per_dim)
            if generation_mask is not None:
                combined = combined * generation_mask.to(per_dim.device).to(per_dim.dtype)
            mask_sum = torch.clamp(combined.sum(), min=1.0)
            return (per_dim * combined).sum() / mask_sum

        # Modality-wise semantic reduction: each representation component first
        # gets its own valid-cell mean, then active components are averaged.
        # This prevents wide modalities (e.g. body rot6d / joint positions) from
        # swallowing small but important ones such as translation/root. When a
        # generation mask exposes only a subset of channels, e.g. x/z position
        # without y, the mean is still taken only over those active cells inside
        # that modality.
        comp_losses = []
        for start, end in self._motion_components(per_dim.shape[-1]):
            comp = per_dim[..., start:end]
            if generation_mask is not None:
                comp_mask = (
                    generation_mask[..., start:end]
                    .to(per_dim.device)
                    .to(per_dim.dtype)
                    * data_mask.unsqueeze(-1)
                )
            else:
                comp_mask = data_mask.unsqueeze(-1).expand_as(comp)
            denom = comp_mask.sum()
            if torch.gt(denom.detach(), 0):
                comp_losses.append((comp * comp_mask).sum() / denom)

        if not comp_losses:
            return per_dim.sum() * 0.0
        return torch.stack(comp_losses).mean()

    _COMP_NAMES = ('trans', 'root_rot', 'body_rot', 'joint_pos')

    def _masked_motion_loss_with_components(
        self,
        per_dim: Tensor,
        data_mask_temporal: Tensor,
        generation_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Like _masked_motion_loss (modality mean path) but also returns
        per-modality scalars for logging.

        Returns:
            (combined_scalar, {"trans": ..., "root_rot": ..., ...})
        """
        data_mask = data_mask_temporal.to(per_dim.device).to(per_dim.dtype)
        comp_ranges = self._motion_components(per_dim.shape[-1])
        comp_names = self._component_names(per_dim.shape[-1])
        comp_dict: Dict[str, Tensor] = {}
        active = []
        for (start, end), name in zip(comp_ranges, comp_names):
            comp = per_dim[..., start:end]
            if generation_mask is not None:
                comp_mask = (
                    generation_mask[..., start:end]
                    .to(per_dim.device)
                    .to(per_dim.dtype)
                    * data_mask.unsqueeze(-1)
                )
            else:
                comp_mask = data_mask.unsqueeze(-1).expand_as(comp)
            denom = comp_mask.sum()
            if torch.gt(denom.detach(), 0):
                val = (comp * comp_mask).sum() / denom
                comp_dict[name] = val
                active.append(val)

        combined = torch.stack(active).mean() if active else per_dim.sum() * 0.0
        return combined, comp_dict

    def forward(
        self,
        pred_vel=None,
        gt_vel=None,
        pred_x1=None,
        gt_x1=None,
        pred_keypoints3d=None,
        gt_keypoints3d=None,
        pred_translation=None,
        gt_translation=None,
        global_step: Optional[int] = None,
        data_mask_temporal: Optional[Tensor] = None,
        generation_mask: Optional[Tensor] = None,
        fk_consistency_loss: Optional[Tensor] = None,
        pred_contact: Optional[Tensor] = None,
        gt_contact: Optional[Tensor] = None,
    ):
        """
        pred_vel: (B, L, D)
        gt_vel: (B, L, D)
        pred_x1: (B, L, D)
        gt_x1: (B, L, D)
        pred_keypoints3d: (B, L, J, 3)
        gt_keypoints3d: (B, L, J, 3)
        pred_translation: (B, L, 3)
        gt_translation: (B, L, 3)
        data_mask_temporal: (B, L) — padding mask (1=valid frame, 0=pad)
        generation_mask: (B, L, D) — optional, 1=generation region, 0=known.
            When provided, velocity/x1 losses are computed only on generation
            regions (mask-aware noise training).
        """
        loss_dict = {}
        assert data_mask_temporal is not None, "data_mask_temporal is required"

        if pred_vel is not None and gt_vel is not None:
            # velocity loss: (B, L, D) -> scalar
            # Apply per-dimension weighting: upweight translation dims (first trans_dims)
            # to compensate for the 3/135 dimension ratio imbalance
            vel_per_dim = self.loss_fn(pred_vel, gt_vel, reduction="none")  # (B, L, D)

            trans_vel_spike_weight = 1.0
            if self.spike_downweight_enabled:
                # This heuristic intentionally synchronizes to CPU for rolling
                # statistics. Keep the path fully disabled unless explicitly
                # enabled by config; official-aligned large runs do not use it.
                trans_vel_loss = vel_per_dim[:, :, :self.trans_dims].mean()
                trans_vel_loss_value = trans_vel_loss.item()
                trans_vel_spike_weight = self._detect_trans_spike(trans_vel_loss_value)
                self._update_spike_detection_stats(trans_vel_loss_value)

            if self.trans_dim_weight != 1.0:
                dim_weights = torch.ones(vel_per_dim.shape[-1], device=vel_per_dim.device)
                dim_weights[:self.trans_dims] = self.trans_dim_weight
                vel_per_dim = vel_per_dim * dim_weights

            # Apply spike downweighting to translation components
            if trans_vel_spike_weight < 1.0:
                vel_per_dim[:, :, :self.trans_dims] = vel_per_dim[:, :, :self.trans_dims] * trans_vel_spike_weight
            if self._uses_modality_mean():
                vel_loss, vel_comps = self._masked_motion_loss_with_components(
                    vel_per_dim, data_mask_temporal, generation_mask
                )
                loss_dict["velocity"] = self.velocity_weight * vel_loss
                for k, v in vel_comps.items():
                    loss_dict[f"velocity_{k}"] = v.detach()
            else:
                loss_dict["velocity"] = self.velocity_weight * self._masked_motion_loss(
                    vel_per_dim, data_mask_temporal, generation_mask
                )

        if pred_x1 is not None and gt_x1 is not None:
            # x1 loss: (B, L, D) -> scalar
            # Apply same per-dimension weighting as velocity loss
            x1_per_dim = self.loss_fn(pred_x1, gt_x1, reduction="none")  # (B, L, D)

            trans_x1_spike_weight = 1.0
            if self.spike_downweight_enabled:
                # See velocity branch above: avoid accidental per-step GPU
                # synchronization when spike downweighting is disabled.
                trans_x1_loss = x1_per_dim[:, :, :self.trans_dims].mean()
                trans_x1_spike_weight = self._detect_trans_spike(trans_x1_loss.item())

            if self.trans_dim_weight != 1.0:
                dim_weights = torch.ones(x1_per_dim.shape[-1], device=x1_per_dim.device)
                dim_weights[:self.trans_dims] = self.trans_dim_weight
                x1_per_dim = x1_per_dim * dim_weights

            # Apply spike downweighting to translation components
            if trans_x1_spike_weight < 1.0:
                x1_per_dim[:, :, :self.trans_dims] = x1_per_dim[:, :, :self.trans_dims] * trans_x1_spike_weight
            if self._uses_modality_mean():
                x1_loss, x1_comps = self._masked_motion_loss_with_components(
                    x1_per_dim, data_mask_temporal, generation_mask
                )
                loss_dict["x1"] = self.x1_weight * x1_loss
                for k, v in x1_comps.items():
                    loss_dict[f"x1_{k}"] = v.detach()
            else:
                loss_dict["x1"] = self.x1_weight * self._masked_motion_loss(
                    x1_per_dim, data_mask_temporal, generation_mask
                )

        if (global_step is None and self.fk_loss_start_step == 0) or (
            global_step is not None and global_step >= self.fk_loss_start_step
        ):
            if pred_keypoints3d is not None and gt_keypoints3d is not None:
                # 计算局部关键点（相对于根节点）
                local_keypoints3d = pred_keypoints3d[:, :, 1:22] - pred_keypoints3d[:, :, 0:1, :]
                local_keypoints3d_gt = gt_keypoints3d[:, :, 1:22] - gt_keypoints3d[:, :, 0:1, :]
                # keypoints3d loss: (B, L, 21, 3) -> (B, L, 21) -> (B, L) -> scalar
                loss_dict["keypoints3d"] = self.keypoints3d_weight * self.loss_fn(
                    local_keypoints3d, local_keypoints3d_gt, reduction="none"
                ).sum(dim=-1).mean(dim=-1)
                # 确保 data_mask_temporal 与 loss_dict["keypoints3d"] 在同一设备上
                data_mask_temporal_kp = data_mask_temporal.to(loss_dict["keypoints3d"].device)
                mask_sum_kp = torch.clamp(data_mask_temporal_kp.sum(), min=1.0)
                loss_dict["keypoints3d"] = (loss_dict["keypoints3d"] * data_mask_temporal_kp).sum() / mask_sum_kp

            if pred_translation is not None and gt_translation is not None and self.translation_weight > 0.0:
                # translation loss: (B, L, 3) -> (B, L) -> scalar
                loss_dict["translation"] = self.translation_weight * self.loss_fn(
                    pred_translation, gt_translation, reduction="none"
                ).mean(dim=-1)
                # 确保 data_mask_temporal 与 loss_dict["translation"] 在同一设备上
                data_mask_temporal_trans = data_mask_temporal.to(loss_dict["translation"].device)
                mask_sum_trans = torch.clamp(data_mask_temporal_trans.sum(), min=1.0)
                loss_dict["translation"] = (loss_dict["translation"] * data_mask_temporal_trans).sum() / mask_sum_trans
        elif global_step is None and self.fk_loss_start_step > 0:
            raise ValueError("global_step is None and fk_loss_start_step is not 0")

        # Motion smoothness loss: penalize deviation in frame-to-frame velocity
        # (temporal difference) between predicted and GT motion. Inspired by
        # KIMODO's velocity loss (γ_vel=2). This operates on the denoised x1
        # space, not the flow velocity.
        if self.motion_smoothness_weight > 0.0 and gt_x1 is not None and pred_x1 is not None:
            # Compute frame-to-frame velocity (temporal difference)
            pred_motion_vel = pred_x1[:, 1:] - pred_x1[:, :-1]  # (B, L-1, D)
            gt_motion_vel = gt_x1[:, 1:] - gt_x1[:, :-1]  # (B, L-1, D)
            smooth_per_dim = self.loss_fn(pred_motion_vel, gt_motion_vel, reduction="none")
            smooth_loss = smooth_per_dim.mean(dim=-1)  # (B, L-1)
            # Mask: both frame t and t+1 must be valid
            smooth_mask = data_mask_temporal[:, 1:] * data_mask_temporal[:, :-1]
            smooth_mask = smooth_mask.to(smooth_loss.device)
            mask_sum_smooth = torch.clamp(smooth_mask.sum(), min=1.0)
            loss_dict["smoothness"] = self.motion_smoothness_weight * (
                smooth_loss * smooth_mask
            ).sum() / mask_sum_smooth

        # FK consistency loss: penalizes inconsistency between rotation/translation
        # and position channels in 198-dim motion. Passed in pre-computed by trainer.
        if (self.fk_consistency_weight > 0.0
                and fk_consistency_loss is not None):
            warmup = 1.0
            if (self.fk_consistency_warmup_steps > 0
                    and global_step is not None
                    and global_step < self.fk_consistency_warmup_steps):
                warmup = global_step / self.fk_consistency_warmup_steps
            loss_dict["fk_consistency"] = (
                self.fk_consistency_weight * warmup * fk_consistency_loss
            )

        # Foot contact BCE loss: penalizes inconsistency between foot contact
        # prediction and ground truth. Passed in pre-computed by trainer.
        if (self.foot_contact_weight > 0.0
                and pred_contact is not None
                and gt_contact is not None):
            warmup = 1.0
            if (self.foot_contact_warmup_steps > 0
                    and global_step is not None
                    and global_step < self.foot_contact_warmup_steps):
                warmup = global_step / self.foot_contact_warmup_steps

            # pred_contact and gt_contact shape: (B, L, 4)
            # Compute binary cross entropy (expects logits vs binary targets)
            bce_loss = F.binary_cross_entropy_with_logits(
                pred_contact, gt_contact, reduction='none'
            )  # (B, L, 4)

            # Average over contact dimensions
            bce_per_frame = bce_loss.mean(dim=-1)  # (B, L)

            # Apply temporal masking to exclude padded frames
            data_mask_temporal_bce = data_mask_temporal.to(bce_per_frame.device).to(bce_per_frame.dtype)
            bce_masked = bce_per_frame * data_mask_temporal_bce
            mask_sum_bce = torch.clamp(data_mask_temporal_bce.sum(), min=1.0)
            bce_loss_scalar = bce_masked.sum() / mask_sum_bce

            loss_dict["foot_contact"] = (
                self.foot_contact_weight * warmup * bce_loss_scalar
            )

        return loss_dict
