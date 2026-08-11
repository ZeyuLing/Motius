"""Sparse-root join training on real hard-projected rollout states.

Ordinary flow matching sees a clean noise-to-motion interpolation. Conditional
inference instead repeatedly advances the model state and hard-projects known
coordinates. The resulting errors are correlated across translation, pose, and
position channels; perturbing translation alone does not reproduce that state.

This trainer keeps the complete ordinary objective. For a small subset of
samples containing isolated root observations it additionally performs a
rollout with the production hard projection and fits the two one-sided root
joins plus their curvature to the clean endpoint. The default uses one
trainable endpoint prediction from a detached rollout state. Optionally, the
rollout reaches the real solver endpoint and backpropagates through only its
last few Euler steps, removing the constant-vector endpoint approximation
without retaining the full inference graph. The rollout can start either from
a clean-path state or from the production noise state.
Pure T2M, temporal spans, dense trajectories, and body-only conditions receive
no auxiliary gradient.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.registry import TRAINERS
from motius.trainers.motioncanvas.trainer import (
    MotionCanvasTrainer,
    _batch_sources,
)


@TRAINERS.register_module()
class MotionCanvasSparseRolloutJoinTrainer(MotionCanvasTrainer):
    """Fit sparse root joins after a short inference-like rollout."""

    def __init__(
        self,
        bundle,
        val_num_steps: int = 10,
        max_text_len: int = 128,
        sparse_rollout_join_weight: float = 0.2,
        sparse_rollout_join_num_steps: int = 4,
        sparse_rollout_join_inference_steps: int = 50,
        sparse_rollout_join_max_time: float = 0.9,
        sparse_rollout_join_tail_fraction: float = 0.2,
        sparse_rollout_join_curvature_fraction: float = 0.5,
        sparse_rollout_join_max_samples: int = 4,
        sparse_rollout_excess_weight: float = 0.0,
        sparse_rollout_from_noise: bool = False,
        sparse_rollout_min_steps: int = 1,
        sparse_rollout_backprop_tail_steps: int = 1,
    ):
        super().__init__(
            bundle=bundle,
            val_num_steps=val_num_steps,
            max_text_len=max_text_len,
        )
        if hasattr(self, "mask_aware_noise"):
            self.mask_aware_noise = True
        self.sparse_rollout_join_weight = float(sparse_rollout_join_weight)
        self.sparse_rollout_join_num_steps = int(sparse_rollout_join_num_steps)
        self.sparse_rollout_join_inference_steps = int(
            sparse_rollout_join_inference_steps
        )
        self.sparse_rollout_join_max_time = float(sparse_rollout_join_max_time)
        self.sparse_rollout_join_tail_fraction = float(
            sparse_rollout_join_tail_fraction
        )
        self.sparse_rollout_join_curvature_fraction = float(
            sparse_rollout_join_curvature_fraction
        )
        self.sparse_rollout_join_max_samples = int(sparse_rollout_join_max_samples)
        self.sparse_rollout_excess_weight = float(sparse_rollout_excess_weight)
        self.sparse_rollout_from_noise = bool(sparse_rollout_from_noise)
        self.sparse_rollout_min_steps = int(sparse_rollout_min_steps)
        self.sparse_rollout_backprop_tail_steps = int(
            sparse_rollout_backprop_tail_steps
        )

        if self.sparse_rollout_join_weight < 0.0:
            raise ValueError("sparse_rollout_join_weight must be non-negative")
        if self.sparse_rollout_join_num_steps <= 1:
            raise ValueError("sparse_rollout_join_num_steps must exceed one")
        if self.sparse_rollout_join_inference_steps <= 0:
            raise ValueError(
                "sparse_rollout_join_inference_steps must be positive"
            )
        if (
            self.sparse_rollout_join_num_steps
            >= self.sparse_rollout_join_inference_steps
        ):
            raise ValueError(
                "sparse rollout must be shorter than the inference schedule"
            )
        if not 0.0 < self.sparse_rollout_join_max_time < 1.0:
            raise ValueError("sparse_rollout_join_max_time must lie in (0, 1)")
        if not 0.0 < self.sparse_rollout_join_tail_fraction <= 1.0:
            raise ValueError(
                "sparse_rollout_join_tail_fraction must lie in (0, 1]"
            )
        if not 0.0 <= self.sparse_rollout_join_curvature_fraction <= 1.0:
            raise ValueError(
                "sparse_rollout_join_curvature_fraction must lie in [0, 1]"
            )
        if self.sparse_rollout_join_max_samples <= 0:
            raise ValueError("sparse_rollout_join_max_samples must be positive")
        if self.sparse_rollout_excess_weight < 0.0:
            raise ValueError("sparse_rollout_excess_weight must be non-negative")
        if self.sparse_rollout_min_steps <= 0:
            raise ValueError("sparse_rollout_min_steps must be positive")
        if self.sparse_rollout_min_steps > self.sparse_rollout_join_num_steps:
            raise ValueError(
                "sparse_rollout_min_steps cannot exceed "
                "sparse_rollout_join_num_steps"
            )
        if self.sparse_rollout_backprop_tail_steps <= 0:
            raise ValueError(
                "sparse_rollout_backprop_tail_steps must be positive"
            )
        if (
            self.sparse_rollout_backprop_tail_steps
            >= self.sparse_rollout_join_inference_steps
        ):
            raise ValueError(
                "backprop tail must be shorter than the inference schedule"
            )
        if (
            self.sparse_rollout_backprop_tail_steps > 1
            and not self.sparse_rollout_from_noise
        ):
            raise ValueError(
                "multi-step backprop tail requires from-noise rollout"
            )
        if (
            self.sparse_rollout_from_noise
            and self.sparse_rollout_join_num_steps
            / float(self.sparse_rollout_join_inference_steps)
            > self.sparse_rollout_join_max_time
        ):
            raise ValueError(
                "from-noise rollout horizon cannot exceed "
                "sparse_rollout_join_max_time"
            )

    @staticmethod
    def _align_mask(mask: Tensor, target: Tensor) -> Tensor:
        if mask.shape[0] != target.shape[0] or mask.shape[-1] != target.shape[-1]:
            raise ValueError("mask and target batch/feature dimensions differ")
        if mask.shape[1] > target.shape[1]:
            raise ValueError("mask cannot be longer than target")
        if mask.shape[1] < target.shape[1]:
            mask = F.pad(
                mask,
                (0, 0, target.shape[1] - mask.shape[1], 0),
                value=0.0,
            )
        return mask

    @staticmethod
    def _isolated_root_centers(
        generation_mask: Tensor,
        data_mask_temporal: Tensor,
    ) -> Tensor:
        """Return generated-known-generated translation stencils per axis."""
        if generation_mask.shape[:2] != data_mask_temporal.shape:
            raise ValueError("generation and temporal masks must align")
        if generation_mask.shape[-1] < 3:
            raise ValueError("generation mask must contain translation channels")
        valid = data_mask_temporal.to(torch.bool)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        if generation_mask.shape[1] < 3:
            return torch.zeros(
                generation_mask.shape[0],
                0,
                3,
                dtype=torch.bool,
                device=generation_mask.device,
            )
        return generated[:, :-2] & known[:, 1:-1] & generated[:, 2:]

    @staticmethod
    def _mean_cvar(
        values: Tensor,
        active: Tensor,
        tail_fraction: float,
    ) -> Tensor:
        """Average per-sample mean and upper-tail risk without batch dilution."""
        if values.shape != active.shape:
            raise ValueError("values and active masks must align")
        if values.ndim != 2:
            raise ValueError("values must have shape (B, N)")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must lie in (0, 1]")

        active_count = active.sum(dim=-1)
        active_sample = active_count > 0
        mean = (
            values * active.to(values.dtype)
        ).sum(dim=-1) / active_count.clamp_min(1).to(values.dtype)

        tail_count = torch.ceil(
            active_count.to(values.dtype) * tail_fraction
        ).to(torch.long).clamp_min(1)
        ranked = values.masked_fill(~active, -torch.inf).sort(
            dim=-1,
            descending=True,
        ).values
        ranks = torch.arange(ranked.shape[-1], device=ranked.device)[None, :]
        selected = (ranks < tail_count[:, None]) & active_sample[:, None]
        tail = torch.where(
            selected,
            ranked,
            torch.zeros_like(ranked),
        ).sum(dim=-1) / tail_count.to(values.dtype)
        sample_loss = 0.5 * (mean + tail)
        return (
            sample_loss * active_sample.to(sample_loss.dtype)
        ).sum() / active_sample.sum().clamp_min(1).to(sample_loss.dtype)

    @classmethod
    def _join_geometry_loss(
        cls,
        flow_error_m: Tensor,
        isolated_centers: Tensor,
        tail_fraction: float,
        curvature_fraction: float,
        pseudo_huber_beta_m: float = 2e-3,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Penalize both one-sided join errors and their coupled curvature."""
        if flow_error_m.ndim != 3 or flow_error_m.shape[-1] != 3:
            raise ValueError("flow_error_m must have shape (B, T, 3)")
        if isolated_centers.shape != (
            flow_error_m.shape[0],
            max(flow_error_m.shape[1] - 2, 0),
            3,
        ):
            raise ValueError("isolated centers must align with root stencils")
        if not 0.0 <= curvature_fraction <= 1.0:
            raise ValueError("curvature_fraction must lie in [0, 1]")
        if pseudo_huber_beta_m <= 0.0:
            raise ValueError("pseudo_huber_beta_m must be positive")

        zero = flow_error_m.sum() * 0.0
        if flow_error_m.shape[1] < 3 or not isolated_centers.any():
            return zero, {
                "edge_error_m": zero.detach(),
                "curvature_error_m": zero.detach(),
            }

        center_active = isolated_centers.any(dim=-1)
        left = flow_error_m[:, :-2]
        right = flow_error_m[:, 2:]
        axis_mask = isolated_centers.to(flow_error_m.dtype)
        edge_vectors = torch.stack((left, right), dim=2)
        edge_norm = torch.sqrt(
            (edge_vectors.square() * axis_mask[:, :, None]).sum(dim=-1) + 1e-12
        )
        edge_active = center_active[:, :, None].expand_as(edge_norm)

        curvature_vector = left + right
        curvature_norm = torch.sqrt(
            (curvature_vector.square() * axis_mask).sum(dim=-1) + 1e-12
        )
        beta = pseudo_huber_beta_m
        edge_penalty = torch.sqrt(edge_norm.square() + beta * beta) - beta
        curvature_penalty = (
            torch.sqrt(curvature_norm.square() + beta * beta) - beta
        )
        edge_loss = cls._mean_cvar(
            edge_penalty.flatten(start_dim=1),
            edge_active.flatten(start_dim=1),
            tail_fraction,
        )
        curvature_loss = cls._mean_cvar(
            curvature_penalty,
            center_active,
            tail_fraction,
        )
        loss = (
            (1.0 - curvature_fraction) * edge_loss
            + curvature_fraction * curvature_loss
        )

        edge_float = edge_active.to(edge_norm.dtype)
        center_float = center_active.to(curvature_norm.dtype)
        return loss, {
            "edge_error_m": (
                (edge_norm * edge_float).sum() / edge_float.sum().clamp_min(1.0)
            ).detach(),
            "curvature_error_m": (
                (curvature_norm * center_float).sum()
                / center_float.sum().clamp_min(1.0)
            ).detach(),
        }

    @classmethod
    def _background_calibrated_excess_loss(
        cls,
        pred_root_m: Tensor,
        gt_root_m: Tensor,
        generation_mask: Tensor,
        data_mask_temporal: Tensor,
        isolated_centers: Tensor,
        remaining: Tensor,
        tail_fraction: float,
        pseudo_huber_beta_m: float = 2e-3,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Suppress only curvature above both GT and generated background.

        The background norm uses every translation axis observed anywhere in
        the sample. A two-frame guard band prevents the same join error from
        leaking into its own detached per-sample background ceiling. Taking the
        maximum with clean GT curvature preserves genuine high-acceleration
        events instead of imposing generic smoothing.
        """
        if pred_root_m.shape != gt_root_m.shape or pred_root_m.shape[-1] != 3:
            raise ValueError("predicted and GT roots must share shape (B, T, 3)")
        if generation_mask.shape[:2] != pred_root_m.shape[:2]:
            raise ValueError("generation mask must align with root motion")
        if data_mask_temporal.shape != pred_root_m.shape[:2]:
            raise ValueError("temporal mask must align with root motion")
        if isolated_centers.shape != (
            pred_root_m.shape[0],
            max(pred_root_m.shape[1] - 2, 0),
            3,
        ):
            raise ValueError("isolated centers must align with root stencils")
        if remaining.shape != (pred_root_m.shape[0], 1, 1):
            raise ValueError("remaining flow time must have shape (B, 1, 1)")
        if pseudo_huber_beta_m <= 0.0:
            raise ValueError("pseudo_huber_beta_m must be positive")

        zero = pred_root_m.sum() * 0.0
        if pred_root_m.shape[1] < 3 or not isolated_centers.any():
            return zero, {
                "condition_curvature_m": zero.detach(),
                "background_curvature_m": zero.detach(),
                "excess_curvature_m": zero.detach(),
                "condition_to_background_ratio": zero.detach(),
            }

        valid = data_mask_temporal.to(torch.bool)
        known_axes = (generation_mask[..., :3] < 0.5) & valid[..., None]
        known_frames = known_axes.any(dim=-1)
        active_axes = known_axes.any(dim=1)
        valid_window = valid[:, :-2] & valid[:, 1:-1] & valid[:, 2:]
        guarded_condition = F.max_pool1d(
            known_frames[:, None].to(pred_root_m.dtype),
            kernel_size=5,
            stride=1,
            padding=2,
        )[:, 0] > 0.5
        condition_window = guarded_condition[:, 1:-1]
        background = (
            valid_window
            & ~condition_window
            & active_axes.any(dim=-1, keepdim=True)
        )

        pred_curvature = (
            pred_root_m[:, 2:]
            - 2.0 * pred_root_m[:, 1:-1]
            + pred_root_m[:, :-2]
        )
        gt_curvature = (
            gt_root_m[:, 2:]
            - 2.0 * gt_root_m[:, 1:-1]
            + gt_root_m[:, :-2]
        )
        active_axis_float = active_axes[:, None].to(pred_curvature.dtype)
        background_norm = torch.sqrt(
            (pred_curvature.square() * active_axis_float).sum(dim=-1) + 1e-12
        )
        background_float = background.to(background_norm.dtype)
        background_count = background_float.sum(dim=-1)
        background_mean = (
            (background_norm * background_float).sum(dim=-1)
            / background_count.clamp_min(1.0)
        ).detach()

        center_float = isolated_centers.to(pred_curvature.dtype)
        center_active = isolated_centers.any(dim=-1)
        pred_condition_norm = torch.sqrt(
            (pred_curvature.square() * center_float).sum(dim=-1) + 1e-12
        )
        gt_condition_norm = torch.sqrt(
            (gt_curvature.square() * center_float).sum(dim=-1) + 1e-12
        ).detach()
        background_ceiling = background_mean[:, None].expand_as(
            pred_condition_norm
        )
        has_background = background_count[:, None] > 0
        ceiling = torch.where(
            has_background,
            torch.maximum(background_ceiling, gt_condition_norm),
            gt_condition_norm,
        )
        excess_m = torch.relu(pred_condition_norm - ceiling)
        flow_excess_m = excess_m / remaining.view(-1, 1)
        beta = pseudo_huber_beta_m
        penalty = torch.sqrt(flow_excess_m.square() + beta * beta) - beta
        loss = cls._mean_cvar(penalty, center_active, tail_fraction)

        center_active_float = center_active.to(pred_condition_norm.dtype)
        center_count = center_active_float.sum().clamp_min(1.0)
        condition_mean = (
            pred_condition_norm * center_active_float
        ).sum() / center_count
        excess_mean = (excess_m * center_active_float).sum() / center_count
        active_sample = center_active.any(dim=-1) & (background_count > 0)
        active_sample_float = active_sample.to(background_mean.dtype)
        active_sample_count = active_sample_float.sum().clamp_min(1.0)
        background_active_mean = (
            background_mean * active_sample_float
        ).sum() / active_sample_count
        ratio = condition_mean / background_active_mean.clamp_min(1e-6)
        return loss, {
            "condition_curvature_m": condition_mean.detach(),
            "background_curvature_m": background_active_mean.detach(),
            "excess_curvature_m": excess_mean.detach(),
            "condition_to_background_ratio": ratio.detach(),
        }

    def _rollout_student_endpoint(
        self,
        batch: Dict[str, Any],
        ctx: Dict[str, Any],
    ) -> Tuple[
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
        Optional[Tensor],
    ]:
        generation_mask = ctx["generation_mask"]
        if generation_mask is None:
            return None, None, None, None, None, None
        aligned_mask = self._align_mask(generation_mask, ctx["x1"])
        centers = self._isolated_root_centers(
            aligned_mask,
            ctx["tgt_padding_mask"],
        )
        active = centers.flatten(start_dim=1).any(dim=1)
        if not self.sparse_rollout_from_noise:
            rollout_horizon = (
                self.sparse_rollout_join_num_steps
                / float(self.sparse_rollout_join_inference_steps)
            )
            latest_start = self.sparse_rollout_join_max_time - rollout_horizon
            active &= ctx["timesteps"] <= latest_start
        if not active.any():
            return None, None, None, None, None, None
        index = active.nonzero(as_tuple=False).flatten()
        if index.numel() > self.sparse_rollout_join_max_samples:
            order = torch.randperm(index.numel(), device=index.device)
            index = index[order[: self.sparse_rollout_join_max_samples]]

        known = aligned_mask[index] < 0.5
        exact_endpoint = self.sparse_rollout_backprop_tail_steps > 1
        if self.sparse_rollout_from_noise:
            # Sample states from the trajectory the production solver actually
            # visits. Starting from a clean-flow x_t and taking only a few
            # projected steps severely underestimates accumulated off-path
            # error at sparse waypoints.
            state = torch.where(
                known,
                ctx["x1"][index],
                ctx["x0"][index],
            ).detach()
            rollout_t = torch.zeros_like(ctx["t"][index])
            if exact_endpoint:
                rollout_steps = 0
            else:
                # Every DDP rank must execute the same number of model
                # forwards. Cycling by the shared optimizer step covers the
                # horizon without rank-local RNG or a synchronization call.
                horizon_count = (
                    self.sparse_rollout_join_num_steps
                    - self.sparse_rollout_min_steps
                    + 1
                )
                rollout_steps = self.sparse_rollout_min_steps + (
                    self.get_global_step() % horizon_count
                )
        else:
            state = ctx["x_t"][index].detach()
            rollout_t = ctx["t"][index].detach()
            rollout_steps = self.sparse_rollout_join_num_steps
        step = 1.0 / float(self.sparse_rollout_join_inference_steps)
        sources = _batch_sources(batch, ctx["x1"].shape[0])
        if sources is not None:
            sources = [sources[i] for i in index.detach().cpu().tolist()]
        task_emb = ctx.get("task_emb")
        if task_emb is not None:
            task_emb = task_emb[index]

        common = dict(
            task_emb=task_emb,
            ctxt_input=ctx["ctxt_input"][index],
            vtxt_input=ctx["vtxt_input"][index],
            x_mask_temporal=ctx["tgt_padding_mask"][index],
            ctxt_mask_temporal=ctx["ctxt_mask_temporal"][index],
            sources=sources,
            trigger_sources={"Taobao", "Game"},
        )
        if exact_endpoint:
            detached_steps = (
                self.sparse_rollout_join_inference_steps
                - self.sparse_rollout_backprop_tail_steps
            )
        else:
            detached_steps = rollout_steps

        with torch.no_grad():
            for _ in range(detached_steps):
                pred = self.bundle.predict_flow(
                    x_input=torch.cat(
                        [state, ctx["condition_context"][index]],
                        dim=-1,
                    ),
                    timesteps=rollout_t.view(-1),
                    **common,
                )
                state = state + step * pred
                state = torch.where(known, ctx["x1"][index], state)
                rollout_t = rollout_t + step

        if exact_endpoint:
            for _ in range(self.sparse_rollout_backprop_tail_steps):
                pred = self.bundle.predict_flow(
                    x_input=torch.cat(
                        [state, ctx["condition_context"][index]],
                        dim=-1,
                    ),
                    timesteps=rollout_t.view(-1),
                    **common,
                )
                state = state + step * pred
                state = torch.where(known, ctx["x1"][index], state)
                rollout_t = rollout_t + step
            endpoint = state
            # Normalize the truncated-tail gradient to the one-step endpoint
            # objective: each Euler update contributes `step`, while together
            # the differentiable tail spans this much flow time.
            remaining = torch.full_like(
                rollout_t,
                self.sparse_rollout_backprop_tail_steps * step,
            )
        else:
            student_pred = self.bundle.predict_flow(
                x_input=torch.cat(
                    [state, ctx["condition_context"][index]],
                    dim=-1,
                ),
                timesteps=rollout_t.view(-1),
                **common,
            )
            remaining = (1.0 - rollout_t).clamp_min(0.1)
            endpoint = state + remaining * student_pred
            endpoint = torch.where(known, ctx["x1"][index], endpoint)
        return (
            endpoint,
            index,
            centers[index],
            remaining,
            aligned_mask[index],
            rollout_t,
        )

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if self.bundle.pred_type != "velocity":
            raise NotImplementedError(
                "sparse rollout join training requires pred_type='velocity'"
            )
        ctx = self._prepare_and_forward(batch)
        losses = self._compute_base_loss(ctx)
        zero = ctx["pred"].sum() * 0.0
        join_loss = zero
        diagnostics = {
            "active_samples": zero.detach(),
            "edge_error_m": zero.detach(),
            "curvature_error_m": zero.detach(),
            "condition_curvature_m": zero.detach(),
            "background_curvature_m": zero.detach(),
            "excess_curvature_m": zero.detach(),
            "condition_to_background_ratio": zero.detach(),
            "rollout_time": zero.detach(),
        }
        excess_loss = zero

        if (
            self.sparse_rollout_join_weight > 0.0
            or self.sparse_rollout_excess_weight > 0.0
        ):
            endpoint, index, centers, remaining, selected_mask, rollout_time = (
                self._rollout_student_endpoint(
                    batch,
                    ctx,
                )
            )
            if endpoint is not None:
                root_std = self.bundle.std[:3].to(
                    device=endpoint.device,
                    dtype=endpoint.dtype,
                ).clamp_min(1e-3)
                root_mean = self.bundle.mean[:3].to(
                    device=endpoint.device,
                    dtype=endpoint.dtype,
                )
                flow_error_m = (
                    (endpoint[..., :3] - ctx["x1"][index, ..., :3])
                    * root_std
                    / remaining
                )
                join_loss, join_diag = self._join_geometry_loss(
                    flow_error_m,
                    centers,
                    self.sparse_rollout_join_tail_fraction,
                    self.sparse_rollout_join_curvature_fraction,
                )
                diagnostics.update(join_diag)
                if self.sparse_rollout_excess_weight > 0.0:
                    pred_root_m = endpoint[..., :3] * root_std + root_mean
                    gt_root_m = (
                        ctx["x1"][index, ..., :3] * root_std + root_mean
                    )
                    excess_loss, excess_diag = (
                        self._background_calibrated_excess_loss(
                            pred_root_m,
                            gt_root_m,
                            selected_mask,
                            ctx["tgt_padding_mask"][index],
                            centers,
                            remaining,
                            self.sparse_rollout_join_tail_fraction,
                        )
                    )
                    diagnostics.update(excess_diag)
                diagnostics["active_samples"] = torch.tensor(
                    float(index.numel()),
                    device=endpoint.device,
                    dtype=endpoint.dtype,
                )
                diagnostics["rollout_time"] = rollout_time.mean().detach()

        losses["sparse_rollout_join"] = (
            self.sparse_rollout_join_weight * join_loss
        )
        losses["sparse_rollout_excess"] = (
            self.sparse_rollout_excess_weight * excess_loss
        )
        total_loss = self.sum_train_losses(losses)
        result: Dict[str, Any] = {"loss": total_loss}
        for name, value in losses.items():
            result[f"loss_{name}"] = value.detach()
        for name, value in diagnostics.items():
            result[f"loss_sparse_rollout_join_{name}"] = value
        return result
