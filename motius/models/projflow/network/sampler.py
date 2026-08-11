"""ProjFlow projection sampler for ACMDM rectified flow priors.

This module is a package-local adaptation of the official ProjFlow sampler. It
keeps the published update order and defaults while avoiding any runtime import
from an external repository checkout.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .projection import (
    KinematicMetric,
    build_kinematic_metric,
    build_pseudo_observations,
    build_skeleton_laplacian,
    curvature_per_frame,
    distribute_trust_over_halo_joints,
    expand_t_like_x,
    flowdps_eta,
    frame_trust_schedule,
    halo_radius_linear,
    metric_project_clean_endpoint,
    noise_mix,
    trust_to_variance,
    tweedie_endpoints_from_velocity,
)


class LinearFlowPath:
    """The linear interpolation used by the released ACMDM Flow checkpoint."""

    @staticmethod
    def compute_alpha_t(t: torch.Tensor):
        return t, torch.ones_like(t)

    @staticmethod
    def compute_sigma_t(t: torch.Tensor):
        return 1.0 - t, -torch.ones_like(t)


class ProjFlowSampler:
    """Kinematics-aware projection sampler from the ProjFlow release."""

    def __init__(self):
        self.path_sampler = LinearFlowPath()

    def sample_ode(
        self,
        *,
        sampling_method: str = "euler",
        num_steps: int = 50,
        atol: float = 1e-6,
        rtol: float = 1e-3,
        reverse: bool = False,
    ):
        """Match the official ACMDM probability-flow ODE wrapper."""

        from torchdiffeq import odeint

        def sample(x: torch.Tensor, model, **kwargs):
            device = x.device
            times = torch.linspace(0.0, 1.0, num_steps, device=device)
            if reverse:
                times = torch.flip(times, dims=(0,))

            def drift(t, value):
                batch_t = torch.ones(value.shape[0], device=device) * t
                model_t = 1.0 - batch_t if reverse else batch_t
                return model(value, model_t, **kwargs)

            return odeint(
                drift,
                x,
                times,
                method=sampling_method,
                atol=atol,
                rtol=rtol,
            )

        return sample

    def sample_projflow(
        self,
        *,
        num_steps: int = 100,
        generator: Optional[torch.Generator] = None,
        w_kin: float = 10.0,
        ridge: float = 1.0,
        ell_min: float = 3.0,
        ell_max: float = 10.0,
        tau_min: float = 0.1,
        c0: float = 3.0,
        lambda_s: float = 1.0,
        p: float = 2.0,
        pi_min: float = 0.02,
        pi_max: float = 1.0,
        schur_block: int = 1024,
        use_projflow: bool = True,
    ):
        if num_steps < 2:
            raise ValueError("num_steps must be at least 2")
        path = self.path_sampler

        def sample(x: torch.Tensor, model, **kwargs) -> List[torch.Tensor]:
            batch, _, _, joints = x.shape
            device, dtype = x.device, x.dtype
            hard_mask = kwargs["A"].to(device=device, dtype=dtype)
            hard_value = kwargs["y"].to(device=device, dtype=dtype)
            if hard_mask.shape != x.shape or hard_value.shape != x.shape:
                raise ValueError("ProjFlow controls must match the sampled tensor shape")

            laplacian = build_skeleton_laplacian(joints, device=device, dtype=dtype)
            if use_projflow:
                metric = build_kinematic_metric(
                    J=joints,
                    w_kin=w_kin,
                    ridge=ridge,
                    L_kin=laplacian,
                )
            else:
                ones = torch.ones(joints, device=device, dtype=dtype)
                metric = KinematicMetric(
                    apply_Rinv=lambda value: value,
                    diag_Rinv_joint=ones,
                    joint_weights_q=ones,
                    L_kin=laplacian,
                )

            times = torch.linspace(0.0, 1.0, num_steps, device=device, dtype=dtype)
            samples: List[torch.Tensor] = []
            for step in range(num_steps - 1):
                t_value = float(times[step].item())
                next_value = float(times[step + 1].item())
                t = torch.full((batch,), t_value, device=device, dtype=dtype)
                next_t = torch.full((batch,), next_value, device=device, dtype=dtype)

                velocity = model(x, t, **kwargs)
                x1_hat, x0_hat = tweedie_endpoints_from_velocity(
                    x_t=x,
                    t=t,
                    v=velocity,
                    path_sampler=path,
                )

                if use_projflow:
                    radius = halo_radius_linear(
                        step,
                        num_steps,
                        ell_min=ell_min,
                        ell_max=ell_max,
                    )
                    pseudo, halo = build_pseudo_observations(
                        hard_mask=hard_mask,
                        hard_value=hard_value,
                        halo_radius=radius,
                    )
                    selector = ((hard_mask > 0.5) | (halo > 0.5)).to(dtype)
                    targets = torch.where(
                        hard_mask > 0.5,
                        hard_value,
                        torch.where(halo > 0.5, pseudo, hard_value),
                    )
                    curvature = curvature_per_frame(
                        x1_hat=x1_hat,
                        metric=metric,
                        w_kin=w_kin,
                        ridge=ridge,
                    )
                    frame_trust = frame_trust_schedule(
                        t_scalar=t_value,
                        curvature=curvature,
                        pi_min=pi_min,
                        pi_max=pi_max,
                        tau_min=tau_min,
                        c0=c0,
                        lambda_s=lambda_s,
                        p=p,
                    )
                    joint_trust = distribute_trust_over_halo_joints(
                        pi_frame=frame_trust,
                        M_halo_any=halo[:, 0] > 0.5,
                        q_joint=metric.joint_weights_q,
                        pi_min=pi_min,
                        pi_max=pi_max,
                    )
                    variance = trust_to_variance(
                        pi_joint=joint_trust,
                        metric=metric,
                        hard_mask=hard_mask,
                        M_all=selector,
                    )
                else:
                    selector = (hard_mask > 0.5).to(dtype)
                    targets = hard_value
                    variance = None

                projected = metric_project_clean_endpoint(
                    x1_hat=x1_hat,
                    selector=selector,
                    targets=targets,
                    apply_Rinv=metric.apply_Rinv,
                    sigma2=variance,
                    block_size=schur_block,
                )
                alpha_next, _ = path.compute_alpha_t(expand_t_like_x(next_t, x))
                sigma_next, _ = path.compute_sigma_t(expand_t_like_x(next_t, x))
                if use_projflow:
                    mixed_noise = noise_mix(
                        x0_hat,
                        flowdps_eta(sigma_next),
                        generator,
                    )
                else:
                    mixed_noise = x0_hat
                x = alpha_next * projected + sigma_next * mixed_noise
                samples.append(x)
            return samples

        return sample


__all__ = ["LinearFlowPath", "ProjFlowSampler"]
