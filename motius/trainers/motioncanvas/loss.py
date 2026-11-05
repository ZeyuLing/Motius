"""Unified training objective for HYMotion T2M, M2M, and UMO.

The loss combines one representation-space flow-matching objective with
decoded-space geometry objectives. Optional mask-derived root-completion and
motion-adaptive support terms target sparse-condition discontinuities and
root/body coordination without assuming a benchmark-specific waypoint pattern
or a fixed set of contacting joints.

Conventions
-----------
* Motion tensors are ``(B, T, D)`` and normalized unless stated otherwise.
* ``generation_mask`` uses ``1=generate`` and ``0=known condition``.
* ``data_mask_temporal`` uses ``1=valid frame`` and ``0=padding``.
* The 198-D layout is ``trans[0:3] + rot6d[3:135] + position[135:198]``.
  Position stores 21 non-pelvis joints relative to the pelvis.
* Known coordinates follow the clamped path ``x_t=x_1`` and therefore have
  zero flow velocity. Generated and known coordinates are reduced separately so
  sparse evidence is not diluted by the much larger target region. Generated
  coordinates near evidence can also be reduced separately, while retaining
  the exact same flow target, to keep sparse joins from disappearing in the
  whole-clip average.
* Geometry losses receive a condition-projected ``pred_x1``. The trainer must
  replace known coordinates with their clean condition values before calling
  this module; velocity-to-x1 conversion is invalid on MAN-clamped coordinates.
  Generated coordinates close to evidence may additionally supervise this
  projected clean endpoint. Since ``pred_x1-x1=(1-t)(pred_v-v)`` on generated
  coordinates, this targets high-noise denoising errors that ordinary flow
  matching underweights without prescribing a smoothed motion target.

For ``x_t=(1-t)x_0+t x_1``, the primary target is ``v=x_1-x_0``. Geometry
terms operate on denormalized metres. Optional ``t^2`` weighting suppresses
unreliable FK-decoded geometry near the noise endpoint, including condition-
transition derivatives. The primary representation-space flow loss remains
active over the complete path.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


_SMPL22_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19,
)


def _smpl22_descendant_matrix() -> Tuple[Tuple[float, ...], ...]:
    """Return a source-joint to affected-descendant adjacency matrix."""
    rows = []
    for source in range(len(_SMPL22_PARENTS)):
        row = []
        for joint in range(len(_SMPL22_PARENTS)):
            current = joint
            while current >= 0 and current != source:
                current = _SMPL22_PARENTS[current]
            row.append(float(current == source))
        rows.append(tuple(row))
    return tuple(rows)


_SMPL22_DESCENDANT_MATRIX = _smpl22_descendant_matrix()


def _safe_std(std: Tensor) -> Tensor:
    """Match normalization semantics while avoiding division-scale collapse."""
    return torch.where(std < 1e-3, torch.ones_like(std), std)


def _fk_global_positions(
    motion_135_denorm: Tensor,
    bone_offsets: Tensor,
    rotation_space: str,
) -> Tensor:
    """Decode ``trans+rot6d`` to world-space SMPL-22 joint positions."""
    from motius.motion.pipeline_utils.differentiable_fk import motion135_to_fk

    world_pos, _, _, _ = motion135_to_fk(
        motion_135_denorm,
        bone_offsets,
        rotation_space=rotation_space,
    )
    return world_pos


def _strict_ric_relative(world_pos: Tensor) -> Tensor:
    """Encode world joints as pelvis-relative non-pelvis XYZ (21 x 3)."""
    body_rel = world_pos[..., 1:, :] - world_pos[..., 0:1, :]
    return body_rel.reshape(*body_rel.shape[:-2], 63)


def _temporal_mean_masked(per_frame: Tensor, mask: Tensor) -> Tensor:
    """Average per-frame values over valid temporal entries only."""
    mask = mask.to(device=per_frame.device, dtype=per_frame.dtype)
    return (per_frame * mask).sum() / torch.clamp(mask.sum(), min=1.0)


class M2MLoss(nn.Module):
    """Masked flow-matching plus decoded motion geometry supervision.

    The optimized objective is

    ``L = w_flow (L_flow + g L_near) + h L_near_x1``
    ``+ q L_near_world``
    ``+ w_traj L_root_traj``
    ``+ w_h1 L_cond_root_h1 + w_h2 L_cond_root_h2``
    ``+ w_pos L_joint_pos + w_vel (L_joint_vel + a L_cond_vel``
    ``+ b L_cond_acc + c L_support_vel + d L_transition``
    ``+ e L_jerk_ceiling + f L_secant_excess)``
    ``+ r L_transition_residual + s L_transition_sobolev``
    ``+ u L_root_anchor + v L_root_h2_join + w_fk L_fk``.

    ``L_flow`` is Smooth-L1/L1/MSE on the predicted flow velocity. Generated
    coordinates target ``x_1-x_0`` while clean-imputed known coordinates target
    zero velocity; the two supports are reduced independently. ``L_root_traj``
    directly compares the reconstructed root translation against
    the clean trajectory on generated translation coordinates for every task.
    It is not ``t^2`` weighted, so physical-scale translation supervision covers
    the complete flow path. ``L_joint_pos`` and ``L_joint_vel`` compare all 22
    FK-decoded world joints in metres. These two terms supervise root/body
    coordination without assuming which body parts are in contact. ``L_fk``
    ties the redundant 198-D position channels to positions decoded from
    translation and rotations.

    Args:
        loss_type: Pointwise loss used by every optimized term.
        velocity_weight: Weight of the masked flow-matching objective.
        velocity_loss_reduction: ``element_mean``/``official_element_mean``
            average all active coordinates; ``component_mean``/``modality_mean``
            first average translation, root rotation, body rotation, and joint
            position independently, then average active components.
        condition_neighborhood_flow_weight: Relative weight for the ordinary
            generated-coordinate flow objective within 1, 2, and 4 frames of
            known evidence on the same representation channel. It changes only
            sampling importance: the target remains exactly ``x_1-x_0``. Each
            active sample and component is reduced independently so an isolated
            joint-axis condition cannot vanish in a whole-clip average.
        condition_neighborhood_x1_weight: Weight for reconstructed clean-motion
            error on generated coordinates within 1, 2, and 4 frames of known
            evidence on the same representation channel. The endpoint is
            condition-projected before this loss is evaluated. Unlike a
            derivative smoothness loss, the target is the original clean motion.
        condition_neighborhood_world_weight: Weight for clean world-joint
            positions at condition-affected coordinates and their 1, 2, and
            4-frame neighborhoods. Translation evidence affects every joint on
            that world axis, rotations affect kinematic descendants, and
            position atoms affect the corresponding joint-axis. This couples
            conditioned root motion to generated body pose without imposing a
            zero-velocity or hand-authored interpolation target.
        root_trajectory_weight: Weight for direct root-translation trajectory
            reconstruction in metres on every generated translation coordinate.
            Known coordinates are excluded because inference hard-projects them.
        joint_pos_weight: Weight for world-space joint-position reconstruction.
        joint_vel_weight: Weight for finite-difference world-joint velocity.
        condition_kinematic_weight: Relative weight for world-joint velocity
            on edges supported by motion conditions. The support is derived
            directly from the coordinate mask and reduced independently.
        condition_acceleration_weight: Relative weight for second differences
            in the same condition-supported neighborhoods.
        conditioned_root_h1_weight: Weight for mask-derived root completion.
            For each translation axis observed anywhere in a sample, this term
            matches the generated trajectory and its first difference to the
            clean target. It does not invent a path between observations.
        conditioned_root_h2_weight: Weight for full-path root completion in
            position, velocity, and acceleration space. It uses a near-L1
            physical-space penalty against the clean target, without ``t^2``
            attenuation or a moving background-error threshold. It is active
            only on translation axes containing both known and generated frames,
            so T2M, body-only conditions, and fully observed trajectories receive
            exactly zero loss.
        stationary_support_velocity_weight: Relative weight, inside the world
            joint-velocity objective, for slowly moving GT joints in samples
            with root conditions. The continuous weights cover any support
            joint and the target remains the GT velocity rather than zero.
        condition_transition_excess_weight: Relative weight for velocity and
            acceleration/jerk errors at known/generated transitions that exceed
            the ordinary error of the same joint-axis elsewhere in the sample.
            This self-disables once condition boundaries are no harder than the
            generated background and follows the decoded-geometry timestep
            weighting.
        condition_transition_jerk_ceiling_weight: Relative weight for a one-sided
            predicted-jerk penalty at known/generated transitions. The ceiling is
            the larger of the same joint-axis generated-background jerk and the
            detached GT transition jerk. It therefore suppresses artificial
            boundary spikes without fitting noisy GT third differences or
            flattening legitimate action dynamics.
        condition_transition_secant_excess_weight: Relative weight for
            multi-scale first-order displacement errors across condition
            boundaries. Secants over 1, 2, and 4 frames are compared with GT
            and penalized only above the same joint-axis error in ordinary
            generated regions. This directly targets boundary continuity
            without differentiating noisy motion three times.
        condition_transition_residual_weight: Weight for first-, second-, and
            third-order derivatives of the clean-endpoint world-space error in
            a two-frame band around coordinate-mask transitions. Only absolute
            residual above the detached ordinary-region level of the same
            sample and joint-axis is optimized. This gives small physical
            discontinuities an L1-strength gradient instead of the very weak
            quadratic gradient produced by metre-scale Smooth-L1 near zero.
        condition_transition_sobolev_weight: Weight for a dimensionless local
            Sobolev-parity objective. For derivative orders one through three,
            transition-band residuals are divided by the detached ordinary-region
            residual of the same sample and joint-axis. Translation, root rotation,
            body rotation, and joint position evidence are reduced independently.
            A robust logarithmic hinge then penalizes only ratios above one. This
            keeps useful gradients at millimetre-scale errors while self-disabling
            at background parity.
        condition_transition_root_sobolev_weight: Weight for translation-native
            transition parity. This term is activated directly by sparse
            root-coordinate evidence and cannot be cancelled by relative body
            motion after FK. A one-sided Smooth-L1 penalty retains constant
            outlier gradient above background parity, and no timestep attenuation
            is applied because the normalized ratio already removes endpoint scale.
        condition_transition_root_sobolev_orders: Temporal derivative orders used
            by translation-native parity. ``(1, 2)`` targets velocity and curvature
            without the noisy, optimization-stiff third finite difference.
        condition_transition_root_anchor_weight: Weight for translation-native,
            multi-scale secants that join each sparse root observation directly
            to generated frames 1, 2, and 4 steps away. The target is the clean
            GT displacement rather than a hand-authored smooth trajectory. Each
            scale is normalized by ordinary generated-region secant error, so the
            term stops once a hard-imputation join is no harder than background
            motion and is exactly zero for T2M, body-only, and dense-root samples.
        condition_transition_root_h2_weight: Weight for exact first- and
            second-order root-error supervision in a mask-derived join band.
            Unlike parity losses, the clean GT is the fixed zero-error target;
            unlike full-path H2, ordinary completion windows cannot dilute a
            sparse boundary. The term has no ``t^2`` attenuation and is zero for
            T2M, body-only conditions, and fully observed root trajectories.
        condition_transition_root_tail_weight: Weight for a tail-sensitive
            translation-native parity objective. It computes the RMS of
            normalized excess error near sparse root joins over derivative
            orders one through three. Unlike a temporal mean, RMS gives the few
            visually dominant join spikes proportionally larger gradients. The
            support is a mask-derived band and the term is exactly zero for T2M,
            body-only conditions, and dense root trajectories.
        condition_transition_root_tail_band_radius: Number of derivative
            windows added on either side of an exact root known/generated join.
            Zero selects only stencils that cross the join; larger values also
            supervise its approach and departure.
        condition_transition_root_cvar_weight: Weight for bounded tail-risk
            optimization over sparse root joins. For each sample, translation
            axis, and derivative order, it averages only the worst 20 percent
            of the same Smooth-L1 parity residual used by the stable root
            Sobolev objective. This focuses rare visible spikes without the
            magnitude-proportional gradients of an RMS tail loss.
        condition_transition_root_dynamics_ceiling_weight: Weight for a direct
            one-sided acceleration/jerk ceiling at sparse root joins. The
            ceiling is the larger of the clean GT derivative and the detached
            ordinary generated-region magnitude from the same sample-axis, so
            legitimate sharp actions remain unconstrained while imputation
            spikes above normal motion receive a scale-normalized gradient.
        condition_transition_root_endpoint_weight: Weight for metric-space
            clean-endpoint recovery on generated root coordinates one, two,
            and four frames from sparse observations on the same axis. A
            millimetre-scale Huber transition preserves an L1-strength gradient
            for visible joins without prescribing zero velocity or smoothing
            legitimate target dynamics.
        condition_transition_root_endpoint_tail_weight: Weight for the
            worst-20-percent variant of clean-endpoint recovery. The tail is
            selected independently for every active sample and translation
            axis, so numerous already-correct neighbours cannot dilute a few
            visible sparse-waypoint joins. It uses the same clean metric-space
            target and bounded Huber gradient as the endpoint mean.
        condition_transition_root_waypoint_curvature_weight: Weight for direct
            clean-curvature recovery at isolated root observations. It compares
            the projected generated-known-generated second-difference stencil
            with the clean GT stencil in metric coordinates. This targets the
            exact sparse-waypoint jump without imposing zero acceleration on a
            legitimate action.
        condition_transition_root_waypoint_tail_weight: Weight for vector-valued
            tail-risk recovery at isolated root observations. It applies a
            bounded pseudo-Huber penalty to the joint active-axis curvature
            error, then averages the worst 20 percent of waypoints per sample.
            This matches the XZ/XYZ evaluation geometry and targets rare visible
            spikes that a coordinate-wise temporal mean can hide.
        condition_transition_root_waypoint_edge_weight: Weight for flow-scale,
            vector-valued one-sided edge recovery at isolated root observations.
            The left and right generated-to-known velocity errors are penalized
            separately, so opposite endpoint errors cannot cancel as they can in
            a second difference. Dividing endpoint error by ``1-t`` gives every
            flow timestep equal effective gradient. The worst 20 percent of
            active edges are averaged per sample with bounded pseudo-Huber
            gradients.
        condition_transition_root_waypoint_edge_tail_fraction: Fraction of
            active one-sided waypoint edges included by the edge objective.
            ``0.2`` retains the tail-risk reduction, while ``1.0`` supervises
            every edge so neither side of an isolated observation is left
            without a direct clean-trajectory gradient.
        condition_transition_root_waypoint_basin_weight: Weight for multi-scale
            flow-equivalent recovery around sparse root observations. Generated
            coordinates two through eight frames to either side are
            supervised only when the whole path to the observation is unknown.
            Each radius uses its own worst-20-percent reduction, and inverse
            radius weighting prioritizes the near basin. The separate waypoint
            edge objective retains full-strength supervision at distance one.
        condition_transition_root_waypoint_h1_weight: Weight for local
            flow-equivalent residual-velocity recovery around isolated root
            observations. Together with the distance-one waypoint edge term,
            it matches every generated edge two through eight frames from the
            anchor to the corresponding clean GT velocity. This propagates the
            zero anchor error without smoothing legitimate target dynamics.
        condition_transition_root_waypoint_h1_detach_inner: Make the local H1
            objective directional. Each outer frame is fitted to the detached
            residual of its inner neighbour, preventing a distant error from
            pulling an already-correct near-anchor frame away from evidence.
        fk_consistency_weight: Weight tying predicted position channels to FK.
        motion_dim: Minimum representation width required by geometry terms.
        joint_pos_warmup_steps: Linear warmup length for joint position.
        joint_vel_warmup_steps: Linear warmup length for joint velocity.
        fk_consistency_warmup_steps: Linear warmup length for FK consistency.
        timestep_squared_weighting: Multiply ordinary geometry samples by
            ``t^2``.
    """

    _MODALITY_MEAN_REDUCTIONS = ("component_mean", "modality_mean")
    _VALID_REDUCTIONS = (
        "element_mean",
        "official_element_mean",
        "component_mean",
        "modality_mean",
    )

    def __init__(
        self,
        loss_type: str = "smooth_l1",
        velocity_weight: float = 1.0,
        velocity_loss_reduction: str = "element_mean",
        condition_neighborhood_flow_weight: float = 0.0,
        condition_neighborhood_x1_weight: float = 0.0,
        condition_neighborhood_world_weight: float = 0.0,
        root_trajectory_weight: float = 0.0,
        joint_pos_weight: float = 0.0,
        joint_vel_weight: float = 0.0,
        condition_kinematic_weight: float = 0.0,
        condition_acceleration_weight: float = 0.0,
        conditioned_root_h1_weight: float = 0.0,
        conditioned_root_h2_weight: float = 0.0,
        stationary_support_velocity_weight: float = 0.0,
        condition_transition_excess_weight: float = 0.0,
        condition_transition_jerk_ceiling_weight: float = 0.0,
        condition_transition_secant_excess_weight: float = 0.0,
        condition_transition_residual_weight: float = 0.0,
        condition_transition_sobolev_weight: float = 0.0,
        condition_transition_root_sobolev_weight: float = 0.0,
        condition_transition_root_sobolev_orders: Tuple[int, ...] = (1, 2),
        condition_transition_root_anchor_weight: float = 0.0,
        condition_transition_root_h2_weight: float = 0.0,
        condition_transition_root_tail_weight: float = 0.0,
        condition_transition_root_tail_band_radius: int = 4,
        condition_transition_root_cvar_weight: float = 0.0,
        condition_transition_root_dynamics_ceiling_weight: float = 0.0,
        condition_transition_root_endpoint_weight: float = 0.0,
        condition_transition_root_endpoint_tail_weight: float = 0.0,
        condition_transition_root_waypoint_curvature_weight: float = 0.0,
        condition_transition_root_waypoint_tail_weight: float = 0.0,
        condition_transition_root_waypoint_edge_weight: float = 0.0,
        condition_transition_root_waypoint_edge_tail_fraction: float = 0.2,
        condition_transition_root_waypoint_basin_weight: float = 0.0,
        condition_transition_root_waypoint_h1_weight: float = 0.0,
        condition_transition_root_waypoint_h1_detach_inner: bool = False,
        fk_consistency_weight: float = 0.0,
        motion_dim: int = 198,
        joint_pos_warmup_steps: int = 0,
        joint_vel_warmup_steps: int = 0,
        fk_consistency_warmup_steps: int = 0,
        timestep_squared_weighting: bool = True,
    ):
        super().__init__()
        self.velocity_weight = float(velocity_weight)
        self.velocity_loss_reduction = velocity_loss_reduction
        self.condition_neighborhood_flow_weight = float(
            condition_neighborhood_flow_weight
        )
        self.condition_neighborhood_x1_weight = float(
            condition_neighborhood_x1_weight
        )
        self.condition_neighborhood_world_weight = float(
            condition_neighborhood_world_weight
        )
        self.root_trajectory_weight = float(root_trajectory_weight)
        self.joint_pos_weight = float(joint_pos_weight)
        self.joint_vel_weight = float(joint_vel_weight)
        self.condition_kinematic_weight = float(condition_kinematic_weight)
        self.condition_acceleration_weight = float(
            condition_acceleration_weight
        )
        self.conditioned_root_h1_weight = float(conditioned_root_h1_weight)
        self.conditioned_root_h2_weight = float(conditioned_root_h2_weight)
        self.stationary_support_velocity_weight = float(
            stationary_support_velocity_weight
        )
        self.condition_transition_excess_weight = float(
            condition_transition_excess_weight
        )
        self.condition_transition_jerk_ceiling_weight = float(
            condition_transition_jerk_ceiling_weight
        )
        self.condition_transition_secant_excess_weight = float(
            condition_transition_secant_excess_weight
        )
        self.condition_transition_residual_weight = float(
            condition_transition_residual_weight
        )
        self.condition_transition_sobolev_weight = float(
            condition_transition_sobolev_weight
        )
        self.condition_transition_root_sobolev_weight = float(
            condition_transition_root_sobolev_weight
        )
        self.condition_transition_root_sobolev_orders = tuple(
            int(order) for order in condition_transition_root_sobolev_orders
        )
        self.condition_transition_root_anchor_weight = float(
            condition_transition_root_anchor_weight
        )
        self.condition_transition_root_h2_weight = float(
            condition_transition_root_h2_weight
        )
        self.condition_transition_root_tail_weight = float(
            condition_transition_root_tail_weight
        )
        self.condition_transition_root_tail_band_radius = int(
            condition_transition_root_tail_band_radius
        )
        self.condition_transition_root_cvar_weight = float(
            condition_transition_root_cvar_weight
        )
        self.condition_transition_root_dynamics_ceiling_weight = float(
            condition_transition_root_dynamics_ceiling_weight
        )
        self.condition_transition_root_endpoint_weight = float(
            condition_transition_root_endpoint_weight
        )
        self.condition_transition_root_endpoint_tail_weight = float(
            condition_transition_root_endpoint_tail_weight
        )
        self.condition_transition_root_waypoint_curvature_weight = float(
            condition_transition_root_waypoint_curvature_weight
        )
        self.condition_transition_root_waypoint_tail_weight = float(
            condition_transition_root_waypoint_tail_weight
        )
        self.condition_transition_root_waypoint_edge_weight = float(
            condition_transition_root_waypoint_edge_weight
        )
        self.condition_transition_root_waypoint_edge_tail_fraction = float(
            condition_transition_root_waypoint_edge_tail_fraction
        )
        self.condition_transition_root_waypoint_basin_weight = float(
            condition_transition_root_waypoint_basin_weight
        )
        self.condition_transition_root_waypoint_h1_weight = float(
            condition_transition_root_waypoint_h1_weight
        )
        self.condition_transition_root_waypoint_h1_detach_inner = bool(
            condition_transition_root_waypoint_h1_detach_inner
        )
        self.fk_consistency_weight = float(fk_consistency_weight)
        self.motion_dim = int(motion_dim)
        self.joint_pos_warmup_steps = int(joint_pos_warmup_steps)
        self.joint_vel_warmup_steps = int(joint_vel_warmup_steps)
        self.fk_consistency_warmup_steps = int(fk_consistency_warmup_steps)
        self.timestep_squared_weighting = bool(timestep_squared_weighting)

        if velocity_loss_reduction not in self._VALID_REDUCTIONS:
            raise ValueError(
                f"velocity_loss_reduction must be one of {self._VALID_REDUCTIONS}, "
                f"got {velocity_loss_reduction!r}"
            )
        if self.root_trajectory_weight < 0.0:
            raise ValueError("root_trajectory_weight must be non-negative")
        if self.condition_neighborhood_flow_weight < 0.0:
            raise ValueError(
                "condition_neighborhood_flow_weight must be non-negative"
            )
        if self.condition_neighborhood_x1_weight < 0.0:
            raise ValueError(
                "condition_neighborhood_x1_weight must be non-negative"
            )
        if self.condition_neighborhood_world_weight < 0.0:
            raise ValueError(
                "condition_neighborhood_world_weight must be non-negative"
            )
        if self.condition_kinematic_weight < 0.0:
            raise ValueError("condition_kinematic_weight must be non-negative")
        if self.condition_acceleration_weight < 0.0:
            raise ValueError(
                "condition_acceleration_weight must be non-negative"
            )
        if self.conditioned_root_h1_weight < 0.0:
            raise ValueError("conditioned_root_h1_weight must be non-negative")
        if self.conditioned_root_h2_weight < 0.0:
            raise ValueError("conditioned_root_h2_weight must be non-negative")
        if self.stationary_support_velocity_weight < 0.0:
            raise ValueError(
                "stationary_support_velocity_weight must be non-negative"
            )
        if self.condition_transition_excess_weight < 0.0:
            raise ValueError(
                "condition_transition_excess_weight must be non-negative"
            )
        if self.condition_transition_jerk_ceiling_weight < 0.0:
            raise ValueError(
                "condition_transition_jerk_ceiling_weight must be non-negative"
            )
        if self.condition_transition_secant_excess_weight < 0.0:
            raise ValueError(
                "condition_transition_secant_excess_weight must be non-negative"
            )
        if self.condition_transition_residual_weight < 0.0:
            raise ValueError(
                "condition_transition_residual_weight must be non-negative"
            )
        if self.condition_transition_sobolev_weight < 0.0:
            raise ValueError(
                "condition_transition_sobolev_weight must be non-negative"
            )
        if self.condition_transition_root_sobolev_weight < 0.0:
            raise ValueError(
                "condition_transition_root_sobolev_weight must be non-negative"
            )
        if self.condition_transition_root_anchor_weight < 0.0:
            raise ValueError(
                "condition_transition_root_anchor_weight must be non-negative"
            )
        if self.condition_transition_root_h2_weight < 0.0:
            raise ValueError(
                "condition_transition_root_h2_weight must be non-negative"
            )
        if self.condition_transition_root_tail_weight < 0.0:
            raise ValueError(
                "condition_transition_root_tail_weight must be non-negative"
            )
        if self.condition_transition_root_tail_band_radius < 0:
            raise ValueError(
                "condition_transition_root_tail_band_radius must be non-negative"
            )
        if self.condition_transition_root_cvar_weight < 0.0:
            raise ValueError(
                "condition_transition_root_cvar_weight must be non-negative"
            )
        if self.condition_transition_root_dynamics_ceiling_weight < 0.0:
            raise ValueError(
                "condition_transition_root_dynamics_ceiling_weight must be "
                "non-negative"
            )
        if self.condition_transition_root_endpoint_weight < 0.0:
            raise ValueError(
                "condition_transition_root_endpoint_weight must be non-negative"
            )
        if self.condition_transition_root_endpoint_tail_weight < 0.0:
            raise ValueError(
                "condition_transition_root_endpoint_tail_weight must be "
                "non-negative"
            )
        if self.condition_transition_root_waypoint_curvature_weight < 0.0:
            raise ValueError(
                "condition_transition_root_waypoint_curvature_weight must be "
                "non-negative"
            )
        if self.condition_transition_root_waypoint_tail_weight < 0.0:
            raise ValueError(
                "condition_transition_root_waypoint_tail_weight must be "
                "non-negative"
            )
        if self.condition_transition_root_waypoint_edge_weight < 0.0:
            raise ValueError(
                "condition_transition_root_waypoint_edge_weight must be "
                "non-negative"
            )
        if not (
            0.0
            < self.condition_transition_root_waypoint_edge_tail_fraction
            <= 1.0
        ):
            raise ValueError(
                "condition_transition_root_waypoint_edge_tail_fraction must "
                "be in (0, 1]"
            )
        if self.condition_transition_root_waypoint_basin_weight < 0.0:
            raise ValueError(
                "condition_transition_root_waypoint_basin_weight must be "
                "non-negative"
            )
        if self.condition_transition_root_waypoint_h1_weight < 0.0:
            raise ValueError(
                "condition_transition_root_waypoint_h1_weight must be "
                "non-negative"
            )
        if (
            not self.condition_transition_root_sobolev_orders
            or len(set(self.condition_transition_root_sobolev_orders))
            != len(self.condition_transition_root_sobolev_orders)
            or any(
                order not in (1, 2, 3)
                for order in self.condition_transition_root_sobolev_orders
            )
        ):
            raise ValueError(
                "condition_transition_root_sobolev_orders must be a non-empty "
                "sequence of unique values from {1, 2, 3}"
            )
        if (
            (
                self.stationary_support_velocity_weight > 0.0
                or self.condition_transition_excess_weight > 0.0
                or self.condition_transition_jerk_ceiling_weight > 0.0
                or self.condition_transition_secant_excess_weight > 0.0
            )
            and self.joint_vel_weight <= 0.0
        ):
            raise ValueError(
                "condition velocity auxiliaries require joint_vel_weight > 0"
            )
        if loss_type == "smooth_l1":
            self.loss_fn = F.smooth_l1_loss
        elif loss_type == "l1":
            self.loss_fn = F.l1_loss
        elif loss_type in ("mse", "l2"):
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    @property
    def geometry_enabled(self) -> bool:
        """Whether decoded-space inputs and differentiable FK are required."""
        return (
            self.condition_neighborhood_x1_weight > 0.0
            or self.condition_neighborhood_world_weight > 0.0
            or self.root_trajectory_weight > 0.0
            or self.conditioned_root_h1_weight > 0.0
            or self.conditioned_root_h2_weight > 0.0
            or self.joint_pos_weight > 0.0
            or self.joint_vel_weight > 0.0
            or self.stationary_support_velocity_weight > 0.0
            or self.condition_transition_excess_weight > 0.0
            or self.condition_transition_jerk_ceiling_weight > 0.0
            or self.condition_transition_secant_excess_weight > 0.0
            or self.condition_transition_residual_weight > 0.0
            or self.condition_transition_sobolev_weight > 0.0
            or self.condition_transition_root_sobolev_weight > 0.0
            or self.condition_transition_root_anchor_weight > 0.0
            or self.condition_transition_root_h2_weight > 0.0
            or self.condition_transition_root_tail_weight > 0.0
            or self.condition_transition_root_cvar_weight > 0.0
            or self.condition_transition_root_dynamics_ceiling_weight > 0.0
            or self.condition_transition_root_endpoint_weight > 0.0
            or self.condition_transition_root_endpoint_tail_weight > 0.0
            or self.condition_transition_root_waypoint_curvature_weight > 0.0
            or self.condition_transition_root_waypoint_tail_weight > 0.0
            or self.condition_transition_root_waypoint_edge_weight > 0.0
            or self.condition_transition_root_waypoint_basin_weight > 0.0
            or self.condition_transition_root_waypoint_h1_weight > 0.0
            or self.fk_consistency_weight > 0.0
        )

    @staticmethod
    def _motion_components(dim: int):
        """Semantic channel ranges used by component-mean flow reduction."""
        if dim >= 198:
            return ((0, 3), (3, 9), (9, 135), (135, 198))
        if dim >= 135:
            return ((0, 3), (3, 9), (9, 135))
        if dim == 38:
            return ((0, 3), (3, 9), (9, 38))
        return ((0, dim),)

    @staticmethod
    def _component_names(dim: int):
        """Stable diagnostic names corresponding to ``_motion_components``."""
        if dim >= 198:
            return ("trans", "root_rot", "body_rot", "joint_pos")
        if dim >= 135:
            return ("trans", "root_rot", "body_rot")
        if dim == 38:
            return ("trans", "root_rot", "joint")
        return ("all",)

    def _uses_modality_mean(self) -> bool:
        return self.velocity_loss_reduction in self._MODALITY_MEAN_REDUCTIONS

    def _flow_loss_with_components(
        self,
        per_dim: Tensor,
        data_mask_temporal: Tensor,
        generation_mask: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Reduce per-coordinate flow loss and return detached diagnostics."""
        data_mask = data_mask_temporal.to(per_dim.device, per_dim.dtype)
        if generation_mask is not None:
            generation_mask = generation_mask.to(per_dim.device, per_dim.dtype)
        if not self._uses_modality_mean():
            mask = data_mask.unsqueeze(-1).expand_as(per_dim)
            if generation_mask is not None:
                mask = mask * generation_mask
            total = (per_dim * mask).sum() / torch.clamp(mask.sum(), min=1.0)
            return total, {}

        component_values: Dict[str, Tensor] = {}
        for (start, end), name in zip(
            self._motion_components(per_dim.shape[-1]),
            self._component_names(per_dim.shape[-1]),
        ):
            component = per_dim[..., start:end]
            mask = data_mask.unsqueeze(-1).expand_as(component)
            if generation_mask is not None:
                mask = mask * generation_mask[..., start:end].to(
                    per_dim.device,
                    per_dim.dtype,
                )
            if torch.gt(mask.sum().detach(), 0):
                component_values[name] = (
                    (component * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                )
        if not component_values:
            return per_dim.sum() * 0.0, {}
        total = torch.stack(list(component_values.values())).mean()
        return total, component_values

    def _condition_neighborhood_flow_loss(
        self,
        per_dim: Tensor,
        data_mask_temporal: Tensor,
        generation_mask: Tensor,
        radii: Tuple[int, ...] = (1, 2, 4),
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Reweight exact flow targets near evidence on the same channel.

        Hard imputation makes isolated evidence exact, but its generated
        neighbors occupy only a tiny fraction of the ordinary flow objective.
        This method temporally dilates each *individual* known channel and
        reduces the generated coordinates inside that neighborhood separately.
        It does not map translation evidence to body channels, construct a
        derivative target, or prescribe a benchmark-specific mask pattern.

        Cumulative radii deliberately count the closest neighbors at every
        scale. Thus the immediate join receives more importance than frames
        four steps away, while every active sample and semantic component gets
        equal reduction weight independent of clip length or condition density.
        Padding is excluded before dilation so it cannot create a false
        condition boundary at the end of a clip.
        """
        if per_dim.shape != generation_mask.shape:
            raise ValueError(
                "per_dim and generation_mask must have identical shapes, got "
                f"{tuple(per_dim.shape)} and {tuple(generation_mask.shape)}"
            )
        if data_mask_temporal.shape != per_dim.shape[:2]:
            raise ValueError(
                "data_mask_temporal must match per_dim batch/time axes, got "
                f"{tuple(data_mask_temporal.shape)} and {tuple(per_dim.shape[:2])}"
            )

        valid = data_mask_temporal.to(
            device=per_dim.device,
            dtype=torch.bool,
        )[..., None]
        generation = generation_mask.to(
            device=per_dim.device,
            dtype=per_dim.dtype,
        ) > 0.5
        known = (~generation) & valid
        known_channels = known.permute(0, 2, 1).to(per_dim.dtype)

        scale_losses = []
        component_sums = {
            name: per_dim.sum() * 0.0
            for name in self._component_names(per_dim.shape[-1])
        }
        component_counts = {
            name: per_dim.new_zeros(())
            for name in self._component_names(per_dim.shape[-1])
        }
        for radius in radii:
            if radius <= 0:
                raise ValueError("condition-neighborhood radii must be positive")
            dilated_known = F.max_pool1d(
                known_channels,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).permute(0, 2, 1) > 0.5
            neighborhood = generation & valid & dilated_known

            values = []
            active_components = []
            for (start, end), name in zip(
                self._motion_components(per_dim.shape[-1]),
                self._component_names(per_dim.shape[-1]),
            ):
                component_mask = neighborhood[..., start:end]
                value = self._batch_masked_mean(
                    per_dim[..., start:end],
                    component_mask,
                )
                active = (component_mask.sum() > 0).to(per_dim.dtype)
                values.append(value)
                active_components.append(active)
                component_sums[name] = component_sums[name] + value * active
                component_counts[name] = component_counts[name] + active

            values_tensor = torch.stack(values)
            active_tensor = torch.stack(active_components)
            scale_losses.append(
                (values_tensor * active_tensor).sum()
                / active_tensor.sum().clamp_min(1.0)
            )

        if not scale_losses:
            zero = per_dim.sum() * 0.0
            return zero, {name: zero for name in component_sums}
        components = {
            name: component_sums[name]
            / component_counts[name].clamp_min(1.0)
            for name in component_sums
        }
        return torch.stack(scale_losses).mean(), components

    def _condition_neighborhood_x1_loss(
        self,
        pred_x1: Tensor,
        gt_x1: Tensor,
        data_mask_temporal: Tensor,
        generation_mask: Tensor,
        radii: Tuple[int, ...] = (1, 2, 4),
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match the clean endpoint near evidence on the same channel.

        ``pred_x1`` must already contain the exact clean condition at known
        coordinates. Only generated neighbors are selected, so this objective
        cannot learn a shortcut by copying or changing the hard-imputed point.
        """
        if pred_x1.shape != gt_x1.shape:
            raise ValueError(
                "pred_x1 and gt_x1 must have identical shapes, got "
                f"{tuple(pred_x1.shape)} and {tuple(gt_x1.shape)}"
            )
        pointwise = self.loss_fn(pred_x1, gt_x1, reduction="none")
        return self._condition_neighborhood_flow_loss(
            pointwise,
            data_mask_temporal,
            generation_mask,
            radii=radii,
        )

    def _condition_neighborhood_world_loss(
        self,
        pred_world: Tensor,
        gt_world: Tensor,
        data_mask_temporal: Tensor,
        generation_mask: Tensor,
        radii: Tuple[int, ...] = (0, 1, 2, 4),
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match clean world joints where motion evidence can affect them.

        The mask mapping follows kinematics rather than representation width:
        a sparse root-X observation selects every joint's world-X coordinate,
        while a wrist-position atom remains local. Radius zero also supervises
        generated body channels at a partially observed frame. Fully observed
        frames already decode to the clean target after condition projection
        and therefore contribute zero without a special case.
        """
        if pred_world.shape != gt_world.shape:
            raise ValueError(
                "pred_world and gt_world must have identical shapes, got "
                f"{tuple(pred_world.shape)} and {tuple(gt_world.shape)}"
            )
        if generation_mask.shape[:2] != pred_world.shape[:2]:
            raise ValueError(
                "generation_mask and world positions must share batch/time axes"
            )
        valid = data_mask_temporal.to(
            device=pred_world.device,
            dtype=torch.bool,
        )[..., None]
        known = (generation_mask[..., :198] < 0.5) & valid
        support = self._condition_channels_to_world(known) > 0.5
        support_channels = support.permute(0, 2, 3, 1).reshape(
            pred_world.shape[0],
            -1,
            pred_world.shape[1],
        ).to(pred_world.dtype)
        pointwise = self.loss_fn(pred_world, gt_world, reduction="none")
        valid_world = valid[..., None].expand_as(pointwise)

        scale_losses = []
        diagnostics: Dict[str, Tensor] = {}
        for radius in radii:
            if radius < 0:
                raise ValueError("condition-neighborhood radii must be non-negative")
            if radius == 0:
                dilated = support
            else:
                dilated = F.max_pool1d(
                    support_channels,
                    kernel_size=2 * radius + 1,
                    stride=1,
                    padding=radius,
                ).reshape(
                    pred_world.shape[0],
                    pred_world.shape[2],
                    pred_world.shape[3],
                    pred_world.shape[1],
                ).permute(0, 3, 1, 2) > 0.5
            value = self._batch_masked_mean(
                pointwise,
                dilated & valid_world,
            )
            scale_losses.append(value)
            diagnostics[f"r{radius}"] = value

        if not scale_losses:
            zero = pointwise.sum() * 0.0
            return zero, {}
        return torch.stack(scale_losses).mean(), diagnostics

    @staticmethod
    def _warmup(weight: float, steps: int, global_step: Optional[int]) -> float:
        """Linearly ramp a scalar weight from zero to its configured value."""
        if weight == 0.0 or steps <= 0 or global_step is None:
            return weight
        return weight * min(1.0, float(global_step) / float(steps))

    @staticmethod
    def _align_generation_mask(
        generation_mask: Optional[Tensor],
        *,
        length: int,
        dim: int,
    ) -> Optional[Tensor]:
        """Align a target-only generation mask with the decoded clean motion."""
        if generation_mask is None:
            return None
        mask = generation_mask
        if mask.shape[-1] != dim:
            raise ValueError(
                "generation_mask feature dimension must match pred_x1, got "
                f"{mask.shape[-1]} and {dim}"
            )
        if mask.shape[1] < length:
            # Reference poses prepended by the model are known evidence.
            mask = F.pad(mask, (0, 0, length - mask.shape[1], 0), value=0.0)
        elif mask.shape[1] > length:
            mask = mask[:, -length:]
        return mask

    @staticmethod
    def _condition_channels_to_world(supported: Tensor) -> Tensor:
        """Map supported 198-D channels to affected world-joint axes."""
        if supported.shape[-1] < 198:
            raise ValueError(
                "condition kinematic mapping requires the 198-D layout"
            )
        supported = supported[..., :198]
        batch, edges = supported.shape[:2]
        weights = torch.zeros(
            batch,
            edges,
            22,
            3,
            device=supported.device,
            dtype=torch.float32,
        )

        translation = supported[..., :3].to(weights.dtype)
        weights = torch.maximum(weights, translation.unsqueeze(-2))

        rotation = supported[..., 3:135].reshape(batch, edges, 22, 6)
        rotation = rotation.any(dim=-1).to(weights.dtype)
        descendant_matrix = torch.as_tensor(
            _SMPL22_DESCENDANT_MATRIX,
            device=weights.device,
            dtype=weights.dtype,
        )
        rotation_world = torch.einsum(
            "bes,sj->bej", rotation, descendant_matrix
        ).clamp_max_(1.0)
        weights = torch.maximum(weights, rotation_world.unsqueeze(-1))

        position = supported[..., 135:198].reshape(batch, edges, 21, 3)
        weights[:, :, 1:, :] = torch.maximum(
            weights[:, :, 1:, :],
            position.to(weights.dtype),
        )
        return weights

    @classmethod
    def _condition_kinematic_map(cls, generation_mask: Tensor) -> Tensor:
        """Map every condition-supported edge to world-joint coordinates.

        An edge is supported when either endpoint contains a known coordinate.
        This includes both sides of an isolated waypoint and every edge of a
        dense trajectory. Translation axes affect every joint on the same
        world axis, a local rotation affects its kinematic descendants, and a
        pelvis-relative position atom affects its corresponding joint-axis.
        """
        known = generation_mask[..., :198] < 0.5
        return cls._condition_channels_to_world(known[:, 1:] | known[:, :-1])

    @classmethod
    def _condition_transition_map(cls, generation_mask: Tensor) -> Tensor:
        """Map only known/generated state changes to world-joint axes.

        XOR excludes known-known span interiors and fully dense conditions.
        Isolated evidence still activates both adjacent edges. The definition
        is coordinate-local, so heterogeneous frame/joint/axis masks require
        no task-specific clauses.
        """
        known = generation_mask[..., :198] < 0.5
        return cls._condition_channels_to_world(known[:, 1:] ^ known[:, :-1])

    @staticmethod
    def _batch_masked_mean(pointwise: Tensor, mask: Tensor) -> Tensor:
        """Average each active sample independently, then average the batch.

        Per-sample reduction prevents long clips, dense observations, or XYZ
        evidence from silently outweighing short clips and single-axis evidence.
        """
        mask = mask.to(device=pointwise.device, dtype=pointwise.dtype)
        reduce_dims = tuple(range(1, pointwise.ndim))
        numerator = (pointwise * mask).sum(dim=reduce_dims)
        denominator = mask.sum(dim=reduce_dims)
        sample_mean = numerator / denominator.clamp_min(1.0)
        active = (denominator > 0).to(pointwise.dtype)
        # Keep the reduction entirely on device.  A Python ``torch.any`` check
        # synchronizes every rank and is especially costly for condition losses
        # that call this helper several times per step.  With no active sample,
        # both numerator and active_count are zero, so this still returns a
        # differentiable zero with exactly the previous semantics.
        return (sample_mean * active).sum() / active.sum().clamp_min(1.0)

    @classmethod
    def _condition_transition_excess_loss(
        cls,
        pointwise: Tensor,
        transition_map: Tensor,
        condition_support_map: Tensor,
        valid_temporal: Tensor,
    ) -> Tensor:
        """Penalize boundary error above each joint-axis background level.

        The detached baseline is computed per sample and world joint-axis over
        non-transition time entries. This makes the target relative to the
        model's current ordinary-region accuracy instead of forcing every
        legitimate action transition toward zero acceleration. Axes without a
        non-transition reference, such as a fully dense trajectory axis, are
        excluded rather than assigned an arbitrary target.
        """
        if (
            pointwise.shape != transition_map.shape
            or pointwise.shape != condition_support_map.shape
        ):
            raise ValueError(
                "pointwise, transition_map, and condition_support_map must "
                "have identical shapes"
            )
        valid = (valid_temporal > 0.5)[..., None, None].expand_as(pointwise)
        transition = (transition_map > 0.5) & valid
        condition_supported = (condition_support_map > 0.5) & valid
        # Known-known interiors are exact after hard projection and would make
        # an artificially easy baseline. Compare only against fully generated
        # entries of the same joint-axis.
        background = (~condition_supported) & valid

        background_float = background.to(pointwise.dtype)
        background_count = background_float.sum(dim=1, keepdim=True)
        background_mean = (
            (pointwise * background_float).sum(dim=1, keepdim=True)
            / background_count.clamp_min(1.0)
        ).detach()

        comparable = transition & (background_count > 0)
        excess = torch.relu(pointwise - background_mean)
        return cls._batch_masked_mean(excess, comparable)

    def _condition_transition_jerk_ceiling_loss(
        self,
        pred_jerk: Tensor,
        gt_jerk: Tensor,
        transition_map: Tensor,
        condition_support_map: Tensor,
        valid_temporal: Tensor,
        timestep_weight: Optional[Tensor],
    ) -> Tensor:
        """Suppress only artificial jerk above a motion-adaptive ceiling.

        The generated-background mean defines the ordinary prediction scale for
        each sample and world joint-axis. The detached GT jerk at the transition
        raises that ceiling when the action contains a legitimate sharp change.
        Unlike direct jerk reconstruction, values below the ceiling receive no
        gradient, so this term cannot encourage the model to reproduce mocap
        high-frequency noise.
        """
        if (
            pred_jerk.shape != gt_jerk.shape
            or pred_jerk.shape != transition_map.shape
            or pred_jerk.shape != condition_support_map.shape
        ):
            raise ValueError(
                "jerk tensors, transition_map, and condition_support_map must "
                "have identical shapes"
            )

        valid = (valid_temporal > 0.5)[..., None, None].expand_as(pred_jerk)
        transition = (transition_map > 0.5) & valid
        condition_supported = (condition_support_map > 0.5) & valid
        background = (~condition_supported) & valid

        pred_magnitude = pred_jerk.abs()
        background_float = background.to(pred_jerk.dtype)
        background_count = background_float.sum(dim=1, keepdim=True)
        background_mean = (
            (pred_magnitude * background_float).sum(dim=1, keepdim=True)
            / background_count.clamp_min(1.0)
        ).detach()
        ceiling = torch.maximum(background_mean, gt_jerk.detach().abs())
        overshoot = torch.relu(pred_magnitude - ceiling)
        pointwise = self.loss_fn(
            overshoot,
            torch.zeros_like(overshoot),
            reduction="none",
        )
        if timestep_weight is not None:
            pointwise = pointwise * timestep_weight[:, None, None, None]

        comparable = transition & (background_count > 0)
        return self._batch_masked_mean(pointwise, comparable)

    def _condition_transition_secant_excess_loss(
        self,
        pred_world: Tensor,
        gt_world: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        timestep_weight: Optional[Tensor],
        horizons: Tuple[int, ...] = (1, 2, 4),
    ) -> Tensor:
        """Match low-order motion across condition boundaries at several scales.

        For each horizon, exactly one endpoint must be condition-supported. The
        endpoint secant velocity is matched to GT only when its error exceeds
        the detached error of the same joint-axis in fully generated windows.
        Looking over 1, 2, and 4 frames constrains both the immediate join and
        its local trend while avoiding acceleration or jerk reconstruction.
        """
        zero = pred_world.sum() * 0.0
        if pred_world.shape != gt_world.shape:
            raise ValueError("pred_world and gt_world must have identical shapes")
        if generation_mask.shape[:2] != pred_world.shape[:2]:
            raise ValueError(
                "generation_mask and world positions must share batch/time axes"
            )

        known_world = self._condition_channels_to_world(
            generation_mask[..., :198] < 0.5
        ) > 0.5
        scale_losses = []
        for horizon in horizons:
            if horizon <= 0:
                raise ValueError("secant horizons must be positive")
            if pred_world.shape[1] <= horizon:
                continue

            pred_secant = (
                pred_world[:, horizon:] - pred_world[:, :-horizon]
            ) / float(horizon)
            gt_secant = (
                gt_world[:, horizon:] - gt_world[:, :-horizon]
            ) / float(horizon)
            pointwise = self.loss_fn(
                pred_secant,
                gt_secant,
                reduction="none",
            )
            if timestep_weight is not None:
                pointwise = pointwise * timestep_weight[:, None, None, None]

            left_known = known_world[:, :-horizon]
            right_known = known_world[:, horizon:]
            transition = left_known ^ right_known
            window_support = torch.stack(
                [
                    known_world[
                        :, offset : offset + pred_secant.shape[1]
                    ]
                    for offset in range(horizon + 1)
                ],
                dim=0,
            ).any(dim=0)
            valid_pair = data_mask[:, horizon:] * data_mask[:, :-horizon]
            scale_losses.append(
                self._condition_transition_excess_loss(
                    pointwise,
                    transition,
                    window_support,
                    valid_pair,
                )
            )

        if not scale_losses:
            return zero
        return torch.stack(scale_losses).mean()

    @staticmethod
    def _finite_difference(value: Tensor, order: int) -> Tensor:
        """Return a forward temporal difference without changing other axes."""
        if order == 1:
            return value[:, 1:] - value[:, :-1]
        if order == 2:
            return value[:, 2:] - 2.0 * value[:, 1:-1] + value[:, :-2]
        if order == 3:
            return (
                value[:, 3:]
                - 3.0 * value[:, 2:-1]
                + 3.0 * value[:, 1:-2]
                - value[:, :-3]
            )
        raise ValueError(f"finite-difference order must be 1, 2, or 3, got {order}")

    @staticmethod
    def _dilate_temporal_support(support: Tensor, radius: int) -> Tensor:
        """Dilate a ``(B,T,J,A)`` boolean support along time only."""
        if radius < 0:
            raise ValueError("temporal dilation radius must be non-negative")
        if radius == 0 or support.shape[1] == 0:
            return support
        batch, length, joints, axes = support.shape
        channels = support.permute(0, 2, 3, 1).reshape(
            batch,
            joints * axes,
            length,
        ).to(torch.float32)
        dilated = F.max_pool1d(
            channels,
            kernel_size=2 * radius + 1,
            stride=1,
            padding=radius,
        )
        return (
            dilated.reshape(batch, joints, axes, length).permute(0, 3, 1, 2)
            > 0.5
        )

    def _condition_transition_residual_loss(
        self,
        pred_world: Tensor,
        gt_world: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        timestep_weight: Optional[Tensor],
        band_radius: int = 2,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match local error derivatives to ordinary-region accuracy.

        Let ``e = pred_world - gt_world`` after hard condition projection. A
        visible join is exactly a large temporal derivative of ``e`` around a
        known/generated transition. For derivative orders one through three,
        this objective applies an L1-strength penalty only to residual above
        the detached mean residual of the same sample and joint-axis in fully
        generated background windows. The transition support is dilated by two
        frames so the model must repair the local approach and departure, not
        merely the hard-imputed coordinate itself.

        Dense conditions and T2M contain no known/generated transition and
        therefore receive a differentiable zero. Long known spans are excluded
        from the background, preventing exact projected interiors from making
        the parity target artificially easy.
        """
        if pred_world.shape != gt_world.shape:
            raise ValueError("pred_world and gt_world must have identical shapes")
        if generation_mask.shape[:2] != pred_world.shape[:2]:
            raise ValueError(
                "generation_mask and world positions must share batch/time axes"
            )

        zero = pred_world.sum() * 0.0
        known_world = self._condition_channels_to_world(
            generation_mask[..., :198] < 0.5
        ) > 0.5
        edge_transition = known_world[:, 1:] ^ known_world[:, :-1]
        error = pred_world - gt_world
        order_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for order in (1, 2, 3):
            if error.shape[1] <= order:
                continue
            length = error.shape[1] - order
            residual = self._finite_difference(error, order).abs()
            if timestep_weight is not None:
                residual = residual * timestep_weight[:, None, None, None]

            transition = torch.stack(
                [
                    edge_transition[:, offset : offset + length]
                    for offset in range(order)
                ],
                dim=0,
            ).any(dim=0)
            band = self._dilate_temporal_support(transition, band_radius)
            window_known = torch.stack(
                [
                    known_world[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            )
            condition_window = window_known.any(dim=0)
            fully_known_window = window_known.all(dim=0)
            valid_window = torch.stack(
                [
                    data_mask[:, offset : offset + length] > 0.5
                    for offset in range(order + 1)
                ],
                dim=0,
            ).all(dim=0)[..., None, None]
            valid = valid_window.expand_as(residual)
            band = band & (~fully_known_window) & valid
            background = (~band) & (~condition_window) & valid

            background_float = background.to(residual.dtype)
            background_count = background_float.sum(dim=1, keepdim=True)
            background_mean = (
                (residual * background_float).sum(dim=1, keepdim=True)
                / background_count.clamp_min(1.0)
            ).detach()
            comparable = band & (background_count > 0)
            excess = torch.relu(residual - background_mean)
            value = self._batch_masked_mean(excess, comparable)
            order_losses.append(value)
            diagnostics[f"d{order}"] = value

        if not order_losses:
            return zero, {}
        return torch.stack(order_losses).mean(), diagnostics

    def _condition_transition_sobolev_parity_loss(
        self,
        pred_world: Tensor,
        gt_world: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        timestep_weight: Optional[Tensor],
        band_radius: int = 2,
        scale_floor_m: float = 1e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match local derivative error to ordinary generated-region error.

        The absolute residual of a mature motion model is small enough that a
        metre-space Smooth-L1 objective has a vanishing gradient. This objective
        instead optimizes the dimensionless ratio

        ``log(1 + max(|Delta^k(pred-gt)| / stopgrad(background_k) - 1, 0))``

        for derivative orders ``k=1,2,3``. ``background_k`` is computed per
        sample and world joint-axis over fully generated windows outside the
        transition band. Translation, root rotation, body rotation, and joint
        position evidence are reduced independently before taking their mean.
        This prevents a sparse physical modality from being diluted by the many
        world coordinates affected by another modality in a heterogeneous mask.
        The logarithm limits outlier influence while the ratio preserves a useful
        gradient at millimetre scale. The objective is zero for T2M, fully dense
        conditions, padding, and boundaries already no worse than ordinary
        generated motion.
        """
        if pred_world.shape != gt_world.shape:
            raise ValueError("pred_world and gt_world must have identical shapes")
        if generation_mask.shape[:2] != pred_world.shape[:2]:
            raise ValueError(
                "generation_mask and world positions must share batch/time axes"
            )
        if scale_floor_m <= 0.0:
            raise ValueError("scale_floor_m must be positive")

        zero = pred_world.sum() * 0.0
        known_channels = generation_mask[..., :198] < 0.5
        all_known_world = self._condition_channels_to_world(known_channels) > 0.5
        component_known_worlds = []
        for start, end in self._motion_components(198):
            component_channels = torch.zeros_like(known_channels)
            component_channels[..., start:end] = known_channels[..., start:end]
            component_known_worlds.append(
                self._condition_channels_to_world(component_channels) > 0.5
            )
        error = pred_world - gt_world
        order_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for order in (1, 2, 3):
            if error.shape[1] <= order:
                continue
            length = error.shape[1] - order
            residual = self._finite_difference(error, order).abs()
            all_window_known = torch.stack(
                [
                    all_known_world[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            )
            condition_window = all_window_known.any(dim=0)
            valid_window = torch.stack(
                [
                    data_mask[:, offset : offset + length] > 0.5
                    for offset in range(order + 1)
                ],
                dim=0,
            ).all(dim=0)[..., None, None]
            valid = valid_window.expand_as(residual)
            component_bands = []
            for component_known_world in component_known_worlds:
                edge_transition = (
                    component_known_world[:, 1:]
                    ^ component_known_world[:, :-1]
                )
                transition = torch.stack(
                    [
                        edge_transition[:, offset : offset + length]
                        for offset in range(order)
                    ],
                    dim=0,
                ).any(dim=0)
                band = self._dilate_temporal_support(transition, band_radius)
                component_window_known = torch.stack(
                    [
                        component_known_world[:, offset : offset + length]
                        for offset in range(order + 1)
                    ],
                    dim=0,
                )
                component_bands.append(
                    band & (~component_window_known.all(dim=0)) & valid
                )
            all_band = torch.stack(component_bands, dim=0).any(dim=0)
            # Other evidence types must not leak their transition residual into
            # the ordinary-motion reference of the current component.
            background = (~all_band) & (~condition_window) & valid
            background_float = background.to(residual.dtype)
            background_count = background_float.sum(dim=1, keepdim=True)
            background_scale = (
                (residual * background_float).sum(dim=1, keepdim=True)
                / background_count.clamp_min(1.0)
            ).detach().clamp_min(scale_floor_m)
            relative_excess = torch.relu(residual / background_scale - 1.0)
            pointwise = torch.log1p(relative_excess)
            if timestep_weight is not None:
                pointwise = pointwise * timestep_weight[:, None, None, None]

            component_sum = zero
            component_count = zero
            for band in component_bands:
                comparable = band & (background_count > 0)
                component_value = self._batch_masked_mean(
                    pointwise,
                    comparable,
                )
                component_active = (
                    comparable.to(residual.dtype).sum() > 0
                ).to(residual.dtype)
                component_sum = component_sum + component_active * component_value
                component_count = component_count + component_active

            value = component_sum / component_count.clamp_min(1.0)
            order_losses.append(value)
            diagnostics[f"d{order}"] = value

        if not order_losses:
            return zero, {}
        return torch.stack(order_losses).mean(), diagnostics

    def _condition_transition_root_sobolev_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        orders: Tuple[int, ...] = (1, 2),
        band_radius: int = 2,
        scale_floor_m: float = 5e-4,
        tail_rms: bool = False,
        tail_fraction: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match sparse root joins to ordinary root-completion accuracy.

        World-joint losses cannot isolate translation: FK adds root translation
        and generated body-relative motion, so the latter can cancel or dilute a
        root discontinuity. This objective instead differentiates the projected
        clean-endpoint error ``pred_root - gt_root`` in native metric coordinates.

        For each configured derivative order, it compares a two-frame band around
        known/generated root-axis transitions with the detached error of ordinary
        generated windows from the same sample and axis. A one-sided Smooth-L1
        penalty is zero at or below parity and has constant gradient for large
        spikes. Unlike decoded-space losses, it intentionally has no ``t^2``
        multiplier: endpoint scaling is already cancelled by the relative error.
        ``tail_fraction`` switches the temporal reduction to CVaR: only the
        largest configured fraction is averaged, while the same Smooth-L1
        pointwise penalty keeps every selected outlier gradient bounded.

        The support is derived only from the coordinate mask. T2M, body-only
        conditions, and fully dense root axes therefore return an exact zero.
        When a highly fragmented mask leaves no ordinary window, the physical
        scale floor supplies a conservative reference instead of disabling the
        loss precisely on the hardest mask.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if scale_floor_m <= 0.0:
            raise ValueError("scale_floor_m must be positive")
        if tail_fraction is not None and not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if tail_rms and tail_fraction is not None:
            raise ValueError("tail_rms and tail_fraction are mutually exclusive")
        if not orders or len(set(orders)) != len(orders) or any(
            order not in (1, 2, 3) for order in orders
        ):
            raise ValueError(
                "orders must be a non-empty sequence of unique values from {1, 2, 3}"
            )

        zero = pred_root.sum() * 0.0
        valid_frame = data_mask > 0.5
        known = (generation_mask[..., :3] < 0.5) & valid_frame[..., None]
        edge_transition = known[:, 1:] ^ known[:, :-1]
        error = pred_root - gt_root
        order_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for order in orders:
            if error.shape[1] <= order:
                continue
            length = error.shape[1] - order
            residual = self._finite_difference(error, order).abs()
            transition = torch.stack(
                [
                    edge_transition[:, offset : offset + length]
                    for offset in range(order)
                ],
                dim=0,
            ).any(dim=0)
            band = self._dilate_temporal_support(
                transition[:, :, None, :],
                band_radius,
            )[:, :, 0, :]
            window_known = torch.stack(
                [
                    known[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            )
            condition_window = window_known.any(dim=0)
            fully_known_window = window_known.all(dim=0)
            valid_window = torch.stack(
                [
                    valid_frame[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            ).all(dim=0)[..., None]
            valid = valid_window.expand_as(residual)
            band = band & (~fully_known_window) & valid
            background = (~band) & (~condition_window) & valid

            background_float = background.to(residual.dtype)
            background_count = background_float.sum(dim=1, keepdim=True)
            background_mean = (
                (residual * background_float).sum(dim=1, keepdim=True)
                / background_count.clamp_min(1.0)
            ).detach()
            background_scale = torch.where(
                background_count > 0,
                background_mean,
                torch.full_like(background_mean, scale_floor_m),
            ).clamp_min(scale_floor_m)
            relative_excess = torch.relu(residual / background_scale - 1.0)
            if tail_rms:
                # Sparse-condition artifacts are a tail problem: a few large
                # joins dominate visual jitter while a temporal mean is diluted
                # by easier neighboring windows. Squaring before the masked
                # reduction and taking an RMS below keeps the statistic scale
                # normalized while increasing each window's gradient in
                # proportion to its excess above ordinary-motion accuracy.
                pointwise = relative_excess.square()
            else:
                pointwise = F.smooth_l1_loss(
                    relative_excess,
                    torch.zeros_like(relative_excess),
                    reduction="none",
                    beta=1.0,
                )

            # Average every active sample-axis independently. XZ evidence must
            # not receive twice the optimization weight of a single-axis hint.
            mask = band.to(pointwise.dtype)
            numerator = (pointwise * mask).sum(dim=1)
            denominator = mask.sum(dim=1)
            axis_mean = numerator / denominator.clamp_min(1.0)
            if tail_rms:
                axis_mean = torch.sqrt(axis_mean + 1e-12) - 1e-6
            elif tail_fraction is not None:
                # CVaR selects a fixed fraction of the worst active windows per
                # sample-axis. Smooth-L1 above keeps every selected outlier's
                # gradient bounded; unlike RMS, a 100x spike cannot dominate a
                # merely visible one. Sorting is differentiable for the chosen
                # values and the rank mask carries no gradient by design.
                axis_values = pointwise.permute(0, 2, 1).reshape(
                    -1, pointwise.shape[1]
                )
                axis_mask = band.permute(0, 2, 1).reshape(
                    -1, band.shape[1]
                )
                active_count = axis_mask.sum(dim=1)
                tail_count = torch.ceil(
                    active_count.to(pointwise.dtype) * tail_fraction
                ).to(torch.long).clamp_min(1)
                sorted_values = axis_values.masked_fill(
                    ~axis_mask,
                    torch.finfo(pointwise.dtype).min,
                ).sort(dim=1, descending=True).values
                ranks = torch.arange(
                    sorted_values.shape[1], device=sorted_values.device
                )[None, :]
                selected = (ranks < tail_count[:, None]) & (
                    ranks < active_count[:, None]
                )
                axis_mean = torch.where(
                    selected,
                    sorted_values,
                    torch.zeros_like(sorted_values),
                ).sum(dim=1) / tail_count.to(pointwise.dtype)
                axis_mean = axis_mean.reshape(pointwise.shape[0], -1)
            active_axis = (denominator > 0).to(pointwise.dtype)
            value = (
                (axis_mean * active_axis).sum()
                / active_axis.sum().clamp_min(1.0)
            )
            order_losses.append(value)
            diagnostics[f"d{order}"] = value

        if not order_losses:
            return zero, {}
        return torch.stack(order_losses).mean(), diagnostics

    def _condition_transition_root_dynamics_ceiling_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        orders: Tuple[int, ...] = (2, 3),
        scale_floor_m: float = 5e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Suppress predicted root dynamics that exceed a local motion ceiling.

        Error-derivative objectives can reduce their shared parameters by making
        the entire trajectory smoother while leaving a few hard-imputation
        spikes intact. This term instead measures the predicted root derivative
        itself on stencils that cross a known/generated boundary. Its detached
        ceiling is ``max(abs(GT derivative), ordinary prediction magnitude)``
        for the same sample-axis. Consequently, genuine action dynamics and
        already ordinary joins receive zero gradient; only artificial excess is
        optimized. The ratio form keeps millimetre-scale spikes trainable and
        Smooth-L1 bounds each selected window's gradient.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if scale_floor_m <= 0.0:
            raise ValueError("scale_floor_m must be positive")
        if not orders or len(set(orders)) != len(orders) or any(
            order not in (2, 3) for order in orders
        ):
            raise ValueError(
                "orders must be a non-empty sequence of unique values from {2, 3}"
            )

        zero = pred_root.sum() * 0.0
        valid_frame = data_mask > 0.5
        known = (generation_mask[..., :3] < 0.5) & valid_frame[..., None]
        edge_transition = known[:, 1:] ^ known[:, :-1]
        order_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for order in orders:
            if pred_root.shape[1] <= order:
                continue
            length = pred_root.shape[1] - order
            pred_derivative = self._finite_difference(pred_root, order)
            gt_derivative = self._finite_difference(gt_root, order)
            transition = torch.stack(
                [
                    edge_transition[:, offset : offset + length]
                    for offset in range(order)
                ],
                dim=0,
            ).any(dim=0)
            window_known = torch.stack(
                [
                    known[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            )
            condition_window = window_known.any(dim=0)
            fully_known_window = window_known.all(dim=0)
            valid_window = torch.stack(
                [
                    valid_frame[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            ).all(dim=0)[..., None]
            valid = valid_window.expand_as(pred_derivative)
            transition = transition & (~fully_known_window) & valid
            background = (~condition_window) & valid

            pred_magnitude = pred_derivative.abs()
            background_float = background.to(pred_magnitude.dtype)
            background_count = background_float.sum(dim=1, keepdim=True)
            background_mean = (
                (pred_magnitude * background_float).sum(dim=1, keepdim=True)
                / background_count.clamp_min(1.0)
            ).detach()
            background_scale = torch.where(
                background_count > 0,
                background_mean,
                torch.full_like(background_mean, scale_floor_m),
            ).clamp_min(scale_floor_m)
            ceiling = torch.maximum(
                background_scale,
                gt_derivative.detach().abs(),
            )
            relative_overshoot = torch.relu(pred_magnitude / ceiling - 1.0)
            pointwise = F.smooth_l1_loss(
                relative_overshoot,
                torch.zeros_like(relative_overshoot),
                reduction="none",
                beta=1.0,
            )

            mask = transition.to(pointwise.dtype)
            numerator = (pointwise * mask).sum(dim=1)
            denominator = mask.sum(dim=1)
            axis_mean = numerator / denominator.clamp_min(1.0)
            active_axis = (denominator > 0).to(pointwise.dtype)
            value = (
                (axis_mean * active_axis).sum()
                / active_axis.sum().clamp_min(1.0)
            )
            order_losses.append(value)
            diagnostics[f"d{order}"] = value

        if not order_losses:
            return zero, {}
        return torch.stack(order_losses).mean(), diagnostics

    def _condition_transition_root_endpoint_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        radii: Tuple[int, ...] = (1, 2, 4),
        huber_beta_m: float = 2e-3,
        tail_fraction: Optional[float] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Recover clean root endpoints immediately around sparse evidence.

        A hard-projected waypoint is exact at the observed frame. Therefore a
        discontinuity can only remain when one of its generated neighbours is
        inaccurate. This objective directly supervises those generated root
        coordinates against the clean training trajectory in metres. Cumulative
        radii make the immediate neighbour appear at every scale, while frames
        farther away receive progressively less weight.

        Unlike acceleration or jerk regularization, the target is the original
        motion rather than a smoothness prior. The small Huber beta keeps a
        bounded, nearly constant gradient for visible millimetre-to-centimetre
        errors. When ``tail_fraction`` is set, the reduction independently
        averages the worst fraction of active endpoints for each sample-axis.
        This prevents the many sub-millimetre neighbours from hiding a small
        number of visible joins. T2M, body-only evidence, fully observed root
        axes, padding, and known coordinates themselves all receive exactly
        zero gradient.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if not radii or any(radius <= 0 for radius in radii):
            raise ValueError("radii must be a non-empty sequence of positives")
        if huber_beta_m <= 0.0:
            raise ValueError("huber_beta_m must be positive")
        if tail_fraction is not None and not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")

        valid = (data_mask > 0.5)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        known_channels = known.permute(0, 2, 1).to(pred_root.dtype)
        pointwise = F.smooth_l1_loss(
            pred_root,
            gt_root,
            reduction="none",
            beta=huber_beta_m,
        )

        scale_losses = []
        diagnostics: Dict[str, Tensor] = {}
        for radius in radii:
            dilated_known = F.max_pool1d(
                known_channels,
                kernel_size=2 * radius + 1,
                stride=1,
                padding=radius,
            ).permute(0, 2, 1) > 0.5
            neighborhood = generated & dilated_known
            if tail_fraction is None:
                value = self._batch_masked_mean(pointwise, neighborhood)
            else:
                # Rank endpoints independently per sample-axis. Averaging axes
                # within each sample first keeps X, XZ, and XYZ evidence from
                # changing a sample's effective optimization weight.
                axis_values = pointwise.permute(0, 2, 1)
                axis_mask = neighborhood.permute(0, 2, 1)
                active_count = axis_mask.sum(dim=-1)
                tail_count = torch.ceil(
                    active_count.to(pointwise.dtype) * tail_fraction
                ).to(torch.long).clamp_min(1)
                sorted_values = axis_values.masked_fill(
                    ~axis_mask,
                    torch.finfo(pointwise.dtype).min,
                ).sort(dim=-1, descending=True).values
                ranks = torch.arange(
                    sorted_values.shape[-1], device=sorted_values.device
                )[None, None, :]
                selected = (ranks < tail_count[..., None]) & (
                    ranks < active_count[..., None]
                )
                axis_mean = torch.where(
                    selected,
                    sorted_values,
                    torch.zeros_like(sorted_values),
                ).sum(dim=-1) / tail_count.to(pointwise.dtype)
                active_axis = active_count > 0
                sample_mean = (
                    axis_mean * active_axis.to(pointwise.dtype)
                ).sum(dim=-1) / active_axis.sum(dim=-1).clamp_min(1).to(
                    pointwise.dtype
                )
                active_sample = active_axis.any(dim=-1)
                value = (
                    sample_mean * active_sample.to(pointwise.dtype)
                ).sum() / active_sample.sum().clamp_min(1).to(pointwise.dtype)
            scale_losses.append(value)
            diagnostics[f"r{radius}"] = value

        return torch.stack(scale_losses).mean(), diagnostics

    @staticmethod
    def _condition_transition_root_waypoint_curvature_loss(
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        huber_beta_m_per_frame2: float = 2e-3,
    ) -> Tensor:
        """Match clean root curvature at isolated observed coordinates.

        Hard projection makes an observed root coordinate exact, but the two
        generated neighbours can still approach it with a mismatched local
        trajectory. For every ``generated-known-generated`` stencil on each
        translation axis, this term matches the projected second difference to
        the clean motion's second difference. The clean target preserves real
        accelerations; this is not a zero-curvature smoothness prior.

        Reduction first averages active stencils per sample-axis, then axes per
        sample, then active samples. Thus X, XZ, and XYZ evidence have equal
        sample weight. T2M, body-only evidence, dense root trajectories,
        condition-span interiors, one-sided boundaries, and padding are zero.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if huber_beta_m_per_frame2 <= 0.0:
            raise ValueError("huber_beta_m_per_frame2 must be positive")
        if pred_root.shape[1] < 3:
            return pred_root.sum() * 0.0

        valid = (data_mask > 0.5)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        isolated = (
            generated[:, :-2]
            & known[:, 1:-1]
            & generated[:, 2:]
        )
        projected_root = torch.where(generated, pred_root, gt_root)

        pred_curvature = (
            projected_root[:, 2:]
            - 2.0 * projected_root[:, 1:-1]
            + projected_root[:, :-2]
        )
        gt_curvature = (
            gt_root[:, 2:]
            - 2.0 * gt_root[:, 1:-1]
            + gt_root[:, :-2]
        )
        pointwise = F.smooth_l1_loss(
            pred_curvature,
            gt_curvature,
            reduction="none",
            beta=huber_beta_m_per_frame2,
        )

        mask = isolated.permute(0, 2, 1).to(pointwise.dtype)
        values = pointwise.permute(0, 2, 1)
        count = mask.sum(dim=-1)
        axis_mean = (values * mask).sum(dim=-1) / count.clamp_min(1.0)
        active_axis = count > 0
        sample_mean = (
            axis_mean * active_axis.to(pointwise.dtype)
        ).sum(dim=-1) / active_axis.sum(dim=-1).clamp_min(1).to(
            pointwise.dtype
        )
        active_sample = active_axis.any(dim=-1)
        return (
            sample_mean * active_sample.to(pointwise.dtype)
        ).sum() / active_sample.sum().clamp_min(1).to(pointwise.dtype)

    @staticmethod
    def _condition_transition_root_waypoint_tail_loss(
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        tail_fraction: float = 0.2,
        pseudo_huber_beta_m_per_frame2: float = 2e-3,
    ) -> Tensor:
        """Optimize worst vector-curvature errors at sparse root waypoints.

        The benchmark measures the Euclidean XZ/XYZ curvature magnitude at an
        isolated hard observation. Selecting tails independently per axis can
        miss a jointly bad vector, so this term first forms the active-axis
        curvature-error norm at each waypoint. A vector pseudo-Huber penalty
        has a unit-bounded gradient for large errors and a smooth quadratic
        basin near zero. CVaR then averages the worst fixed fraction per sample,
        preventing rare visible joins from being diluted by easy waypoints.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if pseudo_huber_beta_m_per_frame2 <= 0.0:
            raise ValueError(
                "pseudo_huber_beta_m_per_frame2 must be positive"
            )
        if pred_root.shape[1] < 3:
            return pred_root.sum() * 0.0

        valid = (data_mask > 0.5)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        isolated = (
            generated[:, :-2]
            & known[:, 1:-1]
            & generated[:, 2:]
        )
        active_frame = isolated.any(dim=-1)
        projected_root = torch.where(generated, pred_root, gt_root)
        error_curvature = (
            projected_root[:, 2:]
            - 2.0 * projected_root[:, 1:-1]
            + projected_root[:, :-2]
            - gt_root[:, 2:]
            + 2.0 * gt_root[:, 1:-1]
            - gt_root[:, :-2]
        )
        squared_norm = (
            error_curvature.square() * isolated.to(error_curvature.dtype)
        ).sum(dim=-1)
        beta = pseudo_huber_beta_m_per_frame2
        pointwise = torch.sqrt(squared_norm + beta * beta) - beta

        active_count = active_frame.sum(dim=-1)
        tail_count = torch.ceil(
            active_count.to(pointwise.dtype) * tail_fraction
        ).to(torch.long).clamp_min(1)
        sorted_values = pointwise.masked_fill(
            ~active_frame,
            torch.finfo(pointwise.dtype).min,
        ).sort(dim=-1, descending=True).values
        ranks = torch.arange(
            sorted_values.shape[-1], device=sorted_values.device
        )[None, :]
        selected = (ranks < tail_count[:, None]) & (
            ranks < active_count[:, None]
        )
        sample_tail = torch.where(
            selected,
            sorted_values,
            torch.zeros_like(sorted_values),
        ).sum(dim=-1) / tail_count.to(pointwise.dtype)
        active_sample = active_count > 0
        return (
            sample_tail * active_sample.to(pointwise.dtype)
        ).sum() / active_sample.sum().clamp_min(1).to(pointwise.dtype)

    @staticmethod
    def _condition_transition_root_waypoint_edge_loss(
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        timesteps: Tensor,
        tail_fraction: float = 0.2,
        pseudo_huber_beta_m: float = 2e-3,
        endpoint_scale_floor: float = 1e-3,
    ) -> Tensor:
        """Recover both one-sided root edges at isolated observations.

        With a hard-projected observation at frame ``i``, clean curvature error
        is ``e[i-1] + e[i+1]``. Opposite endpoint errors can therefore cancel
        even though both visible velocity joins are wrong. This objective keeps
        the two sides separate and forms an active-axis X/XZ/XYZ vector before
        reducing the worst edges per sample.

        Reconstructed endpoint error satisfies
        ``pred_x1 - x1 = (1-t) * (pred_v - v)`` on generated coordinates.
        Dividing by ``1-t`` makes this a flow-equivalent error and prevents the
        auxiliary gradient from vanishing near the clean endpoint. Known axes
        are projected to zero error first, so they remain gradient-free. The GT
        trajectory is still the unique optimum; no zero-velocity or smoothing
        target is introduced.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if timesteps.numel() != pred_root.shape[0]:
            raise ValueError("timesteps must contain one value per sample")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if pseudo_huber_beta_m <= 0.0:
            raise ValueError("pseudo_huber_beta_m must be positive")
        if endpoint_scale_floor <= 0.0:
            raise ValueError("endpoint_scale_floor must be positive")
        if pred_root.shape[1] < 3:
            return pred_root.sum() * 0.0

        valid = (data_mask > 0.5)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        isolated = (
            generated[:, :-2]
            & known[:, 1:-1]
            & generated[:, 2:]
        )

        endpoint_scale = (
            1.0
            - timesteps.to(device=pred_root.device, dtype=pred_root.dtype).reshape(
                pred_root.shape[0], 1, 1
            )
        ).clamp_min(endpoint_scale_floor)
        projected_error = torch.where(
            generated,
            pred_root - gt_root,
            torch.zeros_like(pred_root),
        ) / endpoint_scale
        edge_error = projected_error[:, 1:] - projected_error[:, :-1]
        side_errors = torch.stack(
            (edge_error[:, :-1], edge_error[:, 1:]),
            dim=2,
        )

        active_edge = isolated.any(dim=-1)[..., None].expand(-1, -1, 2)
        squared_norm = (
            side_errors.square() * isolated[..., None, :].to(side_errors.dtype)
        ).sum(dim=-1)
        beta = pseudo_huber_beta_m
        pointwise = torch.sqrt(squared_norm + beta * beta) - beta

        flat_values = pointwise.flatten(start_dim=1)
        flat_active = active_edge.flatten(start_dim=1)
        active_count = flat_active.sum(dim=-1)
        tail_count = torch.ceil(
            active_count.to(pointwise.dtype) * tail_fraction
        ).to(torch.long).clamp_min(1)
        sorted_values = flat_values.masked_fill(
            ~flat_active,
            torch.finfo(pointwise.dtype).min,
        ).sort(dim=-1, descending=True).values
        ranks = torch.arange(
            sorted_values.shape[-1], device=sorted_values.device
        )[None, :]
        selected = (ranks < tail_count[:, None]) & (
            ranks < active_count[:, None]
        )
        sample_tail = torch.where(
            selected,
            sorted_values,
            torch.zeros_like(sorted_values),
        ).sum(dim=-1) / tail_count.to(pointwise.dtype)
        active_sample = active_count > 0
        return (
            sample_tail * active_sample.to(pointwise.dtype)
        ).sum() / active_sample.sum().clamp_min(1).to(pointwise.dtype)

    @staticmethod
    def _condition_transition_root_waypoint_basin_loss(
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        timesteps: Tensor,
        radii: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
        tail_fraction: float = 0.2,
        pseudo_huber_beta_m: float = 2e-3,
        endpoint_scale_floor: float = 1e-3,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Propagate a sparse root observation into a clean local basin.

        Hard projection guarantees the observed coordinate itself but cannot
        make nearby generated frames use that observation. For each sparse
        root coordinate, this objective directly fits generated coordinates at
        multiple distances on either side. An endpoint is active only when
        every intervening coordinate on the same axis is generated, so dense
        spans and another observation inside the path are excluded.

        Endpoint errors are divided by ``1-t`` to recover metric flow error.
        Active X/XZ/XYZ axes form one vector, the worst endpoints are selected
        independently at every radius, and inverse-radius averaging emphasizes
        the visible join while still teaching longer-range propagation. The
        target is always the original GT trajectory, never a smoothed or
        zero-velocity surrogate.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if timesteps.numel() != pred_root.shape[0]:
            raise ValueError("timesteps must contain one value per sample")
        if not radii or len(set(radii)) != len(radii) or any(
            radius <= 0 for radius in radii
        ):
            raise ValueError("radii must be unique positive integers")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if pseudo_huber_beta_m <= 0.0:
            raise ValueError("pseudo_huber_beta_m must be positive")
        if endpoint_scale_floor <= 0.0:
            raise ValueError("endpoint_scale_floor must be positive")

        zero = pred_root.sum() * 0.0
        valid = (data_mask > 0.5)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        endpoint_scale = (
            1.0
            - timesteps.to(device=pred_root.device, dtype=pred_root.dtype).reshape(
                pred_root.shape[0], 1, 1
            )
        ).clamp_min(endpoint_scale_floor)
        flow_error = torch.where(
            generated,
            pred_root - gt_root,
            torch.zeros_like(pred_root),
        ) / endpoint_scale

        weighted_values = []
        active_weights = []
        diagnostics: Dict[str, Tensor] = {}
        sequence_length = pred_root.shape[1]
        for radius in radii:
            if sequence_length <= 2 * radius:
                continue
            center_known = known[:, radius:-radius]
            left_path_generated = torch.stack(
                [
                    generated[
                        :,
                        radius - offset : sequence_length - radius - offset,
                    ]
                    for offset in range(1, radius + 1)
                ],
                dim=0,
            ).all(dim=0)
            right_path_generated = torch.stack(
                [
                    generated[
                        :,
                        radius + offset : sequence_length - radius + offset,
                    ]
                    for offset in range(1, radius + 1)
                ],
                dim=0,
            ).all(dim=0)
            side_mask = torch.stack(
                (
                    center_known & left_path_generated,
                    center_known & right_path_generated,
                ),
                dim=2,
            )
            side_errors = torch.stack(
                (
                    flow_error[:, : -2 * radius],
                    flow_error[:, 2 * radius :],
                ),
                dim=2,
            )
            squared_norm = (
                side_errors.square() * side_mask.to(side_errors.dtype)
            ).sum(dim=-1)
            beta = pseudo_huber_beta_m
            pointwise = torch.sqrt(squared_norm + beta * beta) - beta
            active_endpoint = side_mask.any(dim=-1)

            flat_values = pointwise.flatten(start_dim=1)
            flat_active = active_endpoint.flatten(start_dim=1)
            active_count = flat_active.sum(dim=-1)
            tail_count = torch.ceil(
                active_count.to(pointwise.dtype) * tail_fraction
            ).to(torch.long).clamp_min(1)
            sorted_values = flat_values.masked_fill(
                ~flat_active,
                torch.finfo(pointwise.dtype).min,
            ).sort(dim=-1, descending=True).values
            ranks = torch.arange(
                sorted_values.shape[-1], device=sorted_values.device
            )[None, :]
            selected = (ranks < tail_count[:, None]) & (
                ranks < active_count[:, None]
            )
            sample_tail = torch.where(
                selected,
                sorted_values,
                torch.zeros_like(sorted_values),
            ).sum(dim=-1) / tail_count.to(pointwise.dtype)
            active_sample = active_count > 0
            value = (
                sample_tail * active_sample.to(pointwise.dtype)
            ).sum() / active_sample.sum().clamp_min(1).to(pointwise.dtype)
            diagnostics[f"r{radius}"] = value

            radius_weight = 1.0 / float(radius)
            weighted_values.append(radius_weight * value)
            active_weights.append(
                radius_weight * active_sample.any().to(pointwise.dtype)
            )

        if not weighted_values:
            return zero, {}
        return (
            torch.stack(weighted_values).sum()
            / torch.stack(active_weights).sum().clamp_min(1e-12),
            diagnostics,
        )

    @staticmethod
    def _condition_transition_root_waypoint_h1_loss(
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        timesteps: Tensor,
        radii: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8),
        tail_fraction: float = 0.2,
        tail_mix: float = 0.5,
        pseudo_huber_beta_m_per_frame: float = 2e-3,
        endpoint_scale_floor: float = 1e-3,
        detach_inner: bool = False,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match every local residual edge around isolated root waypoints.

        Let ``e = (pred_x1 - x1) / (1-t)`` on generated coordinates and zero
        on observed coordinates. The first difference ``e[j] - e[j-1]`` is
        exactly the metric flow error of the reconstructed root velocity. At an
        isolated observation ``e[i] = 0``; matching consecutive residual edges
        therefore propagates the clean anchor through its local generated
        neighbourhood instead of fitting each frame independently.

        Radius one is deliberately left to the separate waypoint-edge loss.
        This term covers the generated-generated edges at radii two through
        eight. It activates only when both immediate neighbours of the anchor
        and the complete path to the selected edge are generated on the same
        axis. Consequently temporal prefix/suffix boundaries, dense root
        conditions, body-only conditions, and inactive axes receive no gradient.
        Each radius blends its mean with the worst-20-percent tail, then all
        active radii are averaged equally so a distant edge cannot be hidden by
        an easier near-anchor edge. With ``detach_inner=True``, only the outer
        endpoint of each residual edge receives gradient. The exact anchor is
        then propagated outward without allowing a distant error to drag the
        near-anchor prediction away from its independently supervised target.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if timesteps.numel() != pred_root.shape[0]:
            raise ValueError("timesteps must contain one value per sample")
        if not radii or len(set(radii)) != len(radii) or any(
            radius < 2 for radius in radii
        ):
            raise ValueError("radii must be unique integers greater than one")
        if not 0.0 < tail_fraction <= 1.0:
            raise ValueError("tail_fraction must be in (0, 1]")
        if not 0.0 <= tail_mix <= 1.0:
            raise ValueError("tail_mix must be in [0, 1]")
        if pseudo_huber_beta_m_per_frame <= 0.0:
            raise ValueError("pseudo_huber_beta_m_per_frame must be positive")
        if endpoint_scale_floor <= 0.0:
            raise ValueError("endpoint_scale_floor must be positive")

        zero = pred_root.sum() * 0.0
        sequence_length = pred_root.shape[1]
        if sequence_length < 3:
            return zero, {}

        valid = (data_mask > 0.5)[..., None]
        generated = (generation_mask[..., :3] > 0.5) & valid
        known = (~generated) & valid
        isolated = torch.zeros_like(known)
        isolated[:, 1:-1] = (
            known[:, 1:-1] & generated[:, :-2] & generated[:, 2:]
        )

        endpoint_scale = (
            1.0
            - timesteps.to(device=pred_root.device, dtype=pred_root.dtype).reshape(
                pred_root.shape[0], 1, 1
            )
        ).clamp_min(endpoint_scale_floor)
        flow_error = torch.where(
            generated,
            pred_root - gt_root,
            torch.zeros_like(pred_root),
        ) / endpoint_scale
        residual_edges = flow_error[:, 1:] - flow_error[:, :-1]

        radius_values = []
        radius_weights = []
        diagnostics: Dict[str, Tensor] = {}
        for radius in radii:
            center_count = sequence_length - 2 * radius
            if center_count <= 0:
                continue
            center_isolated = isolated[:, radius : sequence_length - radius]
            left_path_generated = torch.stack(
                [
                    generated[
                        :,
                        radius - offset : sequence_length - radius - offset,
                    ]
                    for offset in range(1, radius + 1)
                ],
                dim=0,
            ).all(dim=0)
            right_path_generated = torch.stack(
                [
                    generated[
                        :,
                        radius + offset : sequence_length - radius + offset,
                    ]
                    for offset in range(1, radius + 1)
                ],
                dim=0,
            ).all(dim=0)
            side_mask = torch.stack(
                (
                    center_isolated & left_path_generated,
                    center_isolated & right_path_generated,
                ),
                dim=2,
            )
            if detach_inner:
                left_outer = flow_error[:, :center_count]
                left_inner = flow_error[:, 1 : center_count + 1]
                right_inner = flow_error[:, 2 * radius - 1 : -1]
                right_outer = flow_error[:, 2 * radius :]
                side_errors = torch.stack(
                    (
                        left_inner.detach() - left_outer,
                        right_outer - right_inner.detach(),
                    ),
                    dim=2,
                )
            else:
                side_errors = torch.stack(
                    (
                        residual_edges[:, :center_count],
                        residual_edges[:, 2 * radius - 1 :],
                    ),
                    dim=2,
                )
            squared_norm = (
                side_errors.square() * side_mask.to(side_errors.dtype)
            ).sum(dim=-1)
            beta = pseudo_huber_beta_m_per_frame
            pointwise = torch.sqrt(squared_norm + beta * beta) - beta
            active_edge = side_mask.any(dim=-1)

            flat_values = pointwise.flatten(start_dim=1)
            flat_active = active_edge.flatten(start_dim=1)
            active_count = flat_active.sum(dim=-1)
            sample_mean = (
                (flat_values * flat_active.to(flat_values.dtype)).sum(dim=-1)
                / active_count.clamp_min(1).to(flat_values.dtype)
            )
            tail_count = torch.ceil(
                active_count.to(pointwise.dtype) * tail_fraction
            ).to(torch.long).clamp_min(1)
            sorted_values = flat_values.masked_fill(
                ~flat_active,
                torch.finfo(pointwise.dtype).min,
            ).sort(dim=-1, descending=True).values
            ranks = torch.arange(
                sorted_values.shape[-1], device=sorted_values.device
            )[None, :]
            selected = (ranks < tail_count[:, None]) & (
                ranks < active_count[:, None]
            )
            sample_tail = torch.where(
                selected,
                sorted_values,
                torch.zeros_like(sorted_values),
            ).sum(dim=-1) / tail_count.to(pointwise.dtype)
            sample_value = (1.0 - tail_mix) * sample_mean + tail_mix * sample_tail
            active_sample = active_count > 0
            value = (
                sample_value * active_sample.to(pointwise.dtype)
            ).sum() / active_sample.sum().clamp_min(1).to(pointwise.dtype)
            diagnostics[f"r{radius}"] = value
            radius_active = active_sample.any().to(pointwise.dtype)
            radius_values.append(radius_active * value)
            radius_weights.append(radius_active)

        if not radius_values:
            return zero, {}
        return (
            torch.stack(radius_values).sum()
            / torch.stack(radius_weights).sum().clamp_min(1.0),
            diagnostics,
        )

    def _condition_transition_root_anchor_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        lags: Tuple[int, ...] = (1, 2, 4),
        scale_floor_m_per_frame: float = 5e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Match sparse root observations to nearby generated trajectory.

        Hard imputation makes the endpoint error exactly zero on an observed
        coordinate. A visible waypoint jump is therefore the secant error from
        that anchor to a nearby generated frame. For each lag ``s`` this term
        directly compares

        ``((pred[t+s]-pred[t]) - (gt[t+s]-gt[t])) / s``

        on known/generated endpoint pairs. The reference scale is the detached
        secant error of fully generated windows from the same sample and root
        axis. Optimizing only excess above that reference asks condition joins to
        be as continuous as ordinary motion without suppressing legitimate GT
        velocity or inventing a blending curve. Lags 1, 2, and 4 constrain the
        immediate join and its approach band without a noisy third derivative.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if (
            not lags
            or len(set(lags)) != len(lags)
            or any(lag <= 0 for lag in lags)
        ):
            raise ValueError("lags must be a non-empty sequence of unique positives")
        if scale_floor_m_per_frame <= 0.0:
            raise ValueError("scale_floor_m_per_frame must be positive")

        zero = pred_root.sum() * 0.0
        valid_frame = data_mask > 0.5
        known = (generation_mask[..., :3] < 0.5) & valid_frame[..., None]
        error = pred_root - gt_root
        scale_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for lag in lags:
            if error.shape[1] <= lag:
                continue
            length = error.shape[1] - lag
            valid_pair = (
                valid_frame[:, lag:] & valid_frame[:, :-lag]
            )[..., None]
            left_known = known[:, :-lag]
            right_known = known[:, lag:]
            anchor_pair = (left_known ^ right_known) & valid_pair

            secant_error = (
                (error[:, lag:] - error[:, :-lag]).abs() / float(lag)
            )
            window_known = torch.stack(
                [
                    known[:, offset : offset + length]
                    for offset in range(lag + 1)
                ],
                dim=0,
            ).any(dim=0)
            background = (~window_known) & valid_pair
            background_float = background.to(secant_error.dtype)
            background_count = background_float.sum(dim=1, keepdim=True)
            background_mean = (
                (secant_error * background_float).sum(dim=1, keepdim=True)
                / background_count.clamp_min(1.0)
            ).detach()
            background_scale = torch.where(
                background_count > 0,
                background_mean,
                torch.full_like(background_mean, scale_floor_m_per_frame),
            ).clamp_min(scale_floor_m_per_frame)

            relative_excess = torch.relu(secant_error / background_scale - 1.0)
            pointwise = F.smooth_l1_loss(
                relative_excess,
                torch.zeros_like(relative_excess),
                reduction="none",
                beta=1.0,
            )

            # Reduce every active sample-axis independently, then every lag.
            # Thus XZ evidence, XYZ evidence, and clips with many waypoints have
            # equal optimization weight rather than rewarding denser masks.
            mask = anchor_pair.to(pointwise.dtype)
            numerator = (pointwise * mask).sum(dim=1)
            denominator = mask.sum(dim=1)
            axis_mean = numerator / denominator.clamp_min(1.0)
            active_axis = (denominator > 0).to(pointwise.dtype)
            value = (
                (axis_mean * active_axis).sum()
                / active_axis.sum().clamp_min(1.0)
            )
            scale_losses.append(value)
            diagnostics[f"s{lag}"] = value

        if not scale_losses:
            return zero, {}
        return torch.stack(scale_losses).mean(), diagnostics

    def _conditioned_root_h1_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        t_sq: Optional[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        """Match GT root paths on axes that occur as motion evidence.

        The H1 objective contains position and first-difference residuals. An
        axis is selected solely from the coordinate mask: if it is observed at
        least once, every generated value on that axis belongs to its completion
        problem. This covers arbitrary densities and mixed per-frame axis masks
        without prescribing a line or spline between observations.
        """
        if pred_root.shape[-1] != 3 or gt_root.shape[-1] != 3:
            raise ValueError("root tensors must contain exactly three axes")

        valid = data_mask[..., None] > 0.5
        root_generation = generation_mask[..., :3] > 0.5
        conditioned_axis = ((~root_generation) & valid).any(dim=1, keepdim=True)
        completion_mask = root_generation & conditioned_axis & valid

        position_pointwise = self.loss_fn(
            pred_root,
            gt_root,
            reduction="none",
        )
        if t_sq is not None:
            position_pointwise = position_pointwise * t_sq[:, None, None]
        position = self._batch_masked_mean(
            position_pointwise,
            completion_mask,
        )

        if pred_root.shape[1] < 2:
            return position, position * 0.0
        pred_delta = pred_root[:, 1:] - pred_root[:, :-1]
        gt_delta = gt_root[:, 1:] - gt_root[:, :-1]
        delta_pointwise = self.loss_fn(
            pred_delta,
            gt_delta,
            reduction="none",
        )
        if t_sq is not None:
            delta_pointwise = delta_pointwise * t_sq[:, None, None]
        valid_edge = valid[:, 1:] & valid[:, :-1]
        generated_edge = root_generation[:, 1:] | root_generation[:, :-1]
        delta_mask = valid_edge & generated_edge & conditioned_axis
        velocity = self._batch_masked_mean(delta_pointwise, delta_mask)
        return position, velocity

    def _conditioned_root_h2_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        robust_epsilon_m: float = 1e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Fit a sparse-conditioned root completion through second order.

        A sparse hard observation is continuous only if the generated trajectory
        approaches it with the correct position, velocity, and curvature.  The
        previous parity objectives compared these errors with the model's own
        generated-region error, which is a moving target and leaves a non-zero
        jitter floor.  Here the reference is the clean motion itself:

        ``e = pred_root - gt_root`` and ``L_k = rho(Delta^k e), k in {0,1,2}``.

        ``rho`` is a Charbonnier penalty with a 0.1 mm transition, giving an
        L1-strength gradient at the millimetre-scale errors visible as waypoint
        jumps while remaining smooth at zero.  Every derivative order and every
        active sample-axis is reduced independently.  The objective spans the
        complete generated portion of a conditioned axis rather than only a
        hand-selected boundary band, and it deliberately has no timestep
        attenuation: the conversion from predicted flow to ``x1`` already scales
        its gradient by ``1-t``.

        An axis must contain at least one known and one generated valid frame.
        Therefore pure T2M, body-only evidence, padding, and fully observed dense
        trajectories contribute an exact differentiable zero.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if robust_epsilon_m <= 0.0:
            raise ValueError("robust_epsilon_m must be positive")

        valid = data_mask > 0.5
        valid_axis = valid[..., None]
        known = (generation_mask[..., :3] < 0.5) & valid_axis
        generated = (generation_mask[..., :3] > 0.5) & valid_axis
        completion_axis = (
            known.any(dim=1, keepdim=True)
            & generated.any(dim=1, keepdim=True)
        )
        error = pred_root - gt_root
        zero = error.sum() * 0.0
        order_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for order in (0, 1, 2):
            if error.shape[1] <= order:
                continue
            if order == 0:
                residual = error
                window_generated = generated
                valid_window = valid_axis
            else:
                length = error.shape[1] - order
                residual = self._finite_difference(error, order)
                window_generated = torch.stack(
                    [
                        generated[:, offset : offset + length]
                        for offset in range(order + 1)
                    ],
                    dim=0,
                ).any(dim=0)
                valid_window = torch.stack(
                    [
                        valid[:, offset : offset + length]
                        for offset in range(order + 1)
                    ],
                    dim=0,
                ).all(dim=0)[..., None]

            pointwise = torch.sqrt(
                residual.square() + robust_epsilon_m**2
            ) - robust_epsilon_m
            mask = window_generated & valid_window & completion_axis
            mask_float = mask.to(pointwise.dtype)
            numerator = (pointwise * mask_float).sum(dim=1)
            denominator = mask_float.sum(dim=1)
            axis_mean = numerator / denominator.clamp_min(1.0)
            active_axis = (denominator > 0).to(pointwise.dtype)
            value = (
                (axis_mean * active_axis).sum()
                / active_axis.sum().clamp_min(1.0)
            )
            order_losses.append(value)
            diagnostics[f"d{order}"] = value

        if not order_losses:
            return zero, {}
        return torch.stack(order_losses).mean(), diagnostics

    def _condition_transition_root_h2_loss(
        self,
        pred_root: Tensor,
        gt_root: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        band_radius: int = 2,
        robust_epsilon_m: float = 1e-4,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Fit GT root derivatives only around known/generated joins.

        Hard projection makes ``pred_root == gt_root`` on every observed root
        coordinate, but it does not teach adjacent generated frames to approach
        that observation continuously.  For each translation axis, this method
        finds every XOR transition between known and generated coordinates,
        dilates that support by ``band_radius`` derivative windows, and fits the
        first and second differences of the clean-endpoint error:

        ``e = pred_root - gt_root`` and ``L_k = rho(Delta^k e), k in {1, 2}``.

        The target is always the original clean motion, not a line, spline, or
        zero-velocity prior.  A Charbonnier penalty preserves an approximately
        constant gradient for visible millimetre-scale joins.  Every active
        sample-axis and derivative order has equal reduction weight, independent
        of condition density.  With no root transition (T2M, body-only evidence,
        or a fully observed dense trajectory), the loss is exactly zero.
        """
        if pred_root.shape != gt_root.shape or pred_root.shape[-1] != 3:
            raise ValueError(
                "pred_root and gt_root must have identical (B,T,3) shapes"
            )
        if generation_mask.shape[:2] != pred_root.shape[:2]:
            raise ValueError(
                "generation_mask and root translation must share batch/time axes"
            )
        if band_radius < 0:
            raise ValueError("band_radius must be non-negative")
        if robust_epsilon_m <= 0.0:
            raise ValueError("robust_epsilon_m must be positive")

        valid = data_mask > 0.5
        valid_axis = valid[..., None]
        known = (generation_mask[..., :3] < 0.5) & valid_axis
        generated = (generation_mask[..., :3] > 0.5) & valid_axis
        edge_transition = (
            (known[:, 1:] ^ known[:, :-1])
            & valid[:, 1:, None]
            & valid[:, :-1, None]
        )
        error = pred_root - gt_root
        zero = error.sum() * 0.0
        order_losses = []
        diagnostics: Dict[str, Tensor] = {}

        for order in (1, 2):
            if error.shape[1] <= order:
                continue
            length = error.shape[1] - order
            residual = self._finite_difference(error, order)

            # A k-th order window contains k adjacent temporal edges.  It is a
            # join window when any one of those edges crosses known/generated.
            transition = torch.stack(
                [
                    edge_transition[:, offset : offset + length]
                    for offset in range(order)
                ],
                dim=0,
            ).any(dim=0)
            band = self._dilate_temporal_support(
                transition[:, :, None, :],
                band_radius,
            )[:, :, 0, :]

            valid_window = torch.stack(
                [
                    valid[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            ).all(dim=0)[..., None]
            generated_window = torch.stack(
                [
                    generated[:, offset : offset + length]
                    for offset in range(order + 1)
                ],
                dim=0,
            ).any(dim=0)
            mask = band & valid_window & generated_window

            pointwise = torch.sqrt(
                residual.square() + robust_epsilon_m**2
            ) - robust_epsilon_m
            mask_float = mask.to(pointwise.dtype)
            numerator = (pointwise * mask_float).sum(dim=1)
            denominator = mask_float.sum(dim=1)
            axis_mean = numerator / denominator.clamp_min(1.0)
            active_axis = (denominator > 0).to(pointwise.dtype)
            value = (
                (axis_mean * active_axis).sum()
                / active_axis.sum().clamp_min(1.0)
            )
            order_losses.append(value)
            diagnostics[f"d{order}"] = value

        if not order_losses:
            return zero, {}
        return torch.stack(order_losses).mean(), diagnostics

    def _stationary_support_velocity_loss(
        self,
        pred_world: Tensor,
        gt_world: Tensor,
        generation_mask: Tensor,
        data_mask: Tensor,
        t_sq: Optional[Tensor],
    ) -> Tensor:
        """Emphasize slowly moving GT joints for every training task.

        This is a continuous, identity-free support prior. It never declares a
        particular foot, hand, knee, or torso joint to be in contact and never
        forces a legal nonzero velocity to zero. Instead, all 22 joints receive
        a detached weight based on their GT speed, and the prediction is matched
        to that GT world velocity. Applying the same prior to all tasks keeps the
        body/root coupling objective independent of the particular condition mask.
        """
        zero = pred_world.sum() * 0.0
        if pred_world.shape[1] < 2:
            return zero

        valid_frame = data_mask > 0.5
        valid_edge = valid_frame[:, 1:] & valid_frame[:, :-1]
        pred_velocity = pred_world[:, 1:] - pred_world[:, :-1]
        gt_velocity = gt_world[:, 1:] - gt_world[:, :-1]
        pointwise = self.loss_fn(
            pred_velocity,
            gt_velocity,
            reduction="none",
        )
        if t_sq is not None:
            pointwise = pointwise * t_sq[:, None, None, None]

        gt_speed = gt_velocity.norm(dim=-1, keepdim=True)
        valid_joint = valid_edge[..., None, None].expand_as(gt_speed)
        speed_sum = (gt_speed * valid_joint).sum(dim=(1, 2, 3), keepdim=True)
        speed_count = valid_joint.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1)
        speed_scale = (speed_sum / speed_count).detach().clamp_min(1e-4)
        slow_weight = 1.0 / (1.0 + (gt_speed / speed_scale).square())

        mask = valid_edge[..., None, None].expand_as(pointwise)
        mask = mask.to(pointwise.dtype) * slow_weight.expand_as(pointwise)
        return self._batch_masked_mean(pointwise, mask)

    def _geometry_losses(
        self,
        pred_x1: Tensor,
        gt_x1: Tensor,
        mean: Tensor,
        std: Tensor,
        bone_offsets: Tensor,
        rotation_space: str,
        data_mask_temporal: Tensor,
        generation_mask: Optional[Tensor],
        timesteps: Optional[Tensor],
        global_step: Optional[int],
    ) -> Dict[str, Tensor]:
        """Compute all enabled geometry terms from one shared FK decode."""
        if pred_x1.shape[-1] < self.motion_dim:
            raise ValueError(
                f"Geometry losses require at least {self.motion_dim} motion "
                f"channels, got {pred_x1.shape[-1]}"
            )

        data_mask = data_mask_temporal.to(pred_x1.device)
        if data_mask.shape[-1] != pred_x1.shape[1]:
            data_mask = data_mask[..., -pred_x1.shape[1]:]

        safe_std = _safe_std(std).to(pred_x1.device, pred_x1.dtype)
        mean = mean.to(pred_x1.device, pred_x1.dtype)
        pred_denorm = pred_x1 * safe_std + mean
        gt_denorm = gt_x1 * safe_std + mean
        generation_mask = self._align_generation_mask(
            generation_mask,
            length=pred_x1.shape[1],
            dim=pred_x1.shape[2],
        )
        if generation_mask is not None:
            generation_mask = generation_mask.to(pred_x1.device)
        pred_world = _fk_global_positions(
            pred_denorm[..., :135],
            bone_offsets,
            rotation_space,
        )
        gt_world = None
        if (
            self.joint_pos_weight > 0.0
            or self.joint_vel_weight > 0.0
            or self.stationary_support_velocity_weight > 0.0
            or self.condition_neighborhood_world_weight > 0.0
            or self.condition_transition_residual_weight > 0.0
            or self.condition_transition_sobolev_weight > 0.0
        ):
            gt_world = _fk_global_positions(
                gt_denorm[..., :135],
                bone_offsets,
                rotation_space,
            )

        t_sq = None
        if self.timestep_squared_weighting and timesteps is not None:
            t_sq = timesteps.to(pred_world.device, pred_world.dtype).square()

        losses: Dict[str, Tensor] = {}
        if (
            self.condition_neighborhood_x1_weight > 0.0
            and generation_mask is not None
        ):
            neighborhood_x1, neighborhood_x1_components = (
                self._condition_neighborhood_x1_loss(
                    pred_x1,
                    gt_x1,
                    data_mask,
                    generation_mask,
                )
            )
            losses["condition_neighborhood_x1"] = (
                self.condition_neighborhood_x1_weight * neighborhood_x1
            )
            for name, value in neighborhood_x1_components.items():
                losses[f"condition_neighborhood_x1_{name}"] = (
                    self.condition_neighborhood_x1_weight * value
                ).detach()

        if (
            self.condition_neighborhood_world_weight > 0.0
            and generation_mask is not None
        ):
            neighborhood_world, neighborhood_world_scales = (
                self._condition_neighborhood_world_loss(
                    pred_world,
                    gt_world,
                    data_mask,
                    generation_mask,
                )
            )
            losses["condition_neighborhood_world"] = (
                self.condition_neighborhood_world_weight * neighborhood_world
            )
            for name, value in neighborhood_world_scales.items():
                losses[f"condition_neighborhood_world_{name}"] = (
                    self.condition_neighborhood_world_weight * value
                ).detach()

        if (
            self.condition_transition_residual_weight > 0.0
            and generation_mask is not None
        ):
            transition_residual, transition_residual_orders = (
                self._condition_transition_residual_loss(
                    pred_world,
                    gt_world,
                    generation_mask,
                    data_mask,
                    t_sq,
                )
            )
            losses["condition_transition_residual"] = (
                self.condition_transition_residual_weight
                * transition_residual
            )
            for name, value in transition_residual_orders.items():
                losses[f"condition_transition_residual_{name}"] = (
                    self.condition_transition_residual_weight * value
                ).detach()

        if (
            self.condition_transition_sobolev_weight > 0.0
            and generation_mask is not None
        ):
            transition_sobolev, transition_sobolev_orders = (
                self._condition_transition_sobolev_parity_loss(
                    pred_world,
                    gt_world,
                    generation_mask,
                    data_mask,
                    t_sq,
                )
            )
            losses["condition_transition_sobolev"] = (
                self.condition_transition_sobolev_weight
                * transition_sobolev
            )
            for name, value in transition_sobolev_orders.items():
                losses[f"condition_transition_sobolev_{name}"] = (
                    self.condition_transition_sobolev_weight * value
                ).detach()

        if (
            self.condition_transition_root_sobolev_weight > 0.0
            and generation_mask is not None
        ):
            root_transition, root_transition_orders = (
                self._condition_transition_root_sobolev_loss(
                    pred_denorm[..., :3],
                    gt_denorm[..., :3],
                    generation_mask,
                    data_mask,
                    orders=self.condition_transition_root_sobolev_orders,
                )
            )
            losses["condition_transition_root_sobolev"] = (
                self.condition_transition_root_sobolev_weight * root_transition
            )
            for name, value in root_transition_orders.items():
                losses[f"condition_transition_root_sobolev_{name}"] = (
                    self.condition_transition_root_sobolev_weight * value
                ).detach()

        if (
            self.condition_transition_root_anchor_weight > 0.0
            and generation_mask is not None
        ):
            root_anchor, root_anchor_scales = (
                self._condition_transition_root_anchor_loss(
                    pred_denorm[..., :3],
                    gt_denorm[..., :3],
                    generation_mask,
                    data_mask,
                )
            )
            losses["condition_transition_root_anchor"] = (
                self.condition_transition_root_anchor_weight * root_anchor
            )
            for name, value in root_anchor_scales.items():
                losses[f"condition_transition_root_anchor_{name}"] = (
                    self.condition_transition_root_anchor_weight * value
                ).detach()

        if self.root_trajectory_weight > 0.0:
            pointwise = self.loss_fn(
                pred_denorm[..., :3],
                gt_denorm[..., :3],
                reduction="none",
            )
            mask = data_mask[..., None].expand_as(pointwise).to(pointwise.dtype)
            if generation_mask is not None:
                root_generation = generation_mask[..., :3].to(pointwise.dtype)
                # Known translation coordinates are hard-projected to GT and
                # therefore carry no useful gradient. Every generated root
                # coordinate remains an explicit physical-scale target.
                mask = mask * root_generation
            root_trajectory = (pointwise * mask).sum() / torch.clamp(
                mask.sum(), min=1.0
            )
            losses["root_trajectory"] = (
                self.root_trajectory_weight * root_trajectory
            )

        if self.conditioned_root_h1_weight > 0.0:
            root_position = pred_denorm.sum() * 0.0
            root_velocity = pred_denorm.sum() * 0.0
            if generation_mask is not None:
                root_position, root_velocity = self._conditioned_root_h1_loss(
                    pred_denorm[..., :3],
                    gt_denorm[..., :3],
                    generation_mask,
                    data_mask,
                    t_sq,
                )
            losses["conditioned_root_h1"] = self.conditioned_root_h1_weight * (
                0.5 * (root_position + root_velocity)
            )
            losses["conditioned_root_position"] = root_position.detach()
            losses["conditioned_root_velocity"] = root_velocity.detach()

        if self.conditioned_root_h2_weight > 0.0:
            root_h2 = pred_denorm.sum() * 0.0
            root_h2_orders: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_h2, root_h2_orders = self._conditioned_root_h2_loss(
                    pred_denorm[..., :3],
                    gt_denorm[..., :3],
                    generation_mask,
                    data_mask,
                )
            losses["conditioned_root_h2"] = (
                self.conditioned_root_h2_weight * root_h2
            )
            for name, value in root_h2_orders.items():
                losses[f"conditioned_root_h2_{name}"] = (
                    self.conditioned_root_h2_weight * value
                ).detach()

        if self.condition_transition_root_h2_weight > 0.0:
            root_join_h2 = pred_denorm.sum() * 0.0
            root_join_h2_orders: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_join_h2, root_join_h2_orders = (
                    self._condition_transition_root_h2_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                    )
                )
            losses["condition_transition_root_h2"] = (
                self.condition_transition_root_h2_weight * root_join_h2
            )
            for name, value in root_join_h2_orders.items():
                losses[f"condition_transition_root_h2_{name}"] = (
                    self.condition_transition_root_h2_weight * value
                ).detach()

        if self.condition_transition_root_tail_weight > 0.0:
            root_tail = pred_denorm.sum() * 0.0
            root_tail_orders: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_tail, root_tail_orders = (
                    self._condition_transition_root_sobolev_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                        orders=(1, 2, 3),
                        band_radius=self.condition_transition_root_tail_band_radius,
                        tail_rms=True,
                    )
                )
            losses["condition_transition_root_tail"] = (
                self.condition_transition_root_tail_weight * root_tail
            )
            for name, value in root_tail_orders.items():
                losses[f"condition_transition_root_tail_{name}"] = (
                    self.condition_transition_root_tail_weight * value
                ).detach()

        if self.condition_transition_root_cvar_weight > 0.0:
            root_cvar = pred_denorm.sum() * 0.0
            root_cvar_orders: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_cvar, root_cvar_orders = (
                    self._condition_transition_root_sobolev_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                        orders=(1, 2, 3),
                        band_radius=2,
                        tail_fraction=0.2,
                    )
                )
            losses["condition_transition_root_cvar"] = (
                self.condition_transition_root_cvar_weight * root_cvar
            )
            for name, value in root_cvar_orders.items():
                losses[f"condition_transition_root_cvar_{name}"] = (
                    self.condition_transition_root_cvar_weight * value
                ).detach()

        if self.condition_transition_root_dynamics_ceiling_weight > 0.0:
            root_ceiling = pred_denorm.sum() * 0.0
            root_ceiling_orders: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_ceiling, root_ceiling_orders = (
                    self._condition_transition_root_dynamics_ceiling_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                    )
                )
            losses["condition_transition_root_dynamics_ceiling"] = (
                self.condition_transition_root_dynamics_ceiling_weight
                * root_ceiling
            )
            for name, value in root_ceiling_orders.items():
                losses[f"condition_transition_root_dynamics_ceiling_{name}"] = (
                    self.condition_transition_root_dynamics_ceiling_weight
                    * value
                ).detach()

        if self.condition_transition_root_endpoint_weight > 0.0:
            root_endpoint = pred_denorm.sum() * 0.0
            root_endpoint_scales: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_endpoint, root_endpoint_scales = (
                    self._condition_transition_root_endpoint_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                    )
                )
            losses["condition_transition_root_endpoint"] = (
                self.condition_transition_root_endpoint_weight * root_endpoint
            )
            for name, value in root_endpoint_scales.items():
                losses[f"condition_transition_root_endpoint_{name}"] = (
                    self.condition_transition_root_endpoint_weight * value
                ).detach()

        if self.condition_transition_root_endpoint_tail_weight > 0.0:
            root_endpoint_tail = pred_denorm.sum() * 0.0
            root_endpoint_tail_scales: Dict[str, Tensor] = {}
            if generation_mask is not None:
                root_endpoint_tail, root_endpoint_tail_scales = (
                    self._condition_transition_root_endpoint_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                        tail_fraction=0.2,
                    )
                )
            losses["condition_transition_root_endpoint_tail"] = (
                self.condition_transition_root_endpoint_tail_weight
                * root_endpoint_tail
            )
            for name, value in root_endpoint_tail_scales.items():
                losses[f"condition_transition_root_endpoint_tail_{name}"] = (
                    self.condition_transition_root_endpoint_tail_weight * value
                ).detach()

        if self.condition_transition_root_waypoint_curvature_weight > 0.0:
            root_waypoint_curvature = pred_denorm.sum() * 0.0
            if generation_mask is not None:
                root_waypoint_curvature = (
                    self._condition_transition_root_waypoint_curvature_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                    )
                )
            losses["condition_transition_root_waypoint_curvature"] = (
                self.condition_transition_root_waypoint_curvature_weight
                * root_waypoint_curvature
            )

        if self.condition_transition_root_waypoint_tail_weight > 0.0:
            root_waypoint_tail = pred_denorm.sum() * 0.0
            if generation_mask is not None:
                root_waypoint_tail = (
                    self._condition_transition_root_waypoint_tail_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                    )
                )
            losses["condition_transition_root_waypoint_tail"] = (
                self.condition_transition_root_waypoint_tail_weight
                * root_waypoint_tail
            )

        if self.condition_transition_root_waypoint_edge_weight > 0.0:
            root_waypoint_edge = pred_denorm.sum() * 0.0
            if generation_mask is not None:
                if timesteps is None:
                    raise ValueError(
                        "condition_transition_root_waypoint_edge_weight requires "
                        "timesteps"
                    )
                root_waypoint_edge = (
                    self._condition_transition_root_waypoint_edge_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                        timesteps,
                        tail_fraction=(
                            self.condition_transition_root_waypoint_edge_tail_fraction
                        ),
                    )
                )
            losses["condition_transition_root_waypoint_edge"] = (
                self.condition_transition_root_waypoint_edge_weight
                * root_waypoint_edge
            )

        if self.condition_transition_root_waypoint_basin_weight > 0.0:
            root_waypoint_basin = pred_denorm.sum() * 0.0
            root_waypoint_basin_scales: Dict[str, Tensor] = {}
            if generation_mask is not None:
                if timesteps is None:
                    raise ValueError(
                        "condition_transition_root_waypoint_basin_weight requires "
                        "timesteps"
                    )
                root_waypoint_basin, root_waypoint_basin_scales = (
                    self._condition_transition_root_waypoint_basin_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                        timesteps,
                    )
                )
            losses["condition_transition_root_waypoint_basin"] = (
                self.condition_transition_root_waypoint_basin_weight
                * root_waypoint_basin
            )
            for name, value in root_waypoint_basin_scales.items():
                losses[f"condition_transition_root_waypoint_basin_{name}"] = (
                    self.condition_transition_root_waypoint_basin_weight * value
                ).detach()

        if self.condition_transition_root_waypoint_h1_weight > 0.0:
            root_waypoint_h1 = pred_denorm.sum() * 0.0
            root_waypoint_h1_scales: Dict[str, Tensor] = {}
            if generation_mask is not None:
                if timesteps is None:
                    raise ValueError(
                        "condition_transition_root_waypoint_h1_weight requires "
                        "timesteps"
                    )
                root_waypoint_h1, root_waypoint_h1_scales = (
                    self._condition_transition_root_waypoint_h1_loss(
                        pred_denorm[..., :3],
                        gt_denorm[..., :3],
                        generation_mask,
                        data_mask,
                        timesteps,
                        detach_inner=(
                            self.condition_transition_root_waypoint_h1_detach_inner
                        ),
                    )
                )
            losses["condition_transition_root_waypoint_h1"] = (
                self.condition_transition_root_waypoint_h1_weight
                * root_waypoint_h1
            )
            for name, value in root_waypoint_h1_scales.items():
                losses[f"condition_transition_root_waypoint_h1_{name}"] = (
                    self.condition_transition_root_waypoint_h1_weight * value
                ).detach()

        if self.joint_pos_weight > 0.0:
            per_frame = self.loss_fn(
                pred_world,
                gt_world,
                reduction="none",
            ).mean(dim=(-1, -2))
            if t_sq is not None:
                per_frame = per_frame * t_sq.unsqueeze(-1)
            weight = self._warmup(
                self.joint_pos_weight,
                self.joint_pos_warmup_steps,
                global_step,
            )
            losses["joint_pos"] = weight * _temporal_mean_masked(
                per_frame,
                data_mask,
            )

        if self.joint_vel_weight > 0.0:
            pred_joint_vel = pred_world[:, 1:] - pred_world[:, :-1]
            gt_joint_vel = gt_world[:, 1:] - gt_world[:, :-1]
            pointwise_raw = self.loss_fn(
                pred_joint_vel,
                gt_joint_vel,
                reduction="none",
            )
            pointwise = pointwise_raw
            if t_sq is not None:
                pointwise = pointwise * t_sq[:, None, None, None]
            per_frame = pointwise.mean(dim=(-1, -2))
            velocity_mask = data_mask[:, 1:] * data_mask[:, :-1]
            ordinary = _temporal_mean_masked(per_frame, velocity_mask)

            condition_kinematic = ordinary * 0.0
            condition_acceleration = ordinary * 0.0
            stationary_support = ordinary * 0.0
            transition_velocity_excess = ordinary * 0.0
            transition_acceleration_excess = ordinary * 0.0
            transition_jerk_excess = ordinary * 0.0
            transition_jerk_ceiling = ordinary * 0.0
            transition_secant_excess = ordinary * 0.0
            if generation_mask is not None and (
                self.condition_kinematic_weight > 0.0
                or self.condition_acceleration_weight > 0.0
            ):
                condition_map = self._condition_kinematic_map(generation_mask)
                condition_mask = (
                    condition_map.to(pointwise_raw.dtype)
                    * velocity_mask[..., None, None].to(pointwise_raw.dtype)
                )
                condition_kinematic = (
                    pointwise_raw * condition_mask
                ).sum() / torch.clamp(condition_mask.sum(), min=1.0)

                if (
                    self.condition_acceleration_weight > 0.0
                    and pred_world.shape[1] >= 3
                ):
                    pred_acc = (
                        pred_world[:, 2:]
                        - 2.0 * pred_world[:, 1:-1]
                        + pred_world[:, :-2]
                    )
                    gt_acc = (
                        gt_world[:, 2:]
                        - 2.0 * gt_world[:, 1:-1]
                        + gt_world[:, :-2]
                    )
                    acc_pointwise = self.loss_fn(
                        pred_acc,
                        gt_acc,
                        reduction="none",
                    )
                    acc_map = torch.maximum(
                        condition_map[:, 1:],
                        condition_map[:, :-1],
                    ).to(acc_pointwise.dtype)
                    valid_triple = (
                        data_mask[:, 2:]
                        * data_mask[:, 1:-1]
                        * data_mask[:, :-2]
                    )
                    acc_mask = (
                        acc_map
                        * valid_triple[..., None, None].to(acc_pointwise.dtype)
                    )
                    condition_acceleration = (
                        acc_pointwise * acc_mask
                    ).sum() / torch.clamp(acc_mask.sum(), min=1.0)

            if (
                generation_mask is not None
                and self.condition_transition_excess_weight > 0.0
            ):
                transition_map = self._condition_transition_map(
                    generation_mask
                ).to(pointwise.dtype)
                condition_support_map = self._condition_kinematic_map(
                    generation_mask
                ).to(pointwise.dtype)
                transition_velocity_excess = (
                    self._condition_transition_excess_loss(
                        pointwise,
                        transition_map,
                        condition_support_map,
                        velocity_mask,
                    )
                )

                if pred_world.shape[1] >= 3:
                    pred_acc = (
                        pred_world[:, 2:]
                        - 2.0 * pred_world[:, 1:-1]
                        + pred_world[:, :-2]
                    )
                    gt_acc = (
                        gt_world[:, 2:]
                        - 2.0 * gt_world[:, 1:-1]
                        + gt_world[:, :-2]
                    )
                    transition_acc_pointwise = self.loss_fn(
                        pred_acc,
                        gt_acc,
                        reduction="none",
                    )
                    if t_sq is not None:
                        transition_acc_pointwise = (
                            transition_acc_pointwise
                            * t_sq[:, None, None, None]
                        )
                    transition_acc_map = torch.maximum(
                        transition_map[:, 1:],
                        transition_map[:, :-1],
                    )
                    condition_acc_support_map = torch.maximum(
                        condition_support_map[:, 1:],
                        condition_support_map[:, :-1],
                    )
                    valid_triple = (
                        data_mask[:, 2:]
                        * data_mask[:, 1:-1]
                        * data_mask[:, :-2]
                    )
                    transition_acceleration_excess = (
                        self._condition_transition_excess_loss(
                            transition_acc_pointwise,
                            transition_acc_map,
                            condition_acc_support_map,
                            valid_triple,
                        )
                    )

                if pred_world.shape[1] >= 4:
                    pred_jerk = (
                        pred_world[:, 3:]
                        - 3.0 * pred_world[:, 2:-1]
                        + 3.0 * pred_world[:, 1:-2]
                        - pred_world[:, :-3]
                    )
                    gt_jerk = (
                        gt_world[:, 3:]
                        - 3.0 * gt_world[:, 2:-1]
                        + 3.0 * gt_world[:, 1:-2]
                        - gt_world[:, :-3]
                    )
                    transition_jerk_pointwise = self.loss_fn(
                        pred_jerk,
                        gt_jerk,
                        reduction="none",
                    )
                    if t_sq is not None:
                        transition_jerk_pointwise = (
                            transition_jerk_pointwise
                            * t_sq[:, None, None, None]
                        )
                    transition_jerk_map = torch.maximum(
                        torch.maximum(
                            transition_map[:, 2:],
                            transition_map[:, 1:-1],
                        ),
                        transition_map[:, :-2],
                    )
                    condition_jerk_support_map = torch.maximum(
                        torch.maximum(
                            condition_support_map[:, 2:],
                            condition_support_map[:, 1:-1],
                        ),
                        condition_support_map[:, :-2],
                    )
                    valid_quadruple = (
                        data_mask[:, 3:]
                        * data_mask[:, 2:-1]
                        * data_mask[:, 1:-2]
                        * data_mask[:, :-3]
                    )
                    transition_jerk_excess = (
                        self._condition_transition_excess_loss(
                            transition_jerk_pointwise,
                            transition_jerk_map,
                            condition_jerk_support_map,
                            valid_quadruple,
                        )
                    )

            if (
                generation_mask is not None
                and self.condition_transition_jerk_ceiling_weight > 0.0
                and pred_world.shape[1] >= 4
            ):
                transition_map = self._condition_transition_map(
                    generation_mask
                ).to(pred_world.dtype)
                condition_support_map = self._condition_kinematic_map(
                    generation_mask
                ).to(pred_world.dtype)
                pred_jerk = (
                    pred_world[:, 3:]
                    - 3.0 * pred_world[:, 2:-1]
                    + 3.0 * pred_world[:, 1:-2]
                    - pred_world[:, :-3]
                )
                gt_jerk = (
                    gt_world[:, 3:]
                    - 3.0 * gt_world[:, 2:-1]
                    + 3.0 * gt_world[:, 1:-2]
                    - gt_world[:, :-3]
                )
                transition_jerk_map = torch.maximum(
                    torch.maximum(
                        transition_map[:, 2:],
                        transition_map[:, 1:-1],
                    ),
                    transition_map[:, :-2],
                )
                condition_jerk_support_map = torch.maximum(
                    torch.maximum(
                        condition_support_map[:, 2:],
                        condition_support_map[:, 1:-1],
                    ),
                    condition_support_map[:, :-2],
                )
                valid_quadruple = (
                    data_mask[:, 3:]
                    * data_mask[:, 2:-1]
                    * data_mask[:, 1:-2]
                    * data_mask[:, :-3]
                )
                transition_jerk_ceiling = (
                    self._condition_transition_jerk_ceiling_loss(
                        pred_jerk,
                        gt_jerk,
                        transition_jerk_map,
                        condition_jerk_support_map,
                        valid_quadruple,
                        t_sq,
                    )
                )

            if (
                generation_mask is not None
                and self.condition_transition_secant_excess_weight > 0.0
            ):
                transition_secant_excess = (
                    self._condition_transition_secant_excess_loss(
                        pred_world,
                        gt_world,
                        generation_mask,
                        data_mask,
                        t_sq,
                    )
                )

            transition_excess = (1.0 / 3.0) * (
                transition_velocity_excess
                + transition_acceleration_excess
                + transition_jerk_excess
            )

            if (
                generation_mask is not None
                and self.stationary_support_velocity_weight > 0.0
            ):
                stationary_support = self._stationary_support_velocity_loss(
                    pred_world,
                    gt_world,
                    generation_mask,
                    data_mask,
                    t_sq,
                )

            weight = self._warmup(
                self.joint_vel_weight,
                self.joint_vel_warmup_steps,
                global_step,
            )
            losses["joint_vel"] = weight * (
                ordinary
                + self.condition_kinematic_weight * condition_kinematic
                + self.condition_acceleration_weight * condition_acceleration
                + self.stationary_support_velocity_weight * stationary_support
                + self.condition_transition_excess_weight * transition_excess
                + self.condition_transition_jerk_ceiling_weight
                * transition_jerk_ceiling
                + self.condition_transition_secant_excess_weight
                * transition_secant_excess
            )
            losses["condition_kinematic"] = (
                weight
                * self.condition_kinematic_weight
                * condition_kinematic
            ).detach()
            losses["condition_acceleration"] = (
                weight
                * self.condition_acceleration_weight
                * condition_acceleration
            ).detach()
            losses["stationary_support_velocity"] = (
                weight
                * self.stationary_support_velocity_weight
                * stationary_support
            ).detach()
            losses["condition_transition_excess"] = (
                weight
                * self.condition_transition_excess_weight
                * transition_excess
            ).detach()
            losses["condition_transition_velocity_excess"] = (
                transition_velocity_excess
            ).detach()
            losses["condition_transition_acceleration_excess"] = (
                transition_acceleration_excess
            ).detach()
            losses["condition_transition_jerk_excess"] = (
                transition_jerk_excess
            ).detach()
            losses["condition_transition_jerk_ceiling"] = (
                weight
                * self.condition_transition_jerk_ceiling_weight
                * transition_jerk_ceiling
            ).detach()
            losses["condition_transition_secant_excess"] = (
                weight
                * self.condition_transition_secant_excess_weight
                * transition_secant_excess
            ).detach()

        if self.fk_consistency_weight > 0.0:
            pred_position = pred_denorm[..., 135:198]
            fk_position = _strict_ric_relative(pred_world)
            per_frame = self.loss_fn(
                pred_position,
                fk_position,
                reduction="none",
            ).mean(dim=-1)
            if t_sq is not None:
                per_frame = per_frame * t_sq.unsqueeze(-1)
            weight = self._warmup(
                self.fk_consistency_weight,
                self.fk_consistency_warmup_steps,
                global_step,
            )
            losses["fk_consistency"] = weight * _temporal_mean_masked(
                per_frame,
                data_mask,
            )
        return losses

    def forward(
        self,
        *,
        pred_vel: Optional[Tensor] = None,
        gt_vel: Optional[Tensor] = None,
        data_mask_temporal: Tensor,
        generation_mask: Optional[Tensor] = None,
        pred_x1: Optional[Tensor] = None,
        gt_x1: Optional[Tensor] = None,
        mean: Optional[Tensor] = None,
        std: Optional[Tensor] = None,
        bone_offsets: Optional[Tensor] = None,
        rotation_space: str = "local",
        timesteps: Optional[Tensor] = None,
        global_step: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """Compute the unified objective.

        Args:
            pred_vel: Predicted rectified-flow velocity, normalized space.
            gt_vel: ``x1 - x0`` target with the same shape as ``pred_vel``.
            data_mask_temporal: ``(B,T)`` mask, ``1=valid``, ``0=padding``.
            generation_mask: Optional ``(B,T,D)`` mask, ``1=optimize`` and
                ``0=known condition``. It masks flow and root-trajectory loss;
                decoded world-joint terms supervise the projected clean motion.
            pred_x1: Condition-projected clean-motion estimate. Required when
                any geometry weight is non-zero.
            gt_x1: Normalized clean target corresponding to ``pred_x1``.
            mean: Motion normalization mean, shape ``(D,)``.
            std: Motion normalization std, shape ``(D,)``.
            bone_offsets: SMPL-22 offsets consumed by differentiable FK.
            rotation_space: Rotation convention passed to FK.
            timesteps: Flow time ``t`` in ``[0,1]`` for optional ``t^2`` weight.
            global_step: Optimizer step used by geometry warmups.

        Returns:
            Differentiable keys ``velocity``, ``root_trajectory``, ``joint_pos``,
            ``joint_vel``, ``conditioned_root_h1``, and
            ``fk_consistency`` when enabled. ``velocity_*`` and
            condition-specific diagnostic keys are detached and excluded from
            optimization by ``BaseTrainer``.
        """
        if (pred_vel is None) != (gt_vel is None):
            raise ValueError("pred_vel and gt_vel must be provided together")

        losses: Dict[str, Tensor] = {}
        if pred_vel is not None:
            flow_per_dim = self.loss_fn(pred_vel, gt_vel, reduction="none")
            flow_loss, flow_components = self._flow_loss_with_components(
                flow_per_dim,
                data_mask_temporal,
                generation_mask,
            )

            # Clean imputation defines a piecewise flow path
            #
            #   x_t^M = M * ((1-t)x_0 + t x_1) + (1-M) * x_1,
            #
            # where M=1 denotes generated coordinates. Its exact derivative is
            # ``M * (x_1-x_0)``: known coordinates have zero velocity. Reducing
            # this support separately is important for a waypoint containing
            # only one or two axes; a single all-coordinate mean would make its
            # supervision vanish relative to the generated region.
            condition_flow = flow_loss * 0.0
            neighborhood_flow = flow_loss * 0.0
            condition_components: Dict[str, Tensor] = {}
            neighborhood_components: Dict[str, Tensor] = {}
            if generation_mask is not None:
                condition_per_dim = self.loss_fn(
                    pred_vel,
                    torch.zeros_like(pred_vel),
                    reduction="none",
                )
                condition_mask = 1.0 - generation_mask.to(
                    device=pred_vel.device,
                    dtype=pred_vel.dtype,
                )
                condition_flow, condition_components = (
                    self._flow_loss_with_components(
                        condition_per_dim,
                        data_mask_temporal,
                        condition_mask,
                    )
                )
                if self.condition_neighborhood_flow_weight > 0.0:
                    neighborhood_flow, neighborhood_components = (
                        self._condition_neighborhood_flow_loss(
                            flow_per_dim,
                            data_mask_temporal,
                            generation_mask,
                        )
                    )

            losses["velocity"] = self.velocity_weight * (
                flow_loss
                + condition_flow
                + self.condition_neighborhood_flow_weight * neighborhood_flow
            )
            for name, value in flow_components.items():
                losses[f"velocity_{name}"] = value.detach()
            losses["condition_velocity"] = (
                self.velocity_weight * condition_flow
            ).detach()
            for name, value in condition_components.items():
                losses[f"condition_velocity_{name}"] = value.detach()
            losses["condition_neighborhood_flow"] = (
                self.velocity_weight
                * self.condition_neighborhood_flow_weight
                * neighborhood_flow
            ).detach()
            for name, value in neighborhood_components.items():
                losses[f"condition_neighborhood_flow_{name}"] = (
                    self.velocity_weight
                    * self.condition_neighborhood_flow_weight
                    * value
                ).detach()

        if not self.geometry_enabled:
            return losses
        required = {
            "pred_x1": pred_x1,
            "gt_x1": gt_x1,
            "mean": mean,
            "std": std,
            "bone_offsets": bone_offsets,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(
                "Geometry losses are enabled but required inputs are missing: "
                + ", ".join(missing)
            )
        losses.update(
            self._geometry_losses(
                pred_x1=pred_x1,
                gt_x1=gt_x1,
                mean=mean,
                std=std,
                bone_offsets=bone_offsets,
                rotation_space=rotation_space,
                data_mask_temporal=data_mask_temporal,
                generation_mask=generation_mask,
                timesteps=timesteps,
                global_step=global_step,
            )
        )
        return losses
