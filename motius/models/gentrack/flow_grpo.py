"""Flow-GRPO primitives for the G1 rectified-flow generator.

The generator uses the repository convention ``t=0`` noise and ``t=1`` data.
Flow-GRPO uses the equivalent decreasing noise coordinate ``sigma=1-t``;
therefore its solver receives ``model_output=-v_theta``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def sample_unique_timestep_indices(
    batch_size: int,
    num_transitions: int,
    timesteps_per_update: int,
    device: torch.device | str,
) -> Tensor:
    """Sample distinct diffusion transitions independently for each trajectory."""
    batch_size = int(batch_size)
    num_transitions = int(num_transitions)
    timesteps_per_update = int(timesteps_per_update)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_transitions < 1:
        raise ValueError("num_transitions must be positive")
    if not 1 <= timesteps_per_update <= num_transitions:
        raise ValueError(
            "timesteps_per_update must be in "
            f"[1, {num_transitions}], got {timesteps_per_update}"
        )
    # Independent random permutations avoid duplicate flow times within one
    # trajectory update while retaining unbiased coverage over the schedule.
    random_keys = torch.rand(
        batch_size,
        num_transitions,
        device=device,
    )
    return random_keys.argsort(dim=1)[:, :timesteps_per_update]


def group_relative_advantages(rewards: Tensor, group_size: int, eps: float = 1e-4) -> Tensor:
    """Normalize rewards independently within each same-prompt group."""
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got {tuple(rewards.shape)}")
    if group_size < 2:
        raise ValueError("Flow-GRPO requires at least two samples per prompt")
    if rewards.numel() % group_size:
        raise ValueError(
            f"reward count {rewards.numel()} is not divisible by group_size={group_size}"
        )
    grouped = rewards.reshape(-1, group_size)
    mean = grouped.mean(dim=1, keepdim=True)
    std = grouped.std(dim=1, unbiased=False, keepdim=True)
    advantages = (grouped - mean) / (std + float(eps))
    return advantages.reshape_as(rewards)


def select_reliable_preference_pairs(
    rewards: Tensor,
    *,
    group_size: int,
    min_reward_gap: float,
    valid_mask: Tensor | None = None,
    semantic_distances: Tensor | None = None,
    reference_semantic_distances: Tensor | None = None,
    semantic_floor_margin: float = 0.0,
    pairwise_guard_costs: Tensor | None = None,
    pairwise_guard_margins: Tensor | None = None,
) -> dict[str, Tensor]:
    """Select same-prompt winner/loser pairs above a calibrated reward gap.

    The physical reward determines only the ordering. TMR-G1 is an eligibility
    constraint: when semantic distances are provided, both members of a pair
    must be no worse than their same-noise immutable-G0 counterpart by more
    than ``semantic_floor_margin``. Optional lower-is-better pairwise guard
    costs prevent the selected physical winner from trading away a protected
    metric such as start-aligned root trajectory error. This avoids inventing
    scalar conversions between heterogeneous evaluator metrics.
    """
    if rewards.ndim != 1:
        raise ValueError(f"rewards must be 1-D, got {tuple(rewards.shape)}")
    group_size = int(group_size)
    if group_size < 2:
        raise ValueError("preference selection requires at least two samples")
    if rewards.numel() % group_size:
        raise ValueError(
            f"reward count {rewards.numel()} is not divisible by "
            f"group_size={group_size}"
        )
    if float(min_reward_gap) < 0:
        raise ValueError("min_reward_gap must be non-negative")
    if float(semantic_floor_margin) < 0:
        raise ValueError("semantic_floor_margin must be non-negative")

    eligible = torch.isfinite(rewards)
    if valid_mask is not None:
        if valid_mask.shape != rewards.shape:
            raise ValueError("valid_mask must match rewards")
        eligible &= valid_mask.to(device=rewards.device, dtype=torch.bool)

    semantic_values = (semantic_distances, reference_semantic_distances)
    if (semantic_values[0] is None) != (semantic_values[1] is None):
        raise ValueError(
            "semantic_distances and reference_semantic_distances must be "
            "provided together"
        )
    if semantic_distances is not None:
        if (
            semantic_distances.shape != rewards.shape
            or reference_semantic_distances.shape != rewards.shape
        ):
            raise ValueError("semantic distance tensors must match rewards")
        semantic_distances = semantic_distances.to(rewards.device)
        reference_semantic_distances = reference_semantic_distances.to(
            rewards.device
        )
        eligible &= torch.isfinite(semantic_distances)
        eligible &= torch.isfinite(reference_semantic_distances)
        eligible &= (
            semantic_distances
            <= reference_semantic_distances + float(semantic_floor_margin)
        )

    guard_costs = None
    guard_margins = None
    if pairwise_guard_costs is not None:
        guard_costs = torch.as_tensor(
            pairwise_guard_costs,
            device=rewards.device,
            dtype=rewards.dtype,
        )
        if guard_costs.ndim == 1:
            guard_costs = guard_costs.unsqueeze(1)
        if guard_costs.ndim != 2 or guard_costs.shape[0] != rewards.numel():
            raise ValueError(
                "pairwise_guard_costs must have shape [num_rewards, num_guards]"
            )
        if guard_costs.shape[1] < 1:
            raise ValueError("pairwise_guard_costs must contain at least one guard")
        if pairwise_guard_margins is None:
            guard_margins = rewards.new_zeros((guard_costs.shape[1],))
        else:
            guard_margins = torch.as_tensor(
                pairwise_guard_margins,
                device=rewards.device,
                dtype=rewards.dtype,
            ).flatten()
            if guard_margins.shape != (guard_costs.shape[1],):
                raise ValueError(
                    "pairwise_guard_margins must match the guard dimension"
                )
            if torch.any(guard_margins < 0):
                raise ValueError("pairwise_guard_margins must be non-negative")
        eligible &= torch.isfinite(guard_costs).all(dim=1)

    grouped_rewards = rewards.reshape(-1, group_size)
    grouped_eligible = eligible.reshape(-1, group_size)
    grouped_guard_costs = (
        guard_costs.reshape(-1, group_size, guard_costs.shape[1])
        if guard_costs is not None
        else None
    )
    winner_indices = []
    loser_indices = []
    reward_gaps = []
    accepted_group_mask = torch.zeros(
        grouped_rewards.shape[0],
        dtype=torch.bool,
        device=rewards.device,
    )
    eligible_counts = grouped_eligible.sum(dim=1)

    for group_index in range(grouped_rewards.shape[0]):
        local_indices = torch.nonzero(
            grouped_eligible[group_index],
            as_tuple=False,
        ).flatten()
        if local_indices.numel() < 2:
            continue
        best_pair = None
        best_gap = None
        for winner_local in local_indices:
            for loser_local in local_indices:
                if int(winner_local) == int(loser_local):
                    continue
                gap = (
                    grouped_rewards[group_index, winner_local]
                    - grouped_rewards[group_index, loser_local]
                )
                # A tied pair contains no preference information even when the
                # configured noise floor is zero.
                if not torch.isfinite(gap) or float(gap) <= float(min_reward_gap):
                    continue
                if grouped_guard_costs is not None:
                    winner_costs = grouped_guard_costs[group_index, winner_local]
                    loser_costs = grouped_guard_costs[group_index, loser_local]
                    assert guard_margins is not None
                    if not torch.all(winner_costs <= loser_costs + guard_margins):
                        continue
                if best_gap is None or float(gap) > float(best_gap):
                    best_pair = (winner_local, loser_local)
                    best_gap = gap
        if best_pair is None:
            continue
        winner_local, loser_local = best_pair
        assert best_gap is not None
        offset = group_index * group_size
        winner_indices.append(offset + int(winner_local))
        loser_indices.append(offset + int(loser_local))
        reward_gaps.append(best_gap)
        accepted_group_mask[group_index] = True

    if winner_indices:
        winners = torch.tensor(
            winner_indices,
            dtype=torch.long,
            device=rewards.device,
        )
        losers = torch.tensor(
            loser_indices,
            dtype=torch.long,
            device=rewards.device,
        )
        gaps = torch.stack(reward_gaps)
    else:
        winners = torch.empty(0, dtype=torch.long, device=rewards.device)
        losers = torch.empty(0, dtype=torch.long, device=rewards.device)
        gaps = rewards.new_empty((0,))

    return {
        "winner_indices": winners,
        "loser_indices": losers,
        "reward_gaps": gaps,
        "accepted_group_mask": accepted_group_mask,
        "eligible_mask": eligible,
        "eligible_counts": eligible_counts,
    }


def reference_augmented_advantages(
    candidate_rewards: Tensor,
    reference_rewards: Tensor,
    group_size: int,
    eps: float = 1e-4,
) -> Tensor:
    """Normalize trainable candidates against candidate and frozen-G0 samples.

    Unlike candidate-only group normalization, the frozen reference samples
    remain in the baseline statistics. Consequently, every trainable candidate
    can receive a negative advantage when an entire group is worse than G0.
    """
    if candidate_rewards.ndim != 1 or reference_rewards.ndim != 1:
        raise ValueError("candidate_rewards and reference_rewards must be 1-D")
    if candidate_rewards.shape != reference_rewards.shape:
        raise ValueError(
            "candidate_rewards and reference_rewards must have identical shape, "
            f"got {tuple(candidate_rewards.shape)} and {tuple(reference_rewards.shape)}"
        )
    if group_size < 2:
        raise ValueError("Flow-GRPO requires at least two samples per prompt")
    if candidate_rewards.numel() % group_size:
        raise ValueError(
            f"reward count {candidate_rewards.numel()} is not divisible by "
            f"group_size={group_size}"
        )
    candidates = candidate_rewards.reshape(-1, group_size)
    references = reference_rewards.reshape(-1, group_size)
    augmented = torch.cat([candidates, references], dim=1)
    mean = augmented.mean(dim=1, keepdim=True)
    std = augmented.std(dim=1, unbiased=False, keepdim=True)
    return ((candidates - mean) / (std + float(eps))).reshape_as(candidate_rewards)


def paired_reference_advantages(
    candidate_rewards: Tensor,
    reference_rewards: Tensor,
    group_size: int,
    eps: float = 1e-4,
) -> Tensor:
    """Scale same-noise candidate-vs-G0 deltas without changing their sign.

    The immutable reference is a counterfactual control variate, not another
    sampled action. Each candidate is compared with the G0 motion generated
    from exactly the same stochastic path. A same-prompt scale makes physical
    and semantic terms comparable while keeping an all-regressing group
    entirely negative.
    """
    if candidate_rewards.ndim != 1 or reference_rewards.ndim != 1:
        raise ValueError("candidate_rewards and reference_rewards must be 1-D")
    if candidate_rewards.shape != reference_rewards.shape:
        raise ValueError(
            "candidate_rewards and reference_rewards must have identical shape, "
            f"got {tuple(candidate_rewards.shape)} and {tuple(reference_rewards.shape)}"
        )
    if group_size < 2:
        raise ValueError("Flow-GRPO requires at least two samples per prompt")
    if candidate_rewards.numel() % group_size:
        raise ValueError(
            f"reward count {candidate_rewards.numel()} is not divisible by "
            f"group_size={group_size}"
        )
    candidates = candidate_rewards.reshape(-1, group_size)
    references = reference_rewards.reshape(-1, group_size)
    scale = torch.cat([candidates, references], dim=1).std(
        dim=1,
        unbiased=False,
        keepdim=True,
    ).clamp_min(float(eps))
    return ((candidates - references) / scale).reshape_as(candidate_rewards)


def flow_grpo_transition(
    model_output: Tensor,
    latents: Tensor,
    sigma: Tensor,
    sigma_next: Tensor,
    sigma_first_next: Tensor,
    eta: float,
    next_latents: Tensor | None = None,
    frame_mask: Tensor | None = None,
    transition_noise: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample/replay one stochastic Flow-GRPO transition and its log-prob.

    ``model_output`` is in decreasing-sigma coordinates. For this repository's
    rectified-flow model, callers pass ``-predict_flow(...)``.
    """
    if eta <= 0:
        raise ValueError("eta must be positive for a stochastic policy update")
    dtype = model_output.dtype
    device = model_output.device
    batch = model_output.shape[0]

    def expand(value: Tensor) -> Tensor:
        value = torch.as_tensor(value, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.expand(batch)
        return value.reshape(batch, *([1] * (model_output.ndim - 1)))

    sigma_e = expand(sigma)
    sigma_next_e = expand(sigma_next)
    sigma_first_next_e = expand(sigma_first_next)
    dt = sigma_next_e - sigma_e
    if torch.any(dt >= 0):
        raise ValueError("sigma schedule must be strictly decreasing")

    safe_denom_sigma = torch.where(sigma_e == 1, sigma_first_next_e, sigma_e)
    std_dev = torch.sqrt(sigma_e / (1 - safe_denom_sigma).clamp_min(1e-6)) * float(eta)
    transition_std = std_dev * torch.sqrt((-dt).clamp_min(1e-12))
    mean = (
        latents * (1 + std_dev.square() / (2 * sigma_e.clamp_min(1e-6)) * dt)
        + model_output
        * (1 + std_dev.square() * (1 - sigma_e) / (2 * sigma_e.clamp_min(1e-6)))
        * dt
    )
    if next_latents is not None and transition_noise is not None:
        raise ValueError("next_latents and transition_noise are mutually exclusive")
    if next_latents is None:
        if transition_noise is None:
            transition_noise = torch.randn_like(mean)
        else:
            transition_noise = transition_noise.to(device=device, dtype=dtype)
            if transition_noise.shape != mean.shape:
                raise ValueError(
                    f"transition_noise shape {tuple(transition_noise.shape)} "
                    f"does not match transition mean {tuple(mean.shape)}"
                )
        next_latents = mean + transition_std * transition_noise

    per_element = (
        -0.5 * ((next_latents.detach() - mean) / transition_std.clamp_min(1e-8)).square()
        - torch.log(transition_std.clamp_min(1e-8))
        - 0.5 * math.log(2.0 * math.pi)
    )
    if frame_mask is None:
        log_prob = per_element.flatten(1).mean(dim=1)
    else:
        mask = frame_mask.to(device=device, dtype=per_element.dtype)
        if mask.ndim == per_element.ndim - 1:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(per_element)
        log_prob = (per_element * mask).flatten(1).sum(dim=1) / mask.flatten(1).sum(dim=1).clamp_min(1)
    return next_latents, log_prob


def clipped_grpo_loss(
    current_log_prob: Tensor,
    old_log_prob: Tensor,
    advantages: Tensor,
    clip_range: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """PPO-style clipped group-relative policy loss."""
    log_ratio = (current_log_prob - old_log_prob.detach()).clamp(-20.0, 20.0)
    ratio = log_ratio.exp()
    unclipped = -advantages.detach() * ratio
    clipped = -advantages.detach() * ratio.clamp(1.0 - clip_range, 1.0 + clip_range)
    loss = torch.maximum(unclipped, clipped).mean()
    clip_fraction = (torch.abs(ratio - 1.0) > clip_range).float().mean()
    return loss, ratio.detach().mean(), clip_fraction.detach()


def reverse_kl_from_log_probs(current_log_prob: Tensor, reference_log_prob: Tensor) -> Tensor:
    """Non-negative k3 estimator used by GRPO-style reference regularization."""
    log_ratio = (reference_log_prob.detach() - current_log_prob).clamp(-20.0, 20.0)
    return (log_ratio.exp() - log_ratio - 1.0).mean()


def flow_dpo_pair_loss(
    model_winner_mse: Tensor,
    model_loser_mse: Tensor,
    reference_winner_mse: Tensor,
    reference_loser_mse: Tensor,
    beta: float,
) -> dict[str, Tensor]:
    """Diffusion-DPO objective for rectified-flow winner/loser pairs.

    This is the same error-difference objective used by the official
    Diffusion-DPO and Flow-GRPO DPO implementations. Winner and loser losses
    must be measured with identical noise and flow timesteps.
    """
    tensors = (
        model_winner_mse,
        model_loser_mse,
        reference_winner_mse,
        reference_loser_mse,
    )
    if any(value.ndim != 1 for value in tensors):
        raise ValueError("Flow-DPO pair losses must be one-dimensional")
    if any(value.shape != model_winner_mse.shape for value in tensors[1:]):
        raise ValueError("Flow-DPO winner/loser/reference losses must match")
    if not float(beta) > 0:
        raise ValueError("Flow-DPO beta must be positive")

    model_delta = model_winner_mse - model_loser_mse
    reference_delta = reference_winner_mse.detach() - reference_loser_mse.detach()
    preference_delta = model_delta - reference_delta
    logits = -0.5 * float(beta) * preference_delta
    loss = -F.logsigmoid(logits).mean()
    return {
        "loss": loss,
        "logits": logits.detach(),
        "implicit_accuracy": (logits.detach() > 0).float().mean(),
        "model_delta": model_delta.detach().mean(),
        "reference_delta": reference_delta.detach().mean(),
        "preference_delta": preference_delta.detach().mean(),
    }
