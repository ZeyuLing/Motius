# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Task/style GAE and the published G1 advantage normalization protocol."""
import torch


def discounted_advantages(dones, values, rewards, next_values, gamma=0.99, tau=0.95):
    accumulated = 0
    advantages = torch.zeros_like(rewards)
    for index in reversed(range(rewards.shape[0])):
        continuation = 1.0 - dones[index]
        delta = rewards[index] + gamma * next_values[index] - values[index]
        accumulated = delta + gamma * tau * continuation * accumulated
        advantages[index] = accumulated
    return advantages


class ProtoMotionsAdvantageNormalization:
    def __init__(
        self,
        device,
        *,
        enabled=True,
        shift_mean=True,
        use_ema=True,
        ema_alpha=0.05,
        min_std=0.02,
        clamp_range=4.0,
    ):
        self.enabled = enabled
        self.shift_mean = shift_mean
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.min_std = min_std
        self.clamp_range = clamp_range
        self.mean = torch.zeros(1, device=device, dtype=torch.float32)
        self.std = torch.ones(1, device=device, dtype=torch.float32)

    @torch.no_grad()
    def __call__(self, advantages):
        if not self.enabled:
            return advantages
        if not self.use_ema:
            mean = advantages.mean()
            var = advantages.var()
            if (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            ):
                from motius.utils.distributed_moments import gather_moments

                mean, var, _ = gather_moments(mean, var, advantages.numel())
                torch.distributed.broadcast(mean, src=0)
                torch.distributed.broadcast(var, src=0)
            return (advantages - mean if self.shift_mean else advantages) / (
                torch.sqrt(var) + 1e-8
            )
        safe_std = torch.clamp(self.std, min=self.min_std)
        normalized = (
            advantages - self.mean if self.shift_mean else advantages
        ) / safe_std
        clipped = torch.clamp(normalized, -self.clamp_range, self.clamp_range)
        denormalized = (
            clipped * safe_std + self.mean if self.shift_mean else clipped * safe_std
        )
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            self.mean = (
                self.ema_alpha * denormalized.mean() + (1 - self.ema_alpha) * self.mean
            )
            self.std = (
                self.ema_alpha * (denormalized.std() + 1e-8)
                + (1 - self.ema_alpha) * self.std
            )
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(self.mean, src=0)
            torch.distributed.broadcast(self.std, src=0)
        return clipped

    def state_dict(self):
        return dict(mean=self.mean, std=self.std)

    def load_state_dict(self, state):
        self.mean.copy_(state['mean'])
        self.std.copy_(state['std'])


def task_style_advantages(
    rollout, normalization, *, gamma=0.99, tau=0.95, task_weight=0.5, style_weight=2.0
):
    values = rollout['value'].squeeze(-1)
    task = discounted_advantages(
        rollout['dones'],
        values,
        rollout['rewards'],
        rollout['next_value'].squeeze(-1),
        gamma,
        tau,
    )
    disc_values = rollout['disc_value'].squeeze(-1)
    style = discounted_advantages(
        rollout['dones'],
        disc_values,
        rollout['amp_rewards'],
        rollout['next_disc_value'].squeeze(-1),
        gamma,
        tau,
    )
    return dict(
        returns=task + values,
        disc_returns=style + disc_values,
        advantages=normalization(task * task_weight + style * style_weight),
    )
