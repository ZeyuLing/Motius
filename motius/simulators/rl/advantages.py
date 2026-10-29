# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Published GAE and time-major rollout batching outside neural modules."""
import torch


def generalized_advantage_estimation(
    rewards, dones, values, last_values, *, gamma=0.99, lam=0.95, normalize=True
):
    returns = torch.zeros_like(values)
    advantage = 0
    for step in reversed(range(rewards.shape[0])):
        next_values = last_values if step == rewards.shape[0] - 1 else values[step + 1]
        next_is_not_terminal = 1.0 - dones[step].float()
        delta = (
            rewards[step] + next_is_not_terminal * gamma * next_values - values[step]
        )
        advantage = delta + next_is_not_terminal * gamma * lam * advantage
        returns[step] = advantage + values[step]
    advantages = returns - values
    if normalize:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    return returns, advantages


def rollout_minibatches(rollout, num_mini_batches, num_epochs):
    observations = rollout['observations']
    batch_size = observations.shape[0] * observations.shape[1]
    mini_batch_size = batch_size // num_mini_batches
    if mini_batch_size < 1:
        raise ValueError('The rollout is smaller than the number of minibatches')
    indices = torch.randperm(
        num_mini_batches * mini_batch_size,
        requires_grad=False,
        device=observations.device,
    )
    flat = {key: value.flatten(0, 1) for key, value in rollout.items()}
    for epoch in range(num_epochs):
        for index in range(num_mini_batches):
            selected = indices[index * mini_batch_size : (index + 1) * mini_batch_size]
            yield {key: value[selected] for key, value in flat.items()}
