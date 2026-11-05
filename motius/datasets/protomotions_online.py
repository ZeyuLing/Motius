# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Process-local AMP replay and original PPO minibatch ordering."""
import numpy as np
import torch


class AMPReplay:
    def __init__(self, capacity, device):
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.head = 0
        self.full = False
        self.buffers = {}

    def __len__(self):
        return self.capacity if self.full else self.head

    def store(self, values):
        count = next(iter(values.values())).shape[0]
        if count > self.capacity:
            raise ValueError('AMP replay insertion exceeds its capacity')
        for key, value in values.items():
            if value.shape[0] != count:
                raise ValueError('AMP replay columns must have equal lengths')
            if key not in self.buffers:
                self.buffers[key] = torch.zeros(
                    (self.capacity, *value.shape[1:]),
                    dtype=value.dtype,
                    device=self.device,
                )
            if self.head + count >= self.capacity:
                split = self.capacity - self.head
                self.buffers[key][self.head :] = value[:split].clone()
                self.buffers[key][: count - split] = value[split:].clone()
                self.full = True
            else:
                self.buffers[key][self.head : self.head + count] = value.clone()
        self.head = (self.head + count) % self.capacity

    def sample(self, count):
        indices = torch.randint(0, len(self), (count,), device=self.device)
        return {key: value[indices].clone() for key, value in self.buffers.items()}

    def update(self, values, *, keep_probability=0.01):
        count = next(iter(values.values())).shape[0]
        if len(self) >= self.capacity:
            probabilities = torch.ones(count, device=self.device) * keep_probability
            keep = torch.bernoulli(probabilities) == 1.0
            values = {key: value[keep] for key, value in values.items()}
            count = keep.sum().item()
        if count > self.capacity:
            # Preserve the original CPU permutation and its separate RNG stream.
            indices = torch.randperm(count)[: self.capacity]
            values = {key: value[indices] for key, value in values.items()}
        self.store(values)

    def state_dict(self):
        return dict(head=self.head, full=self.full, buffers=self.buffers)

    def load_state_dict(self, state):
        self.head = int(state['head'])
        self.full = bool(state['full'])
        self.buffers = {
            key: value.to(self.device) for key, value in state['buffers'].items()
        }


class ProtoMotionsMinibatches:
    """Keep source NumPy shuffling and modulo indexing of shorter expert data."""

    def __init__(self, values, batch_size, mini_epochs=2):
        self.values = values
        self.batch_size = int(batch_size)
        self.mini_epochs = int(mini_epochs)
        self.length = max(value.shape[0] for value in values.values())
        if self.length % self.batch_size:
            raise ValueError(
                'ProtoMotions rollout length must be divisible by minibatch size'
            )
        self.indices = np.random.permutation(self.length)

    def __iter__(self):
        count = self.length // self.batch_size
        for index in range(count * self.mini_epochs):
            position = index % count
            if position == 0 and index != 0:
                self.indices = np.random.permutation(self.length)
            selected = self.indices[
                position * self.batch_size : (position + 1) * self.batch_size
            ]
            yield index, {
                key: value[selected % value.shape[0]]
                for key, value in self.values.items()
            }


def prepare_amp_minibatches(
    rollout,
    replay,
    expert_provider,
    *,
    batch_size,
    mini_epochs=2,
    discriminator_batch_size=4096,
    replay_keep_probability=0.01,
    discriminator_keys=('historical_max_coords_obs',),
):
    """Augment flattened rollouts before the first training-mode neural call."""
    values = dict(rollout)
    count = next(iter(values.values())).shape[0]
    observed = {key: values[key] for key in discriminator_keys}
    historical = replay.sample(count) if len(replay) else observed
    expert_count = min(
        count, discriminator_batch_size * (count * mini_epochs // batch_size)
    )
    expert = expert_provider(expert_count)
    for key in discriminator_keys:
        values['agent_' + key] = observed[key]
        values['replay_' + key] = historical[key]
        values['expert_' + key] = expert[key]
    replay.update(observed, keep_probability=replay_keep_probability)
    return ProtoMotionsMinibatches(values, batch_size, mini_epochs)
