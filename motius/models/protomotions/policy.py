# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Neural MLP and running-normalization layers used by the G1 tracker."""
import torch
from torch import nn
from collections.abc import Mapping


class RunningObservationNorm(nn.Module):
    def __init__(self, dimension, *, clamp=5.0, epsilon=1e-5, ema_decay=0.999):
        super().__init__()
        self.clamp = clamp
        self.epsilon = epsilon
        self.ema_decay = ema_decay
        self.register_buffer('mean', torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer('var', torch.ones(dimension, dtype=torch.float64))
        self.register_buffer('count', torch.ones((), dtype=torch.long))
        self.freeze_running = False

    def forward(self, observations):
        normalized = (observations - self.mean.float()) / torch.sqrt(
            self.var.float() + self.epsilon
        )
        if self.clamp is not None:
            normalized = torch.clamp(normalized, -self.clamp, self.clamp)
        if self.training and not self.freeze_running:
            self.update(observations.detach())
        return normalized

    @torch.no_grad()
    def update(self, observations):
        mean = torch.mean(observations, dim=0)
        var = torch.var(observations, dim=0, unbiased=False)
        batch_size = observations.shape[0]
        distributed = (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )
        if distributed:
            from motius.utils.distributed_moments import gather_moments

            mean, var, batch_size = gather_moments(mean, var, batch_size)
        if self.ema_decay is None:
            count = batch_size
            # Source combine_moments casts counts to FP32 before its arithmetic.
            old_count = self.count.float()
            batch_count = torch.tensor(count, device=observations.device).float()
            total = old_count + batch_count
            delta = mean - self.mean
            new_mean = self.mean + delta * (batch_count / total)
            moment = (
                self.var * old_count
                + var * batch_count
                + (delta**2) * (old_count * batch_count / total)
            )
            self.mean.copy_(new_mean)
            self.var.copy_(moment / total)
            self.count.fill_(total)
        else:
            decay = self.ema_decay
            self.mean[:] = decay * self.mean + (1 - decay) * mean
            self.var[:] = decay * self.var + (1 - decay) * var
        if distributed:
            torch.distributed.broadcast(self.mean, src=0)
            torch.distributed.broadcast(self.var, src=0)
            torch.distributed.broadcast(self.count, src=0)


class ObservationMLP(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimensions,
        output_dimension,
        *,
        input_keys,
        activation='relu',
        normalize=True,
        clamp=5.0,
        epsilon=1e-5,
        ema_decay=0.999,
        layer_norm=False,
    ):
        super().__init__()
        self.input_keys = list(input_keys)
        self.norm = (
            RunningObservationNorm(
                input_dimension, clamp=clamp, epsilon=epsilon, ema_decay=ema_decay
            )
            if normalize
            else nn.Identity()
        )
        activations = dict(
            relu=nn.ReLU, elu=nn.ELU, tanh=nn.Tanh, gelu=nn.GELU, silu=nn.SiLU
        )
        if activation not in activations:
            raise ValueError('Unsupported neural activation: ' + activation)
        layers = []
        current = input_dimension
        for index, width in enumerate(hidden_dimensions):
            layers.append(nn.Linear(current, width))
            if layer_norm and index == 0:
                layers.append(nn.LayerNorm(width))
            layers.append(activations[activation]())
            current = width
        layers.append(nn.Linear(current, output_dimension))
        self.mlp = nn.Sequential(*layers)

    def forward(self, observations, *, return_normalized=False):
        values = (
            torch.cat([observations[key] for key in self.input_keys], dim=-1)
            if isinstance(observations, Mapping)
            else observations
        )
        # The source flattens only for normalization, then restores dimensions
        # before applying its MLP.
        shape = values.shape
        normalized = self.norm(values.reshape(-1, shape[-1]))
        if values.ndim > 2:
            normalized = normalized.reshape(*shape[:-1], -1)
        output = self.mlp(normalized)
        return (output, normalized) if return_normalized else output


class GaussianActor(nn.Module):
    def __init__(self, mean_options, *, initial_log_std=-2.9, learnable_std=True):
        super().__init__()
        self.logstd = nn.Parameter(
            torch.ones(mean_options['output_dimension']) * initial_log_std,
            requires_grad=learnable_std,
        )
        self.mean = ObservationMLP(**mean_options)

    def forward(self, observations):
        mean = self.mean(observations)
        return dict(mean=mean, std=mean * 0 + torch.exp(self.logstd))
