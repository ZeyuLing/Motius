# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""BeyondMimic's feedforward actor/critic architecture (RSL-RL 2.3.1)."""
import torch
from torch import nn


def resolve_nn_activation(name):
    activations = dict(
        elu=nn.ELU,
        selu=nn.SELU,
        relu=nn.ReLU,
        crelu=nn.ReLU,
        lrelu=nn.LeakyReLU,
        tanh=nn.Tanh,
        sigmoid=nn.Sigmoid,
    )
    if name not in activations:
        raise ValueError('Unsupported neural activation: ' + name)
    return activations[name]()


class BeyondMimicActorCritic(nn.Module):
    def __init__(
        self,
        num_actor_obs,
        num_critic_obs,
        num_actions,
        actor_hidden_dims=[256, 256, 256],
        critic_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        **kwargs,
    ):
        if kwargs:
            raise TypeError("Unexpected architecture options: " + str(sorted(kwargs)))
        super().__init__()
        activation = resolve_nn_activation(activation)

        mlp_input_dim_a = num_actor_obs
        mlp_input_dim_c = num_critic_obs
        # Policy
        actor_layers = []
        actor_layers.append(nn.Linear(mlp_input_dim_a, actor_hidden_dims[0]))
        actor_layers.append(activation)
        for layer_index in range(len(actor_hidden_dims)):
            if layer_index == len(actor_hidden_dims) - 1:
                actor_layers.append(
                    nn.Linear(actor_hidden_dims[layer_index], num_actions)
                )
            else:
                actor_layers.append(
                    nn.Linear(
                        actor_hidden_dims[layer_index],
                        actor_hidden_dims[layer_index + 1],
                    )
                )
                actor_layers.append(activation)
        self.actor = nn.Sequential(*actor_layers)

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(mlp_input_dim_c, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for layer_index in range(len(critic_hidden_dims)):
            if layer_index == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[layer_index], 1))
            else:
                critic_layers.append(
                    nn.Linear(
                        critic_hidden_dims[layer_index],
                        critic_hidden_dims[layer_index + 1],
                    )
                )
                critic_layers.append(activation)
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(
                torch.log(init_noise_std * torch.ones(num_actions))
            )
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'"
            )

    def forward(self, observations, critic_observations=None):
        output = {}
        if observations is not None:
            mean = self.actor(observations)
            standard_deviation = (
                self.std.expand_as(mean)
                if self.noise_std_type == 'scalar'
                else self.log_std.exp().expand_as(mean)
            )
            output.update(mean=mean, std=standard_deviation)
        if critic_observations is not None:
            output['value'] = self.critic(critic_observations)
        return output
