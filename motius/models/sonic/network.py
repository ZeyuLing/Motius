# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""SONIC temporal MLPs, FSQ bottleneck and neural policy/value roots."""
import torch
from torch import nn


class TemporalMLP(nn.Module):
    def __init__(
        self,
        input_dimension,
        hidden_dimensions,
        output_dimension,
        *,
        input_frames=None,
        output_frames=None,
        activation='SiLU',
    ):
        super().__init__()
        self.input_dimension = input_dimension
        self.input_frames = input_frames
        self.output_frames = output_frames
        if activation not in {'SiLU', 'ReLU', 'ELU', 'Tanh', 'GELU'}:
            raise ValueError('Unsupported SONIC neural activation: ' + activation)
        nonlinearity = getattr(nn, activation)()
        layers = []
        previous = input_dimension
        for width in hidden_dimensions:
            layers.extend([nn.Linear(previous, width), nonlinearity])
            previous = width
        layers.append(nn.Linear(previous, output_dimension))
        self.module = nn.Sequential(*layers)

    def forward(self, inputs):
        if self.input_frames is not None:
            inputs = inputs.view(*inputs.shape[:-2], self.input_dimension)
        result = self.module(inputs)
        if self.output_frames is not None:
            result = result.view(
                *result.shape[:-1],
                self.output_frames,
                result.shape[-1] // self.output_frames,
            )
        return result


class RunningObservationNorm(nn.Module):
    def __init__(self, dimension, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.frozen = False
        self.register_buffer(
            'running_mean', torch.zeros(dimension, dtype=torch.float32)
        )
        self.register_buffer('running_var', torch.ones(dimension, dtype=torch.float32))
        self.register_buffer('count', torch.ones((), dtype=torch.float32))

    def forward(self, inputs):
        shape = inputs.shape
        if inputs.ndim == 3:
            inputs = inputs.reshape(-1, shape[-1])
        mean = self.running_mean.view(1, -1)
        var = self.running_var.view(1, -1)
        result = inputs - mean
        result = result / torch.sqrt(var + self.epsilon)
        result = torch.clamp(result, min=-5.0, max=5.0)
        if self.training and not self.frozen:
            batch_mean = inputs.mean([0])
            batch_var = inputs.var([0])
            batch_count = inputs.size()[0]
            delta = batch_mean - self.running_mean
            total = self.count + batch_count
            new_mean = self.running_mean + delta * batch_count / total
            first = self.running_var * self.count
            second = batch_var * batch_count
            moment = first + second + delta**2 * self.count * batch_count / total
            self.running_mean = new_mean
            self.running_var = moment / total
            self.count = total
        return result.view(shape) if len(shape) == 3 else result


class SonicValueNetwork(nn.Module):
    def __init__(self, network_options, *, normalize=True):
        super().__init__()
        self.network = TemporalMLP(**network_options)
        self.normalizer = (
            RunningObservationNorm(network_options['input_dimension'])
            if normalize
            else nn.Identity()
        )

    def forward(self, observations):
        with torch.no_grad():
            observations = self.normalizer(observations)
        return self.network(observations)


class SonicPolicyNetwork(nn.Module):
    def __init__(self, encoders, decoders, levels, *, num_actions=29, initial_std=0.05):
        super().__init__()
        from vector_quantize_pytorch import FSQ

        # The original Actor constructs its backbone before its noise parameter.
        self.encoders = nn.ModuleDict(
            {name: TemporalMLP(**options) for name, options in encoders.items()}
        )
        self.quantizer = FSQ(levels=levels)
        self.decoders = nn.ModuleDict(
            {name: TemporalMLP(**options) for name, options in decoders.items()}
        )
        self.std = nn.Parameter(initial_std * torch.ones(num_actions))

    def encode(self, features, encoder_masks, batch, sequence):
        """Neural mixture of modality encoders and shared FSQ/action decoders.

        Feature packing, observation history and mode selection belong to the
        caller. Inputs retain the original leading batch/sequence dimensions.
        """
        tokens = {}
        latents = {}
        for name, network in self.encoders.items():
            mask = encoder_masks.get(name)
            value = features[name]
            value = value.view(-1, *value.shape[2:])
            if mask is not None:
                value = value[mask]
            latent = network(value)
            codes, _ = self.quantizer(latent)
            tokens[name] = codes.contiguous()
            latents[name] = latent
        if not tokens:
            raise ValueError('No active SONIC modality encoder')
        first = tokens[next(iter(tokens))]
        if len(tokens) == 1:
            combined = first.view(batch, sequence, *first.shape[1:])
        else:
            combined = torch.zeros(
                (batch * sequence, *first.shape[1:]),
                dtype=first.dtype,
                device=first.device,
            )
            for name, value in tokens.items():
                combined[encoder_masks[name]] = value
            combined = combined.view(batch, sequence, *first.shape[1:])
        return combined, tokens, latents

    def decode_actions(self, tokens, proprioception):
        action_input = torch.cat([tokens.flatten(start_dim=-2), proprioception], dim=-1)
        return self.decoders['g1_dyn'](action_input)

    def forward(
        self, features, proprioception, encoder_masks, *, decode_reconstruction=True
    ):
        combined, tokens, latents = self.encode(
            features, encoder_masks, *proprioception.shape[:2]
        )
        result = dict(
            mean=self.decode_actions(combined, proprioception),
            tokens=combined,
            encoded_tokens=tokens,
            encoded_latents=latents,
            std=self.std,
        )
        if decode_reconstruction and 'g1_kin' in self.decoders:
            result['reconstruction'] = self.decoders['g1_kin'](combined)
        return result
