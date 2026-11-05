# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Sample prepared AMP observations from native reference-motion data."""
import inspect
import torch


class ProtoMotionsExpertObservations:
    def __init__(
        self, environment, components, input_keys=('historical_max_coords_obs',)
    ):
        self.environment = environment
        self.components = components
        self.input_keys = set(input_keys)

    @torch.no_grad()
    def __call__(self, count):
        environment = self.environment
        motion_ids = environment.motion_manager.sample_n_motion_ids(count)
        motion_times = environment.motion_manager.sample_time(motion_ids)
        if not self.components:
            raise ValueError('AMP requires reference observation components')
        chunks = []
        for start in range(0, count, environment.num_envs):
            end = min(start + environment.num_envs, count)
            context = dict(
                motion_lib=environment.motion_lib,
                motion_ids=motion_ids[start:end],
                motion_times=motion_times[start:end],
                dt=environment.simulator.dt,
                num_state_history_steps=environment.config.num_state_history_steps,
                contact_body_ids=getattr(environment, 'contact_body_ids', None),
            )
            values = {}
            for name, component in self.components.items():
                if name not in self.input_keys:
                    continue
                function = component.get_compute_func()
                available = {**context, **component.get_params().copy()}
                arguments = {}
                for key, parameter in inspect.signature(function).parameters.items():
                    if parameter.kind in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    ):
                        continue
                    if key in available:
                        arguments[key] = available[key]
                    elif parameter.default is inspect.Parameter.empty:
                        raise TypeError(
                            'Missing reference observation argument: ' + key
                        )
                values[name] = function(**arguments).view(end - start, -1)
            chunks.append(values)
        return {
            name: torch.cat([chunk[name] for chunk in chunks], dim=0)
            for name in chunks[0]
        }
