# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""
Simple replacement for hydra.utils functions to avoid the heavy hydra-core dependency.
Provides get_class and instantiate functions with compatible APIs.
"""
import importlib
from typing import Any


def get_class(path: str) -> type:
    """
    Import and return a class from a string path.

    Args:
        path: Fully qualified class path, e.g., "torch.optim.Adam"

    Returns:
        The class object

    Example:
        >>> Adam = get_class("torch.optim.Adam")
        >>> optimizer = Adam(params, lr=0.001)
    """
    from motius.configuration.protomotions.environment import resolve_environment_target

    return resolve_environment_target(path)


def instantiate(config, **kwargs) -> Any:
    """
    Instantiate a class from a config object.

    Args:
        config: Config object with _target_ attribute specifying the class path,
                or a dict with '_target_' key
        **kwargs: Additional keyword arguments to pass to the constructor,
                  overriding config values

    Returns:
        Instance of the specified class

    Example:
        >>> class Config:
        ...     _target_ = "torch.optim.Adam"
        ...     lr = 0.001
        >>> optimizer = instantiate(config, params=model.parameters())
    """
    if isinstance(config, dict):
        target = config.get("_target_")
        config_dict = {k: v for k, v in config.items() if k != "_target_"}
    else:
        target = getattr(config, "_target_", None)
        config_dict = {
            k: v
            for k, v in vars(config).items()
            if k != "_target_" and (not k.startswith("_"))
        }
    if target is None:
        raise ValueError(
            "Config must have a '_target_' attribute or key specifying the class path"
        )
    cls = get_class(target)
    merged_kwargs = {**config_dict, **kwargs}
    return cls(**merged_kwargs)
