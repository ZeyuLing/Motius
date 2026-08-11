# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Config hydration: env vars, _target_ resolution, and recursive instantiation."""

import importlib
import os


def get_env_var(name: str, default=None):
    """Read env var by name and by lowercased name; return default if neither set."""
    return os.getenv(name, os.getenv(name.lower(), default))


def resolve_target(target: str):
    """Import module and return the attribute named by a dotted path (e.g. 'pkg.mod.Class')."""
    module_name, attr_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def materialize_value(value):
    """Recursively turn dicts with '_target_' into instances; lists/dicts traversed; leaves
    unchanged."""
    if isinstance(value, dict):
        if "_target_" in value:
            return instantiate_from_dict(value)
        return {k: materialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [materialize_value(v) for v in value]
    return value


def instantiate_from_dict(node, overrides=None):
    """Build an instance from a config dict: '_target_' gives the class, other keys are kwargs; overrides merged in."""
    if not isinstance(node, dict) or "_target_" not in node:
        raise ValueError("Config node must be a dict with a '_target_' key.")

    target = resolve_target(node["_target_"])
    kwargs = {}
    for key, value in node.items():
        if key == "_target_":
            continue
        kwargs[key] = materialize_value(value)

    if overrides:
        kwargs.update({k: v for k, v in overrides.items() if v is not None})

    return target(**kwargs)
