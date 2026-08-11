from __future__ import annotations
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by the Motius project for native package integration.
"""ARDY model package: main model class, text encoders, and loading utilities."""

from .ardy_model import Ardy
from .load_model import load_model
from .loading import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    DEFAULT_TEXT_ENCODER_URL,
    MODEL_NAMES,
    load_checkpoint_state_dict,
)

# from .twostage_denoiser import TwostageDenoiser

__all__ = [
    "Ardy",
    # "TwostageDenoiser",
    "load_model",
    "load_checkpoint_state_dict",
    "AVAILABLE_MODELS",
    "DEFAULT_MODEL",
    "DEFAULT_TEXT_ENCODER_URL",
    "MODEL_NAMES",
]


def __getattr__(name):
    if name == "LLM2VecEncoder":
        from .llm2vec import LLM2VecEncoder

        return LLM2VecEncoder
    raise AttributeError(name)
