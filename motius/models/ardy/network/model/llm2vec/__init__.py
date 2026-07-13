from __future__ import annotations
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified by the Motius project for native package integration.
"""LLM2Vec text encoder and wrapper for ARDY."""

from .llm2vec import LLM2Vec
from .llm2vec_wrapper import LLM2VecEncoder

__all__ = [
    "LLM2Vec",
    "LLM2VecEncoder",
]
