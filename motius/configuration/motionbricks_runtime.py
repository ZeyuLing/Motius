# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Generation options for MotionBricks models."""

from dataclasses import dataclass

import numpy as np

from motius.configuration.motionbricks_runtime_contract import (
    MODEL_MAX_TOKEN_COUNT,
    MODEL_MIN_TOKEN_COUNT,
)

DEFAULT_TOKEN_COUNTS = tuple(range(8, 15))


@dataclass(frozen=True, slots=True)
class MotionBricksInferenceConfig:
    """Generation settings for output length, heading alignment, and endpoint correction.

    Invalid values raise ValueError.

    Attributes:
        allowed_token_counts: Token counts the model may choose from, each from 6 to 16.
            The default allows 8 to 14.
        correct_generated_diff: Apply endpoint correction, so the motion starts at the
            last context pose and ends at the first target pose. False keeps the
            predicted endpoint poses.
        heading_alignment: Ground direction that inference turns the last context pose to
            face, in model space: (x, z) for SOMA and (x, y) for G1. The default (1, 0)
            faces +X.
    """

    allowed_token_counts: tuple[int, ...] = DEFAULT_TOKEN_COUNTS
    correct_generated_diff: bool = True
    heading_alignment: tuple[float, float] = (1.0, 0.0)

    def __post_init__(self) -> None:
        if not self.allowed_token_counts:
            raise ValueError("allowed_token_counts cannot be empty.")
        if not all(
            MODEL_MIN_TOKEN_COUNT <= count <= MODEL_MAX_TOKEN_COUNT
            for count in self.allowed_token_counts
        ):
            raise ValueError(
                f"allowed_token_counts values must be {MODEL_MIN_TOKEN_COUNT} to {MODEL_MAX_TOKEN_COUNT}."
            )
        if len(self.heading_alignment) != 2:
            raise ValueError("heading_alignment must contain two values.")
        if (
            not np.isfinite(self.heading_alignment).all()
            or np.linalg.norm(self.heading_alignment) <= 1e-06
        ):
            raise ValueError("heading_alignment must be a finite, non-zero direction.")

    def allowed_token_bits(self) -> int:
        """Return the allowed token counts as a bit mask, with bit 0 for 6 tokens."""
        bits = 0
        for value in self.allowed_token_counts:
            bits |= 1 << value - MODEL_MIN_TOKEN_COUNT
        return bits
