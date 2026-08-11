# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class Unit(Enum):
    """Length units with their scale relative to meters."""

    METERS = 1.0
    CENTIMETERS = 0.01
    MILLIMETERS = 0.001

    @property
    def meters_per_unit(self) -> float:
        return self.value

    @property
    def unit_name(self) -> str:
        """Lowercase name, e.g. ``'meters'``, ``'centimeters'``."""
        return self.name.lower()

    @classmethod
    def from_name(cls, name: str) -> "Unit":
        """Look up a Unit by lowercase name (e.g. ``'meters'``)."""
        return cls[name.upper()]
