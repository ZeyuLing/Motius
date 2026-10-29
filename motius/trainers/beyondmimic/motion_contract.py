"""Compatibility import for the dataset-owned BeyondMimic reference contract."""

from motius.datasets.beyondmimic_motion_contract import (
    MOTION_ARRAY_KEYS,
    load_motion_npz,
)

__all__ = ['MOTION_ARRAY_KEYS', 'load_motion_npz']
