"""Public helpers for ARDY's explicit motion representation."""

from __future__ import annotations

from collections import OrderedDict


_JOINTS = {"ardy_core330": 27, "ardy_g1_414": 34}


def ardy_feature_slices(representation: str):
    """Return the official feature blocks for an ARDY representation."""
    key = representation.lower().replace("-", "_")
    try:
        joints = _JOINTS[key]
    except KeyError as exc:
        raise KeyError(f"unknown ARDY representation {representation!r}") from exc
    start = 0
    fields = OrderedDict()
    for name, width in (
        ("root_position", 3),
        ("global_root_heading", 2),
        ("root_local_joint_positions", (joints - 1) * 3),
        ("global_joint_rotations_6d", joints * 6),
        ("global_joint_velocities", joints * 3),
        ("foot_contacts", 4),
    ):
        fields[name] = slice(start, start + width)
        start += width
    return fields


def split_ardy_features(features, representation: str):
    """Split ``(..., D)`` ARDY features without copying the underlying data."""
    slices = ardy_feature_slices(representation)
    expected = max(value.stop for value in slices.values())
    if features.shape[-1] != expected:
        raise ValueError(f"{representation} expects {expected} channels, got {features.shape[-1]}")
    return {name: features[..., value] for name, value in slices.items()}


def decode_ardy_features(features, *, motion_rep, is_normalized: bool = True, return_numpy: bool = False):
    """Decode ARDY features through the exact checkpoint motion-rep object."""
    return motion_rep.inverse(
        features,
        is_normalized=is_normalized,
        return_numpy=return_numpy,
    )


__all__ = ["ardy_feature_slices", "split_ardy_features", "decode_ardy_features"]
