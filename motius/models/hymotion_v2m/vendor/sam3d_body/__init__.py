from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
import sys

sys.modules["sam_3d_body"] = sys.modules[__name__]

__version__ = "1.0.0"

from .sam_3d_body_estimator import SAM3DBodyEstimator
from .build_models import load_sam_3d_body, load_sam_3d_body_hf

__all__ = [
    "__version__",
    "load_sam_3d_body",
    "load_sam_3d_body_hf",
    "SAM3DBodyEstimator",
]
