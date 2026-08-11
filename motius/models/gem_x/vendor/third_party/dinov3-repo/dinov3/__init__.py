from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import sys

sys.modules["dinov3"] = sys.modules[__name__]

__version__ = "0.0.1"
