from __future__ import annotations

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from motius.utils.dinov3.utils.dtype import as_torch_dtype
from motius.utils.dinov3.utils.utils import cat_keep_shapes
from motius.utils.dinov3.utils.utils import count_parameters
from motius.utils.dinov3.utils.utils import fix_random_seeds
from motius.utils.dinov3.utils.utils import get_conda_env
from motius.utils.dinov3.utils.utils import get_sha
from motius.utils.dinov3.utils.utils import named_apply
from motius.utils.dinov3.utils.utils import named_replace
from motius.utils.dinov3.utils.utils import uncat_with_shapes
