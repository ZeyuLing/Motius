# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Centralized lazy initialization for NVIDIA Warp.

Defers wp.init() until first actual kernel use, allowing DDP workers to properly
set up their CUDA context (via torch.cuda.set_device) before Warp initializes.
"""

import os

import warp as wp

wp.config.enable_mempools_at_init = False  # set config before any init

_initialized = False
_fork_hook_registered = False


def _disable_cuda_context_in_child():
    """Called in child process after os.fork().

    After fork, CUDA contexts inherited from the parent are invalid. Warp's context
    manager calls is_cuda_driver_initialized() around every kernel launch (even CPU
    kernels) to save/restore the current CUDA context. Patching it to return False
    prevents CUDA error 3 from appearing in worker stderr.
    """
    import warp._src.context as _wc

    _wc.is_cuda_driver_initialized = lambda: False


def ensure_warp_initialized():
    global _initialized, _fork_hook_registered
    if not _initialized:
        wp.init()
        _initialized = True
    if not _fork_hook_registered:
        if hasattr(os, "register_at_fork"):
            os.register_at_fork(after_in_child=_disable_cuda_context_in_child)
        _fork_hook_registered = True
