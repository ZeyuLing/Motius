# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 components components; see licenses/sonic."""
import os
import time

timer_indent_level = 0


class Timer:

    def __init__(self, name="", instance_enabled=True):
        self.name = name
        self.start_time = None
        self.enabled = instance_enabled and os.environ.get("TIMER_ENABLED", "0") == "1"
        if "LOCAL_RANK" in os.environ:
            self.rank = int(os.environ["LOCAL_RANK"])
        else:
            self.rank = 0
        self.show_rank = os.environ.get("TIMER_SHOW_RANK", "0") == "1"
        self.rank_zero_only = os.environ.get("TIMER_RANK_ZERO_ONLY", "0") == "1"

    def __enter__(self):
        if not self.enabled or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        self.start_time = time.perf_counter()
        self.current_indent = timer_indent_level
        timer_indent_level += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            return False
        if not self.enabled or (self.rank_zero_only and self.rank != 0):
            return self
        global timer_indent_level
        elapsed_time = time.perf_counter() - self.start_time
        indent = "    " * self.current_indent
        rank_str = f"[rank{self.rank}] " if self.show_rank else ""
        print(f"{indent}{rank_str}[{self.name}] time: {elapsed_time:.4f} seconds")
        timer_indent_level -= 1
