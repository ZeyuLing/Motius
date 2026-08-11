"""Motius pipeline for the frozen dense D3 source-only repair policy."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from motius.motion.repair import (
    DenseD3RepairConfig,
    MotionRepairResult,
    repair_motion135,
)
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionRepairPipeline:
    """Training-free ``motion_repair`` task pipeline.

    This pipeline intentionally has no model bundle or checkpoint: the frozen
    policy operates only on the source motion and fixed SMPL-22 offsets.
    """

    def __init__(
        self,
        config: DenseD3RepairConfig | Mapping[str, Any] | None = None,
        **config_overrides: Any,
    ) -> None:
        if config is None:
            values: dict[str, Any] = {}
        elif isinstance(config, DenseD3RepairConfig):
            if config_overrides:
                raise ValueError(
                    "config_overrides cannot be combined with a config instance."
                )
            self.config = config
            return
        elif isinstance(config, Mapping):
            values = dict(config)
        else:
            raise TypeError(
                "config must be a DenseD3RepairConfig, mapping, or None."
            )
        values.update(config_overrides)
        self.config = DenseD3RepairConfig(**values)

    def infer_motion_repair(
        self,
        motion135: np.ndarray,
        bone_offsets: np.ndarray,
    ) -> MotionRepairResult:
        return repair_motion135(motion135, bone_offsets, self.config)

    __call__ = infer_motion_repair


__all__ = ["MotionRepairPipeline"]
