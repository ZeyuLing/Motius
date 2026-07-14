"""Pipeline wrapper for NVIDIA MotionBricks G1 runtime."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionBricksPipeline(BasePipeline):
    """Run MotionBricks as a stateful Unitree G1 motion primitive runtime."""

    BUNDLE_CLS = "motius.models.motionbricks.MotionBricksBundle"

    @property
    def fps(self) -> int:
        return self.bundle.fps

    @property
    def representation(self) -> str:
        return self.bundle.representation

    def build_demo_agent(self, **overrides: Any):
        """Construct the official MotionBricks navigation demo agent."""
        return self.bundle.load_model(**overrides)

    def validate_checkpoints(self) -> None:
        self.bundle.validate_checkpoints()

    @torch.inference_mode()
    def rollout(self, steps: int = 120, *, controller: str | None = None, **overrides: Any) -> dict[str, Any]:
        """Run a headless MotionBricks rollout and return qpos frames.

        This uses the official `navigation_demo` runtime with `has_viewer=0`.
        It requires complete MotionBricks LFS checkpoints.
        """
        if controller is not None:
            overrides["controller"] = controller
        overrides.setdefault("has_viewer", 0)
        agent = self.build_demo_agent(**overrides)
        agent.full_agent.reset()
        qpos_frames = []
        steps = int(steps)
        idle_tail = min(100, max(0, steps // 5))
        for step in range(steps):
            force_idle = idle_tail > 0 and step >= steps - idle_tail
            qpos = agent.full_agent.get_next_frame()
            qpos_frames.append(np.asarray(qpos, dtype=np.float32).copy())
            context_qpos = agent.full_agent.get_context_mujoco_qpos()
            agent.mj_data.qpos[:] = qpos
            control_signals = agent.controller.generate_control_signals(
                None,
                agent.mj_model,
                agent.mj_data,
                visualize=False,
                control_info={"force_idle": force_idle, "allowed_mode": getattr(agent.args, "allowed_mode", None)},
            )
            control_signals["context_mujoco_qpos"] = context_qpos
            agent.full_agent.generate_new_frames(
                control_signals,
                agent.controller.get_controller_dt() * agent.args.generate_dt,
            )
        return {
            "qpos": np.stack(qpos_frames, axis=0),
            "fps": self.fps,
            "representation": self.representation,
        }

    def infer_t2m(self, *args, **kwargs):  # pragma: no cover - not a T2M model
        raise NotImplementedError("MotionBricks is a G1 primitive/runtime pipeline, not a T2M pipeline.")


__all__ = ["MotionBricksPipeline"]
