"""Closed-loop runner shared by all physical simulation backends."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from motius.simulators.reference import TrackingReference, load_g1_reference


@dataclass
class TrackingRolloutResult:
    method: str
    backend: str
    protocol_id: str
    reference_name: str
    fps: float
    qpos: np.ndarray
    reference_qpos: np.ndarray
    actions: np.ndarray
    metrics: dict[str, Any]
    termination_reason: Optional[str] = None

    def save(self, output: str | Path) -> str:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            qpos=np.asarray(self.qpos, dtype=np.float32),
            reference_qpos=np.asarray(self.reference_qpos, dtype=np.float32),
            actions=np.asarray(self.actions, dtype=np.float32),
            fps=np.asarray([self.fps], dtype=np.float32),
            method=np.asarray(self.method),
            backend=np.asarray(self.backend),
            protocol_id=np.asarray(self.protocol_id),
            reference_name=np.asarray(self.reference_name),
            metrics_json=np.asarray(json.dumps(self.metrics, sort_keys=True)),
            termination_reason=np.asarray(self.termination_reason or ""),
        )
        return str(path)


def _resolve_reference(
    reference_motion: TrackingReference | str | Path,
    *,
    target_fps: float,
    source_fps: Optional[float],
) -> TrackingReference:
    if isinstance(reference_motion, TrackingReference):
        if not np.isclose(reference_motion.fps, target_fps):
            raise ValueError(
                f"Reference is {reference_motion.fps} Hz but backend requires "
                f"{target_fps} Hz. Reload it with load_g1_reference(..., "
                f"target_fps={target_fps})."
            )
        return reference_motion
    return load_g1_reference(
        reference_motion,
        source_fps=source_fps,
        target_fps=target_fps,
    )


def run_motion_tracking_rollout(
    pipeline: Any,
    reference_motion: TrackingReference | str | Path,
    *,
    simulator: str = "mujoco",
    source_fps: Optional[float] = None,
    max_steps: Optional[int] = None,
    render: bool = False,
    output_path: Optional[str | Path] = None,
    **environment_kwargs: Any,
) -> TrackingRolloutResult:
    """Run a physical tracking rollout instead of a single policy forward."""

    backend = simulator.strip().lower().replace("_", "-")
    if backend == "mujoco":
        from motius.simulators.mujoco import MujocoG1TrackingEnvironment

        target_fps = float(environment_kwargs.pop("control_hz", 50.0))
        if not np.isclose(target_fps, 50.0):
            raise ValueError(f"MuJoCo G1 tracking uses a fixed 50 Hz control rate, got {target_fps}.")
        reference = _resolve_reference(
            reference_motion,
            target_fps=target_fps,
            source_fps=source_fps,
        )
        method = str(getattr(pipeline.bundle, "METHOD_NAME", "")).lower()
        if method == "humanoidgpt" and "scene_path" not in environment_kwargs:
            environment_kwargs["scene_path"] = pipeline.bundle.file_paths["scene"]
        environment = MujocoG1TrackingEnvironment(
            reference=reference,
            render=render,
            **environment_kwargs,
        )
    elif backend in {"isaac-lab", "isaaclab"}:
        from motius.simulators.isaaclab import create_isaaclab_tracking_environment

        target_fps = float(environment_kwargs.pop("control_hz", 50.0))
        reference = _resolve_reference(
            reference_motion,
            target_fps=target_fps,
            source_fps=source_fps,
        )
        environment = create_isaaclab_tracking_environment(
            reference=reference,
            render=render,
            **environment_kwargs,
        )
    else:
        raise ValueError(
            f"Unknown motion-tracking simulator {simulator!r}; use 'mujoco' or "
            "'isaaclab'."
        )

    try:
        environment.reset()
        steps = 0
        while not environment.done and (max_steps is None or steps < max_steps):
            inputs = environment.policy_inputs(pipeline)
            output: Mapping[str, Any] = pipeline.infer_motion_tracking(**inputs)
            environment.step(pipeline, output)
            steps += 1
        result = environment.result()
    finally:
        environment.close()

    if max_steps is not None and result.termination_reason is None:
        completed = int(result.metrics.get("completed_steps", 0))
        horizon = min(int(max_steps), reference.num_frames - 1)
        if completed >= horizon:
            result.metrics["total_steps"] = horizon
            result.metrics["survival_rate"] = 1.0
            result.metrics["success_rate"] = 1.0

    if output_path is not None:
        result.save(output_path)
    return result
