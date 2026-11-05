"""Task API for user-exported BeyondMimic G1 policies."""

from __future__ import annotations

from typing import Any
from pathlib import Path
import numpy as np
import json

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class BeyondMimicPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.beyondmimic.BeyondMimicBundle"

    def __init__(
        self, bundle, *, reference_motion=None, control_hz=None, metadata=None
    ):
        super().__init__(bundle)
        if reference_motion is None:
            root = getattr(bundle, 'artifact_root', None)
            if root is not None and (root / 'reference_motion.npz').is_file():
                reference_motion = root / 'reference_motion.npz'
        if isinstance(reference_motion, (str, Path)):
            with np.load(reference_motion, allow_pickle=False) as data:
                reference_motion = {name: data[name].copy() for name in data.files}
        if reference_motion is not None:
            reference_motion = {
                name: (
                    np.asarray(value, dtype=np.float32)
                    if name
                    in {
                        'joint_pos',
                        'joint_vel',
                        'body_pos_w',
                        'body_quat_w',
                        'body_lin_vel_w',
                        'body_ang_vel_w',
                    }
                    else value
                )
                for name, value in reference_motion.items()
            }
        self.reference_motion = reference_motion
        root = getattr(bundle, 'artifact_root', None)
        saved = root / 'beyondmimic_pipeline.json' if root is not None else None
        options = (
            json.loads(saved.read_text())
            if saved is not None and saved.is_file()
            else {}
        )
        self.control_hz = float(
            options.get('control_hz', 50) if control_hz is None else control_hz
        )
        metadata_path = root / 'tracking_metadata.json' if root is not None else None
        self.metadata = (
            metadata
            if metadata is not None
            else (
                json.loads(metadata_path.read_text())
                if metadata_path is not None and metadata_path.is_file()
                else {}
            )
        )

    def infer_motion_tracking(
        self,
        observation: Any,
        time_step: int,
    ) -> dict[str, Any]:
        """Run one policy step and return action plus embedded reference state."""

        if hasattr(self.bundle, 'predict_policy'):
            import torch

            observations = torch.as_tensor(
                observation, device=self.bundle.device, dtype=torch.float32
            )
            if observations.ndim == 1:
                observations = observations[None]
            with torch.no_grad():
                output = {
                    'actions': self.bundle.predict_policy(
                        self.bundle.normalize_actor(observations)
                    )['mean']
                    .cpu()
                    .numpy()
                }
            if self.reference_motion is not None:
                for name, values in self.reference_motion.items():
                    if name in {
                        'joint_pos',
                        'joint_vel',
                        'body_pos_w',
                        'body_quat_w',
                        'body_lin_vel_w',
                        'body_ang_vel_w',
                    }:
                        index = np.minimum(
                            np.asarray(time_step, dtype=np.int64).reshape(-1),
                            len(values) - 1,
                        )
                        output[name] = values[index]
        else:
            output = dict(self.bundle.forward(observation, time_step))
        output.update(
            {
                "control_hz": self.control_hz,
                "robot": "Unitree G1",
                "method": "BeyondMimic",
            }
        )
        return output

    __call__ = infer_motion_tracking

    def save_pretrained(self, save_directory, **kwargs):
        destination = Path(self.bundle.save_pretrained(save_directory, **kwargs))
        if self.reference_motion is not None:
            np.savez(destination / 'reference_motion.npz', **self.reference_motion)
        (destination / 'beyondmimic_pipeline.json').write_text(
            json.dumps(dict(control_hz=self.control_hz), indent=2) + '\n'
        )
        if self.metadata:
            (destination / 'tracking_metadata.json').write_text(
                json.dumps(self.metadata, indent=2) + '\n'
            )
        return str(destination)


__all__ = ["BeyondMimicPipeline"]
