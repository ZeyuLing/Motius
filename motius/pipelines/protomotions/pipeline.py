"""Task API for the ProtoMotions G1 deployment tracker."""

from __future__ import annotations

from typing import Any, Mapping
from pathlib import Path
import json

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class ProtoMotionsPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.protomotions.ProtoMotionsBundle"

    def __init__(self, bundle, *, deployment=None):
        super().__init__(bundle)
        root = getattr(bundle, "artifact_root", None)
        path = Path(root) / "protomotions_pipeline.json" if root is not None else None
        if deployment is None and path is not None and path.is_file():
            deployment = json.loads(path.read_text())
        self.deployment = deployment

    @classmethod
    def from_original_checkpoint(cls, checkpoint, *, configuration=None, device="cpu"):
        from motius.models.protomotions import ProtoMotionsPolicyBundle
        from motius.configuration.protomotions import original_deployment_options

        configuration = (
            configuration or Path(checkpoint).parent / "resolved_configs_inference.yaml"
        )
        bundle = ProtoMotionsPolicyBundle.from_original_checkpoint(
            checkpoint, configuration=configuration, device=device
        )
        return cls(bundle, deployment=original_deployment_options(configuration))

    def _native_step(self, observations):
        import torch
        from motius.motion.representation import protomotions as representation

        if self.deployment is None:
            raise ValueError(
                "Native ProtoMotions inference requires deployment configuration saved with the Pipeline"
            )
        values = {
            name: torch.as_tensor(value, device=self.bundle.device, dtype=torch.float32)
            for name, value in observations.items()
        }
        paths = {
            "current.dof_pos": "current_dof_pos",
            "current.dof_vel": "current_dof_vel",
            "current.anchor_rot": "current_anchor_rot",
            "current.root_local_ang_vel": "current_root_local_ang_vel",
            "mimic.future_anchor_rot": "mimic_future_anchor_rot",
            "mimic.future_dof_pos": "mimic_future_dof_pos",
            "mimic.future_dof_vel": "mimic_future_dof_vel",
            "historical.processed_actions": "historical_processed_actions",
        }
        prepared = {}
        for key, config in self.deployment["observations"].items():
            parameters = {
                name: value
                for name, value in config.items()
                if name not in {"compute_func", "dynamic_vars"}
            }
            for argument, path in config["dynamic_vars"].items():
                # Optional source inputs are eliminated by the released inference
                # configuration when their corresponding features are disabled.
                name = paths.get(path, path.replace(".", "_"))
                if name in values:
                    parameters[argument] = values[name]
            prepared[key] = getattr(representation, config["compute_func"])(
                **parameters
            )
        action = self.bundle.predict_policy(prepared)["mean"]
        parameters = {
            name: torch.as_tensor(value, device=action.device, dtype=action.dtype)
            for name, value in self.deployment["action"].items()
        }
        converted = representation.bm_pd_action(action, **parameters)
        return dict(
            actions=action.cpu().numpy(),
            joint_pos_targets=converted["processed_action"].cpu().numpy(),
            stiffness_targets=converted["stiffness_targets"].cpu().numpy(),
            damping_targets=converted["damping_targets"].cpu().numpy(),
        )

    def infer_motion_tracking(
        self,
        observations: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Run one 50 Hz policy step using the unified ONNX input contract."""

        if hasattr(self.bundle, "predict_policy"):
            import torch

            with torch.no_grad():
                output = self._native_step(observations)
        else:
            output = dict(self.bundle.forward(observations))
        output.update(
            {
                "control_hz": (
                    self.deployment["control_hz"] if self.deployment is not None else 50
                ),
                "robot": "Unitree G1 29-DOF",
                "method": "ProtoMotions",
            }
        )
        return output

    __call__ = infer_motion_tracking

    def infer_prepared_observations(self, observations, *, source_evaluator=True):
        """Evaluate observations produced by the native vector environment.

        The original Python evaluator samples an action and then applies its
        mean. Preserve that RNG draw when reproducing its simulated rollouts;
        deployment inference has no unused action sampling.
        """
        import torch

        values = {
            key: torch.as_tensor(value, device=self.bundle.device)
            for key, value in observations.items()
        }
        with torch.no_grad():
            policy = self.bundle.predict_policy(values)
            if source_evaluator:
                distribution = torch.distributions.Normal(policy["mean"], policy["std"])
                sampled = distribution.sample()
                # Match the full original AMP evaluation graph, including
                # outputs the evaluator computes before selecting the mean.
                -distribution.log_prob(sampled).sum(dim=-1)
                self.bundle.predict_value(values)
                self.bundle.predict_discriminator(values)
                self.bundle.predict_discriminator_value(values)
        return dict(actions=policy["mean"].cpu().numpy())

    def save_pretrained(self, save_directory, **kwargs):
        destination = Path(self.bundle.save_pretrained(save_directory, **kwargs))
        if self.deployment is not None:
            (destination / "protomotions_pipeline.json").write_text(
                json.dumps(self.deployment, indent=2) + "\n"
            )
            manifest = destination / "model_index.json"
            if manifest.is_file():
                data = json.loads(manifest.read_text())
                data.setdefault("required_files", []).append(
                    "protomotions_pipeline.json"
                )
                manifest.write_text(json.dumps(data, indent=2) + "\n")
        return str(destination)


__all__ = ["ProtoMotionsPipeline"]
