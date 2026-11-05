"""Native batched pose-conditioned inference for released SOMA and G1 models.

Raw features use upstream model space. Boundary trimming is optional and does
not perform the endpoint correction of NVIDIA's interactive runtime.
"""

import numpy as np
import torch
from pathlib import Path

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionBricksReleasePipeline(BasePipeline):
    BUNDLE_CLS = "motius.models.motionbricks.release_bundle.MotionBricksReleaseBundle"
    fps = 30

    def __init__(self, bundle):
        super().__init__(bundle)
        from motius.configuration.motionbricks_release import representation
        from motius.pipelines.motionbricks.release_g1_inference import G1Inference
        from motius.pipelines.motionbricks.release_soma_inference import SOMAInference

        rep = representation(bundle.model, bundle.mean, bundle.std).to(bundle.device)
        self.inference = (SOMAInference if bundle.model == "soma" else G1Inference)(
            bundle, rep, bundle.options["pose_args"], bundle.options["root_args"]
        )
        self.config = dict(
            num_inference_step=1,
            smooth_root_traj=False,
            allow_pred_out_of_reach_num_tokens=False,
            pose_token_sampling_use_argmax=True,
            skip_ending_target_cond=False,
            use_onnx_dynamic_batch_root_conversion=True,
            final_root_pred_mode="from_height_constraint_violation",
        )
        asset = (
            Path(__file__).resolve().parents[2]
            / "configuration/motionbricks_assets"
            / f"{bundle.model}_model_rig.npz"
        )
        with np.load(asset, allow_pickle=False) as data:
            arrays = dict(data)
        if bundle.model == "g1":
            self.config.pop("final_root_pred_mode")
            from motius.motion.body_models.motionbricks_release_g1 import (
                G1PlacedPosePreprocess,
                G1PlacedPosePostprocess,
            )

            self.pre = G1PlacedPosePreprocess(
                arrays, rep.dual_rep.global_motion_rep.indices
            ).to(bundle.device)
            self.post = G1PlacedPosePostprocess(
                arrays, rep.dual_rep.global_motion_rep.indices
            ).to(bundle.device)
        else:
            self.capture_offsets = torch.from_numpy(arrays["capture_offsets"]).to(
                bundle.device
            )
            self.standard_offsets = torch.from_numpy(arrays["standard_offsets"]).to(
                bundle.device
            )
            self.local_rotation_slice = slice(87, 267)
            self.global_rotation_slice = slice(92, 272)

    @torch.inference_mode()
    def infer_raw(self, inputs):
        """Return native tensors in the published raw graph contract (no endpoint correction)."""
        from motius.motion.conditioning.motionbricks_release import (
            validate_release_inputs,
        )

        values = validate_release_inputs(inputs, self.bundle.model, self.bundle.device)
        if self.bundle.model == "soma":
            from .release_soma_pose import generate_soma

            motion, tokens = generate_soma(self, **values)
            return {"pred_global_motions": motion, "pred_num_tokens": tokens}
        joint_inputs = self.pre(**values)
        global_root, has_global, local_root, has_local, poses, has_poses, allowed = (
            joint_inputs
        )
        motion, tokens = self.inference.predict(
            global_root,
            has_global.bool(),
            local_root,
            has_local.bool(),
            poses,
            has_poses.bool(),
            num_tokens=None,
            allowed_pred_num_tokens=allowed,
            config=self.config,
            info={},
        )
        qpos, tokens, contacts = self.post(motion, tokens.to(torch.int32))
        return {
            "pred_qpos": qpos,
            "pred_num_tokens": tokens,
            "pred_foot_contacts": contacts,
        }

    def infer_pose_inbetweening(self, inputs, *, trim_boundaries=True):
        """Accept the documented ONNX tensors, return one motion per request."""
        outputs = {
            name: value.detach().cpu().numpy()
            for name, value in self.infer_raw(inputs).items()
        }
        counts = np.asarray(outputs["pred_num_tokens"])
        if counts.ndim != 1 or not np.isin(counts, np.arange(6, 17)).all():
            raise ValueError("Model returned invalid token counts")
        lengths = counts.astype(np.int64) * 4
        is_g1 = self.bundle.model == "g1"
        key = "pred_qpos" if is_g1 else "pred_global_motions"
        motions, contacts = [], []
        for i, length in enumerate(lengths):
            section = (
                slice(3, int(length) - 3) if trim_boundaries else slice(0, int(length))
            )
            motions.append(outputs[key][i, section].copy())
            contact = (
                outputs["pred_foot_contacts"][i] if is_g1 else outputs[key][i, :, -4:]
            )
            contacts.append(contact[section].copy())
        return dict(
            motions=motions,
            foot_contacts=contacts,
            fps=self.fps,
            lengths=lengths - (6 if trim_boundaries else 0),
            pred_num_tokens=counts,
            representation="g1_qpos_xyzw" if is_g1 else "motionbricks_soma_366",
            endpoint_corrected=False,
        )

    __call__ = infer_pose_inbetweening

    def infer_g1_inbetweening(
        self,
        context_qpos,
        target_qpos,
        *,
        quaternion_order="wxyz",
        allowed_pred_num_tokens=None,
        trim_boundaries=True,
    ):
        """Generate G1 transitions from (N,4,36) poses; default is MuJoCo WXYZ.

        Both inputs must already share the official G1 model space (Z up).
        The returned qpos uses the same quaternion order as the inputs.
        """
        if self.bundle.model != "g1":
            raise ValueError("G1 inbetweening requires the g1 model")
        if quaternion_order not in ("wxyz", "xyzw"):
            raise ValueError("quaternion_order must be wxyz or xyzw")
        inputs = {}
        for name, value in [
            ("context_qpos", context_qpos),
            ("target_qpos", target_qpos),
        ]:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            value = np.array(value, dtype=np.float32, copy=True)
            if value.ndim != 3 or value.shape[1:] != (4, 36):
                raise ValueError(f"{name} must have shape (N,4,36)")
            if quaternion_order == "wxyz":
                value[..., 3:7] = value[..., [4, 5, 6, 3]]
            inputs[name] = value
        batch = inputs["context_qpos"].shape[0]
        for name in (
            "has_global_root_values",
            "has_local_root_values",
            "has_local_poses",
        ):
            inputs[name] = np.ones((batch, 8), dtype=np.int32)
        inputs["allowed_pred_num_tokens"] = (
            np.ones((batch, 11), dtype=np.int32)
            if allowed_pred_num_tokens is None
            else allowed_pred_num_tokens
        )
        result = self.infer_pose_inbetweening(inputs, trim_boundaries=trim_boundaries)
        if quaternion_order == "wxyz":
            for motion in result["motions"]:
                motion[..., 3:7] = motion[..., [6, 3, 4, 5]]
            result["representation"] = "g1_qpos"
        result["qpos"] = result["motions"]
        return result

    def create_runtime(self, *, model_to_runtime=None, stream=None):
        """Create the full CUDA pose runtime: alignment, decode, filter and correction."""
        if self.bundle.model == "g1":
            from .runtime_unitree_g1_engine import MotionBricksUnitreeG1Engine

            engine = MotionBricksUnitreeG1Engine
        else:
            from .runtime_soma_engine import MotionBricksSOMAEngine

            engine = MotionBricksSOMAEngine
        return engine(self, model_to_runtime=model_to_runtime, stream=stream)

    def load_scene(self, scene_path, *, assets_root, reference_clip=None, seed=None):
        """Create a headless character/controller from an official scene document."""
        from .scene import MotionBricksScene

        return MotionBricksScene(
            self,
            scene_path,
            assets_root=assets_root,
            reference_clip=reference_clip,
            seed=seed,
        )

    def save_pretrained(self, save_directory, **kwargs):
        return self.bundle.save_pretrained(save_directory, **kwargs)
