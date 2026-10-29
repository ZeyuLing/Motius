"""Data-side codecs shared by training and inference, outside neural Bundles."""

from __future__ import annotations
from typing import Any, Dict
from pathlib import Path
import os.path as osp
import torch
from torch import Tensor


def _get_module_device(module):
    return next(module.parameters()).device


class _MotionCodec:
    def __init__(self, bundle):
        self.bundle = bundle
        self._body_model = None

    @property
    def mean(self):
        return self.bundle.mean

    @property
    def std(self):
        return self.bundle.std

    @property
    def motion_type(self):
        return self.bundle.motion_type

    @property
    def rotation_space(self):
        return self.bundle.rotation_space

    @property
    def _body_model_path(self):
        return self.bundle._body_model_path

    @property
    def _bone_offsets_override(self):
        return getattr(self.bundle, "_bone_offsets_override", None)

    @_bone_offsets_override.setter
    def _bone_offsets_override(self, value):
        self.bundle._bone_offsets_override = value


class MotionCanvasCodec(_MotionCodec):
    @property
    def body_model(self):
        """Lazy-load SmplxLiteJ24 body model."""
        if self._body_model is None:
            from motius.motion.representation.hymotion_runtime.smpl_lite import (
                SmplxLiteJ24,
            )

            kwargs = {}
            if self._body_model_path is not None:
                kwargs["model_path"] = self._body_model_path
            try:
                self._body_model = SmplxLiteJ24(**kwargs)
                self._body_model.to(_get_module_device(self.bundle))
                self._body_model.eval()
            except Exception:
                return None
        return self._body_model

    def decode_motion_from_latent(
        self,
        latent: Tensor,
    ) -> Dict[str, Tensor]:
        """Denormalize latent and run FK to get 3D keypoints.

        Returns dict with keys: keypoints3d, rot6d, transl, latent_denorm.

        When ``rotation_space == 'global'``, the denormalized rot6d is in
        world-frame global rotation.  We convert it back to local (SMPL)
        rotation before FK so that the output NPZ is always SMPL-compatible.
        """
        from motius.motion.representation.hymotion_runtime.geometry import (
            rot6d_to_rotation_matrix,
        )

        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean

        B, L = latent_denorm.shape[:2]
        transl = latent_denorm[..., 0:3].clone()

        # Extract rot6d: (B, L, 22, 6)
        rot6d_all = latent_denorm[..., 3:135].reshape(B, L, 22, 6).clone()

        # If trained in global rotation space, convert back to local for SMPL output
        if self.rotation_space == "global":
            from motius.motion.skeleton.fk import global_to_local_rot6d

            rot6d_all = global_to_local_rot6d(rot6d_all)

        root_rot6d = rot6d_all[:, :, 0:1, :]  # (B, L, 1, 6)
        body6d = rot6d_all[:, :, 1:, :]  # (B, L, 21, 6)
        rot6d = rot6d_all
        root_rotmat = rot6d_to_rotation_matrix(rot6d[:, :, 0, :])

        k3d = None
        if self.body_model is not None:
            try:
                device = latent.device
                betas = torch.zeros(1, 16, device=device)
                k3d_list = []
                for b in range(B):
                    out = self.body_model(
                        body6d[b].to(device),
                        betas,
                        root_rot6d[b].to(device),
                        transl[b].to(device),
                    )
                    k3d_list.append(out)
                k3d = torch.stack(k3d_list, dim=0)
            except Exception:
                k3d = None

        return {
            "latent_denorm": latent_denorm,
            "keypoints3d": k3d,
            "rot6d": rot6d,
            "transl": transl,
            "root_rotations_mat": root_rotmat,
        }

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers."""
        return (motion - self.mean) / self.std

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion."""
        std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        return motion * std + self.mean

    def set_bone_offsets_override(self, bone_offsets: Tensor) -> None:
        """Install an explicit, non-checkpointed FK-offset authority.

        ``SmplxLiteJ24.J_template`` and the precomputed SMPL-H training asset
        can differ by floating-point reconstruction even when they originate
        from the same rest pose.  Formal inference must not let optional body
        model availability choose which representation is fed to the network.
        """
        offsets = torch.as_tensor(bone_offsets).detach().clone().float()
        if tuple(offsets.shape) != (22, 3):
            raise ValueError(
                "bone-offset override must have shape (22,3), got "
                f"{tuple(offsets.shape)}"
            )
        if not torch.isfinite(offsets).all():
            raise ValueError("bone-offset override contains non-finite values")
        first_parameter = next(self.bundle.parameters(), None)
        target_device = (
            first_parameter.device if first_parameter is not None else offsets.device
        )
        self._bone_offsets_override = offsets.to(target_device)

    def get_bone_offsets(self) -> Tensor:
        """Get bone offsets for FK/IK.

        Attempts to compute from body model first; falls back to pre-computed
        file at ``data/hymotion_m2m_data/bone_offsets_22.pt``.

        Returns:
            bone_offsets: (22, 3) tensor of bone offsets.
        """
        from motius.motion.skeleton.names import SMPL22_PARENTS

        # Formal inference can pin the training-representation authority.  The
        # override must win regardless of whether the optional body model is
        # present on a particular runtime host.
        if self._bone_offsets_override is not None:
            return self._bone_offsets_override

        # Try computing from body model
        if self.body_model is not None:
            try:
                J_template = self.body_model.J_template[:22].clone()
                offsets = torch.zeros(
                    22, 3, device=J_template.device, dtype=J_template.dtype
                )
                offsets[0] = J_template[0]
                for j in range(1, 22):
                    parent = SMPL22_PARENTS[j]
                    offsets[j] = J_template[j] - J_template[parent]
                return offsets
            except Exception:
                pass

        # Fallback: load pre-computed file
        fallback_path = osp.join(
            str(Path(__file__).resolve().parents[4]),
            "data",
            "hymotion_m2m_data",
            "bone_offsets_22.pt",
        )
        if osp.isfile(fallback_path):
            offsets = torch.load(fallback_path, map_location="cpu")
            return offsets.to(_get_module_device(self.bundle))

        raise RuntimeError(
            "Cannot compute bone offsets: body model unavailable and "
            f"fallback file not found at {fallback_path}. "
            "Run `python tools/precompute_bone_offsets.py` first."
        )


class HyMotionCodec(_MotionCodec):
    def __init__(self, bundle, *, body_backend="smpl"):
        super().__init__(bundle)
        self.body_backend = body_backend

    @property
    def body_model(self):
        """Lazy-load SmplxLiteJ24 body model."""
        if self._body_model is None:
            if self.body_backend == "wooden":
                from motius.motion.body_models.hymotion_wooden import HYMotionWoodenBody

                self._body_model = HYMotionWoodenBody().to(
                    _get_module_device(self.bundle)
                )
                return self._body_model
            from motius.motion.representation.hymotion_runtime.smpl_lite import (
                SmplxLiteJ24,
            )

            kwargs = {}
            if self._body_model_path is not None:
                kwargs["model_path"] = self._body_model_path
            try:
                self._body_model = SmplxLiteJ24(**kwargs)
                self._body_model.to(_get_module_device(self.bundle))
                self._body_model.eval()
            except Exception:
                return None
        return self._body_model

    def decode_motion_from_latent(
        self,
        latent: Tensor,
        should_apply_smoothing: bool = True,
    ) -> Dict[str, Any]:
        """Denormalize and decode the artifact's native representation.

        When ``should_apply_smoothing`` (the official HY-Motion-1.0 inference
        default) the raw flow-matching output is temporally filtered before FK:
        rot6d gets quaternion-Gaussian SLERP smoothing (sigma=1.0) and the root
        translation gets Savitzky-Golay smoothing (window=11, polyorder=5),
        matching ``MotionFlowMatching.decode_motion_from_latent`` /
        ``_decode_o6dp``. Skipping this (the previous behaviour) leaves severe
        high-frequency jitter in the generated motion.

        SMPL artifacts return body rotations, translation, and optional joints.
        G1 artifacts return exact ``g1_38`` and MuJoCo ``g1_qpos`` tensors.
        """
        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        latent_denorm = latent * std + self.mean

        if self.motion_type in {"g1_29dof", "g1_38", "unitree_g1"}:
            from motius.motion.representation.g1 import decode_g1_to_qpos
            from motius.motion.representation.rotation import rotation_6d_to_matrix

            if latent_denorm.shape[-1] != 38:
                raise ValueError(
                    "G1 decoding requires a 38-dimensional latent, got "
                    f"{latent_denorm.shape[-1]}."
                )
            qpos = decode_g1_to_qpos(latent_denorm, root_velocity=True)
            root_rot6d = latent_denorm[..., 3:9]
            root_rotmat = rotation_6d_to_matrix(root_rot6d, convention="row")
            return {
                "motion": latent_denorm,
                "latent_denorm": latent_denorm,
                "g1_38": latent_denorm,
                "g1_qpos": qpos,
                "qpos": qpos,
                "keypoints3d": None,
                "rot6d": root_rot6d.unsqueeze(-2),
                "transl": qpos[..., :3],
                "root_rotations_mat": root_rotmat,
                "representation": "g1_38",
                "fps": 30.0,
            }

        from motius.motion.representation.hymotion_runtime.geometry import (
            rot6d_to_rotation_matrix,
        )

        B, L = latent_denorm.shape[:2]
        transl = latent_denorm[..., 0:3].clone()
        root_rot6d = latent_denorm[..., 3:9].reshape(B, L, 1, 6).clone()
        body6d = latent_denorm[..., 9:135].reshape(B, L, 21, 6).clone()
        rot6d = torch.cat([root_rot6d, body6d], dim=2)

        if should_apply_smoothing:
            from motius.motion.representation.hymotion_runtime.smoothing import (
                smooth_with_savgol,
                smooth_with_slerp,
            )

            # SLERP smoothing on the 22 body joints' rot6d (sigma=1.0); transl
            # Savitzky-Golay (window=11, polyorder=5) — official inference path.
            rot6d = smooth_with_slerp(rot6d, sigma=1.0)
            transl = smooth_with_savgol(transl, window_length=11, polyorder=5)
            root_rot6d = rot6d[:, :, 0:1, :]
            body6d = rot6d[:, :, 1:, :]

        root_rotmat = rot6d_to_rotation_matrix(rot6d[:, :, 0, :])

        k3d = None
        if self.body_backend == "wooden":
            vertices, joints = zip(
                *[self.body_model(rot6d[index], transl[index]) for index in range(B)]
            )
            min_y = torch.stack(vertices)[..., 1].amin(dim=(1, 2), keepdim=True)
            k3d = torch.stack(joints).clone()
            k3d[..., 1] -= min_y
            transl = transl.clone()
            transl[..., 1] -= min_y.squeeze(-1)
            return {
                "motion": latent_denorm,
                "latent_denorm": latent_denorm,
                "keypoints3d": k3d,
                "rot6d": rot6d,
                "transl": transl,
                "root_rotations_mat": root_rotmat,
                "representation": "hy_motion_201",
                "fps": 30.0,
            }
        if self.body_model is not None:
            try:
                device = latent.device
                betas = torch.zeros(1, 16, device=device)
                k3d_list = []
                for b in range(B):
                    out = self.body_model(
                        body6d[b].to(device),
                        betas,
                        root_rot6d[b].to(device),
                        transl[b].to(device),
                    )
                    k3d_list.append(out)
                k3d = torch.stack(k3d_list, dim=0)
            except Exception:
                k3d = None

        # Ground alignment: offset translation so lowest joint touches Y=0.
        # Matches official HY-Motion-1.0 post-FK processing.
        if k3d is not None:
            # min Y across all joints and frames per batch sample
            min_y = k3d[:, :, :, 1].min(dim=2)[0].min(dim=1)[0]  # (B,)
            transl[:, :, 1] = transl[:, :, 1] - min_y.unsqueeze(1)
            k3d[:, :, :, 1] = k3d[:, :, :, 1] - min_y.unsqueeze(1).unsqueeze(1)

        return {
            "motion": latent_denorm,
            "latent_denorm": latent_denorm,
            "keypoints3d": k3d,
            "rot6d": rot6d,
            "transl": transl,
            "root_rotations_mat": root_rotmat,
            "representation": "hy_motion_201",
            "fps": 30.0,
        }

    def normalize_motion(self, motion: Tensor) -> Tensor:
        """Normalize motion using mean/std buffers.

        Dims with near-zero std (constant in training data) produce 0 after normalization.
        This matches official HY-Motion-1.0 behavior.
        """
        # Safe division: where std==0, output 0 (those dims are constant)
        safe_std = torch.where(self.std < 1e-3, torch.ones_like(self.std), self.std)
        result = (motion - self.mean) / safe_std
        # Zero out dims where std was near-zero
        result = torch.where(
            self.std.unsqueeze(0) < 1e-3, torch.zeros_like(result), result
        )
        return result

    def denormalize_motion(self, motion: Tensor) -> Tensor:
        """Denormalize motion (matching official HY-Motion-1.0: zeros for near-zero std)."""
        std = torch.where(self.std < 1e-3, torch.zeros_like(self.std), self.std)
        return motion * std + self.mean
