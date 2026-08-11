# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from soma import SOMALayer as _SOMALayer
from soma import Unit


class NonPersistentModuleWrapper(nn.Module):
    """Wrap a module but drop all of its state_dict entries."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        return

    def load_state_dict(self, state_dict, strict=True):
        return torch.nn.modules.module._IncompatibleKeys([], [])


class SomaLayer(nn.Module):
    """Wraps soma.SOMALayer (third_party) with the same interface as
    body_models.soma_hybrid.SomaLayer, exposing:
      - forward(global_orient, body_pose, identity_coeffs, scale_params, transl, pose2rot)
      - static_forward(poses, identity_coeffs, scale_params, transl, pose2rot, return_joints_only)
      - temporal_forward(poses, identity_coeffs, scale_params, transl, pose2rot, return_joints_only)
      - get_skeleton(identity_coeffs, scale_params)
    """

    def __init__(
        self, data_root, low_lod=False, device="cuda", identity_model_type="mhr", mode="warp"
    ):
        super().__init__()
        self.soma = _SOMALayer(
            data_root=data_root,
            low_lod=low_lod,
            device=device,
            identity_model_type=identity_model_type,
            mode=mode,
            output_unit=Unit.METERS,
        )
        self.identity_model_type = identity_model_type
        self.device = device
        self.faces = self.soma.faces
        self.parents = self.soma.parents

    def get_skeleton(self, identity_coeffs, scale_params=None):
        """Return joint positions (77 joints) for zero pose.

        Args:
            identity_coeffs: (B, C) or (B, L, C)
            scale_params: (B, S) or (B, L, S), optional
        Returns:
            joints: (B, 77, 3) or (B, L, 77, 3)
        """
        if identity_coeffs.ndim == 3:
            B, L = identity_coeffs.shape[:2]
            zero_poses = torch.zeros(
                (B, L, 77, 3), device=identity_coeffs.device, dtype=torch.float32
            )
            zero_transl = torch.zeros((B, L, 3), device=identity_coeffs.device, dtype=torch.float32)
            output = self.temporal_forward(
                zero_poses, identity_coeffs, scale_params, zero_transl, return_joints_only=True
            )
            return output["joints"].reshape(B, L, 77, 3)
        else:
            B = identity_coeffs.shape[0]
            zero_poses = torch.zeros((B, 77, 3), device=identity_coeffs.device, dtype=torch.float32)
            zero_transl = torch.zeros((B, 3), device=identity_coeffs.device, dtype=torch.float32)
            output = self.static_forward(
                zero_poses, identity_coeffs, scale_params, zero_transl, return_joints_only=True
            )
            return output["joints"].reshape(B, 77, 3)

    def forward(
        self,
        global_orient,
        body_pose,
        identity_coeffs,
        scale_params=None,
        transl=None,
        pose2rot=True,
    ):
        """
        Args:
            global_orient: (B, 3) or (B, L, 3)
            body_pose: (B, 76*3) or (B, 76, 3) or (B, L, 76*3) or (B, L, 76, 3)
            identity_coeffs: (B, C) or (B, L, C)
            scale_params: (B, S) or (B, L, S), optional
            transl: (B, 3) or (B, L, 3), optional
            pose2rot: convert axis-angle to rotation matrices
        """
        if global_orient.ndim == 3 and pose2rot:
            B, L = global_orient.shape[:2]
            body_pose = body_pose.reshape(B, L, 76, 3)
            poses = torch.cat([global_orient[:, :, None], body_pose], dim=2)  # (B, L, 77, 3)
            return self.temporal_forward(
                poses, identity_coeffs, scale_params, transl, pose2rot=pose2rot
            )
        else:
            if body_pose.shape[-1] == 76 * 3:
                assert body_pose.ndim == 2, body_pose.shape
                body_pose = body_pose.reshape(body_pose.shape[0], 76, 3)
            poses = torch.cat([global_orient[:, None], body_pose], dim=1)  # (B, 77, 3)
            return self.static_forward(
                poses, identity_coeffs, scale_params, transl, pose2rot=pose2rot
            )

    @torch.amp.autocast("cuda", enabled=False)
    def static_forward(
        self,
        poses,
        identity_coeffs,
        scale_params=None,
        transl=None,
        pose2rot=True,
        return_joints_only=False,
    ):
        """
        Args:
            poses: (B, 77, 3) axis-angle or (B, 77, 3, 3) rotation matrices
            identity_coeffs: (B, C)
            scale_params: (B, S), optional
            transl: (B, 3), optional
            pose2rot: convert axis-angle to rotation matrices
            return_joints_only: skip vertex computation
        Returns:
            dict with 'vertices' (B, V, 3) and 'joints' (B, 77, 3)
        """
        global_scale = scale_params[:, :1]
        scale_params = scale_params[:, 1:]
        self.soma.prepare_identity(
            identity_coeffs, scale_params, repose_to_bind_pose=False, global_scale=global_scale
        )
        out = self.soma.pose(poses, transl=transl, pose2rot=pose2rot)
        return {
            "vertices": out["vertices"] if not return_joints_only else None,
            "joints": out["joints"],
        }

    @torch.amp.autocast("cuda", enabled=False)
    def temporal_forward(
        self,
        poses,
        identity_coeffs,
        scale_params=None,
        transl=None,
        pose2rot=True,
        return_joints_only=False,
    ):
        """
        Args:
            poses: (B, L, 77, 3) axis-angle
            identity_coeffs: (B, L, C)
            scale_params: (B, L, S), optional
            transl: (B, L, 3), optional
            pose2rot: convert axis-angle to rotation matrices
            return_joints_only: skip vertex computation
        Returns:
            dict with 'vertices' (B, L, V, 3) and 'joints' (B, L, 77, 3)
        """
        B, L = poses.shape[:2]

        avg_identity_coeffs = identity_coeffs.mean(dim=1)  # (B, C)
        if scale_params is not None:
            avg_scale_params = scale_params.mean(dim=1)
        else:
            avg_scale_params = None
        avg_global_scale = avg_scale_params[:, :1]
        avg_scale_params = avg_scale_params[:, 1:]

        self.soma.prepare_identity(
            avg_identity_coeffs,
            avg_scale_params,
            repose_to_bind_pose=False,
            global_scale=avg_global_scale,
        )  # caches (B, V, 3), (B, J, 4, 4)
        _cached_rest_shape = self.soma._cached_rest_shape.unsqueeze(1).repeat(1, L, 1, 1)
        _cached_rest_shape = _cached_rest_shape.reshape(B * L, -1, 3)
        _cached_bind_transforms_world = self.soma._cached_bind_transforms_world.unsqueeze(1).repeat(
            1, L, 1, 1, 1
        )
        _cached_bind_transforms_world = _cached_bind_transforms_world.reshape(B * L, -1, 4, 4)
        self.soma._cached_rest_shape = _cached_rest_shape
        self.soma._cached_bind_transforms_world = _cached_bind_transforms_world

        # 3. Animate SOMA
        poses = poses.reshape(B * L, -1, 3)
        transl = transl.reshape(B * L, 3)
        out = self.soma.pose(poses, transl=transl, pose2rot=pose2rot, apply_correctives=False)

        return {
            "vertices": out["vertices"].reshape(B, L, -1, 3),
            "joints": out["joints"].reshape(B, L, -1, 3),
        }
