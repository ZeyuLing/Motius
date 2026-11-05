# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"Normalization and conversion between local and global motion features."

from __future__ import annotations

from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch


def quat_conjugate(a):
    """Return the conjugates of WXYZ quaternions."""
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((a[:, 0:1], -a[:, 1:]), dim=-1).view(shape)


def quat_apply(a, b):
    """Rotate vectors by WXYZ quaternions."""
    shape = b.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 3)
    xyz = a[:, 1:].clone()
    t = xyz.cross(b, dim=-1) * 2
    return (b + a[:, 0:1].clone() * t + xyz.cross(t, dim=-1)).view(shape)


def diff_angles(angles, fps: float):
    """Return the angular velocity between neighboring angles, by the shortest turn."""
    cos = torch.cos(angles)
    sin = torch.sin(angles)
    cos_diff = cos[..., 1:] * cos[..., :-1] + sin[..., 1:] * sin[..., :-1]
    sin_diff = sin[..., 1:] * cos[..., :-1] - cos[..., 1:] * sin[..., :-1]
    angles_diff = fps * torch.arctan2(sin_diff, cos_diff)
    return angles_diff


class Stats(torch.nn.Module):
    """Mean and standard deviation used to normalize motion features."""

    def __init__(self, eps=1e-05, legacy=False):
        super().__init__()
        self.legacy = legacy
        self.eps = eps

    def register_from_tensors(self, mean: torch.Tensor, std: torch.Tensor):
        """Store the mean and standard deviation as buffers."""
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def normalize(self, data: torch.Tensor, index=None) -> torch.Tensor:
        """Normalize data, optionally for a subset of feature indices."""
        mean = self.mean.to(device=data.device, dtype=data.dtype)
        std = self.std.to(device=data.device, dtype=data.dtype)
        if index is not None:
            mean = mean[..., index]
            std = std[..., index]
        if self.legacy:
            return (data - mean) / torch.clip(std, 0.001)
        return (data - mean) / torch.sqrt(std**2 + self.eps)

    def unnormalize(self, data: torch.Tensor, index=None) -> torch.Tensor:
        """Undo normalize()."""
        mean = self.mean.to(device=data.device, dtype=data.dtype)
        std = self.std.to(device=data.device, dtype=data.dtype)
        if index is not None:
            mean = mean[..., index]
            std = std[..., index]
        if self.legacy:
            return data * torch.clip(std, 0.001) + mean
        return data * torch.sqrt(std**2 + self.eps) + mean

    def is_loaded(self):
        """Return whether the statistics are set."""
        return hasattr(self, "mean")

    def get_dim(self):
        """Return the number of features the statistics cover."""
        return self.mean.shape[0]


class MotionRepBase(torch.nn.Module):
    """Base for motion feature layouts, with their sample rate, skeleton, and statistics."""

    def __init__(
        self,
        fps: float,
        skeleton: SimpleNamespace,
        name: str,
        stats: Optional[Stats] = None,
    ):
        super().__init__()
        self.stats = stats
        self.fps = fps
        self.skeleton = skeleton
        self.name = name
        self.num_joints = self.nbjoints = skeleton.nbjoints

    def normalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Normalize motion features with the layout's statistics."""
        return self.stats.normalize(motion, index=index)

    def unnormalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Undo normalize()."""
        return self.stats.unnormalize(motion, index=index)


class SeparatedRootLocalBody(MotionRepBase):
    """Motion features split into a root part and a body part."""

    def __init__(
        self,
        fps: float,
        skeleton: SimpleNamespace,
        name: str,
        stats: Optional[Stats] = None,
    ):
        super().__init__(fps, skeleton, name, stats)
        self.name = name
        [*self.root_keys], [*self.root_ps] = zip(*self.root_keys_dim.items())
        [*self.body_keys], [*self.body_ps] = zip(*self.body_keys_dim.items())
        self.keys_dim = self.root_keys_dim | self.body_keys_dim
        self.keys = self.root_keys + self.body_keys
        self.ps = self.root_ps + self.body_ps
        self.motion_rep_dim = sum((sum(x) if x else 1 for x in self.ps))
        self.motion_root_dim = sum((sum(x) if x else 1 for x in self.root_ps))
        self.motion_body_dim = sum((sum(x) if x else 1 for x in self.body_ps))
        if stats is not None and stats.is_loaded():
            assert self.motion_rep_dim == stats.get_dim()
        self.motion_rep_subset_dim = self.motion_rep_dim
        self.indices = {}
        idx = 0
        for key, x in zip(self.keys, self.ps):
            dim = sum(x) if x else 1
            self.indices[key] = np.arange(idx, idx + dim)
            idx += dim
        self.indices["all"] = np.arange(0, self.motion_rep_dim)
        self.indices["root"] = np.arange(0, self.motion_root_dim)
        self.indices["body"] = np.arange(
            self.motion_root_dim, self.motion_root_dim + self.motion_body_dim
        )
        self.removed_heading = self.compute_kwargs.get("removing_heading", True)
        self.using_smooth_root = self.compute_kwargs.get("using_smooth_root", False)
        self.extra_skel = self.compute_kwargs.get("extra_skel", False)
        if self.extra_skel:
            assert repr(self.skeleton.__class__).split("'")[1].endswith("Extra"), (
                "The skeleton should be an extra skeleton"
            )

    def slice(self, motion, key: str):
        """Return the features stored under key."""
        assert key in self.indices
        assert motion.shape[-1] == self.motion_rep_dim
        return motion[..., self.indices[key]]

    def detect_all_body_or_root(self, motion: torch.Tensor):
        """Return which part motion holds: "all", "root", or "body".

        The width of the last axis decides.
        """
        dim = motion.shape[-1]
        if dim == self.motion_rep_dim:
            return "all"
        elif dim == self.motion_root_dim:
            return "root"
        elif dim == self.motion_body_dim:
            return "body"
        else:
            raise ValueError(f"Unrecognized input dim: {dim}.")

    def normalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Normalize motion features.

        Without index, the part they cover decides which statistics apply.
        """
        if index is None:
            index = self.indices[self.detect_all_body_or_root(motion)]
        return self.stats.normalize(motion, index=index)

    def unnormalize(self, motion: torch.Tensor, index=None) -> torch.Tensor:
        """Undo normalize()."""
        if index is None:
            index = self.indices[self.detect_all_body_or_root(motion)]
        return self.stats.unnormalize(motion, index=index)

    def extract_root(self, motion: torch.Tensor) -> torch.Tensor:
        """Return the root features of motion."""
        type = self.detect_all_body_or_root(motion)
        if type == "root":
            return motion
        elif type == "all":
            return self.slice(motion, "root")
        else:
            raise ValueError("Cannot compute root info without root data.")

    def extract_body(self, motion: torch.Tensor) -> torch.Tensor:
        """Return the body features of motion."""
        type = self.detect_all_body_or_root(motion)
        if type == "body":
            return motion
        elif type == "all":
            return self.slice(motion, "body")
        else:
            raise ValueError("Cannot compute body info without body data.")

    def concat_root_body(self, root_data: torch.Tensor, body_data: torch.Tensor):
        """Join root and body features into full motion features."""
        assert root_data.shape[-1] == self.motion_root_dim
        assert body_data.shape[-1] == self.motion_body_dim
        motion_data = torch.cat([root_data, body_data], axis=-1)
        assert motion_data.shape[-1] == self.motion_rep_dim
        return motion_data


class GlobalRootLocalBody(SeparatedRootLocalBody):
    """Motion features with the root position and heading given in model space."""

    dual_class = None

    def __init__(
        self,
        fps: float,
        skeleton: SimpleNamespace,
        name: str,
        stats: Optional[Stats] = None,
    ):
        self.root_keys_dim = {"global_root_pos": [3], "global_root_heading": [2]}
        self.body_keys_dim = self.get_body_keys_dim(skeleton.nbjoints)
        if self.dual_class is not None and self.dual_class._name_ in name:
            super().__init__(fps, skeleton, name, stats=None)
            self.dual_rep = self.dual_class(fps, skeleton, name, stats=stats)
            self.dual_rep_mode = "global"
            self.stats = self.dual_rep.global_motion_rep.stats
        else:
            self.dual_rep = None
            super().__init__(fps, skeleton, name, stats)
        self.indices["global_root_pos_2d"] = self.indices["global_root_pos"][[0, 2]]
        self.root_mode = "global"

    def compute_root_pos_and_rot(
        self, motion: torch.Tensor, return_quat: bool = True, return_angle: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the root positions and rotations, as quaternions or angles."""
        root_motion = self.extract_root(motion)
        r_pos = root_motion[..., :3]
        rot_cos = root_motion[..., 3]
        rot_sin = root_motion[..., 4]
        r_rot_ang = torch.atan2(rot_sin, rot_cos)
        r_rot_quat = torch.stack(
            [
                torch.cos(r_rot_ang / 2),
                torch.zeros_like(rot_cos),
                torch.sin(r_rot_ang / 2),
                torch.zeros_like(rot_cos),
            ],
            dim=-1,
        )
        if return_quat and (not return_angle):
            return (r_pos, r_rot_quat)
        if not return_quat and return_angle:
            return (r_pos, r_rot_ang)
        if return_quat and return_angle:
            return (r_pos, r_rot_quat, r_rot_ang)
        if not return_quat and (not return_angle):
            return r_pos

    def compute_root_rep_from_root_pos_and_rot(
        self,
        r_pos: torch.Tensor,
        r_rot_quat: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build root features from root positions and rotations."""
        root_rot_angles = torch.arctan2(r_rot_quat[..., 2], r_rot_quat[..., 0]) * 2
        root_motion = torch.cat(
            [
                r_pos,
                torch.cos(root_rot_angles)[..., None],
                torch.sin(root_rot_angles)[..., None],
            ],
            axis=-1,
        )
        return root_motion


class LocalRootLocalBody(SeparatedRootLocalBody):
    """Motion features with the root given as velocities and height."""

    dual_class = None

    def __init__(
        self,
        fps: float,
        skeleton: SimpleNamespace,
        name: str,
        stats: Optional[Stats] = None,
    ):
        self.nfeats_vel = 3 if self.compute_kwargs["local_root_vel_with_y"] else 2
        self.root_keys_dim = {
            "local_root_rot_vel": [],
            "local_root_vel": [self.nfeats_vel],
            "global_root_y": [],
        }
        self.body_keys_dim = self.get_body_keys_dim(skeleton.nbjoints)
        if self.dual_class is not None and self.dual_class._name_ in name:
            super().__init__(fps, skeleton, name, stats=None)
            self.dual_rep = self.dual_class(fps, skeleton, name, stats=stats)
            self.dual_rep_mode = "local"
            self.stats = self.dual_rep.local_motion_rep.stats
        else:
            self.dual_rep = None
            super().__init__(fps, skeleton, name, stats)
        self.root_mode = "local"

    def compute_root_pos_and_rot(
        self, motion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate root velocities into root positions and rotations."""
        root_motion = self.extract_root(motion)
        device = root_motion.device
        batch, nbframes = root_motion.shape[:2]
        root_rot_vel_angles = root_motion[..., 0]
        root_local_vel = root_motion[..., 1 : 1 + self.nfeats_vel]
        root_y_pos = root_motion[..., 1 + self.nfeats_vel]
        root_rot_vel_angles = root_rot_vel_angles / self.fps
        root_local_vel = root_local_vel / self.fps
        r_rot_ang = torch.zeros_like(root_rot_vel_angles, device=device)
        r_rot_ang[..., 1:] = root_rot_vel_angles[..., :-1]
        r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)
        r_rot_quat = torch.zeros((batch, nbframes, 4), device=device)
        r_rot_quat[..., 0] = torch.cos(r_rot_ang / 2)
        r_rot_quat[..., 2] = torch.sin(r_rot_ang / 2)
        r_pos = torch.zeros((batch, nbframes, 3), device=device)
        r_pos[..., 1:, [0, 2]] = root_local_vel[..., :-1, :]
        removed_heading = self.compute_kwargs.get("removing_heading", True)
        if removed_heading:
            r_pos[..., 1:, :] = quat_apply(r_rot_quat[..., :-1, :], r_pos[..., 1:, :])
        r_pos = torch.cumsum(r_pos, dim=-2)
        r_pos[..., 1] = root_y_pos
        return (r_pos, r_rot_quat)


class DualRootLocalBody(SeparatedRootLocalBody):
    """Pair of the global and local feature layouts, with conversion between them."""

    def __init__(
        self,
        fps: float,
        skeleton: SimpleNamespace,
        name: str,
        *,
        global_class: GlobalRootLocalBody,
        local_class: LocalRootLocalBody,
        stats: Optional[Stats] = None,
    ):
        global_motion_rep = global_class(fps=fps, skeleton=skeleton, name="global")
        local_motion_rep = local_class(fps=fps, skeleton=skeleton, name="local")
        assert (
            global_motion_rep.default_joint_positions_from
            == local_motion_rep.default_joint_positions_from
        )
        self.default_joint_positions_from = (
            global_motion_rep.default_joint_positions_from
        )
        self.root_keys_dim = (
            global_motion_rep.root_keys_dim | local_motion_rep.root_keys_dim
        )
        assert len(self.root_keys_dim) == len(global_motion_rep.root_keys_dim) + len(
            local_motion_rep.root_keys_dim
        )
        assert global_motion_rep.body_keys_dim == local_motion_rep.body_keys_dim
        self.body_keys_dim = global_motion_rep.body_keys_dim
        super().__init__(fps, skeleton, name, stats)
        self.global_motion_rep = global_motion_rep
        self.local_motion_rep = local_motion_rep
        self.indices["global_root"] = np.arange(0, global_motion_rep.motion_root_dim)
        self.indices["local_root"] = np.arange(
            global_motion_rep.motion_root_dim, self.motion_root_dim
        )
        self.indices["global_rep"] = np.concatenate(
            (self.indices["global_root"], self.indices["body"])
        )
        self.indices["local_rep"] = np.concatenate(
            (self.indices["local_root"], self.indices["body"])
        )
        assert global_motion_rep.compute_kwargs == local_motion_rep.compute_kwargs
        self.compute_kwargs = global_motion_rep.compute_kwargs
        self.motion_rep_subset_dim = self.global_motion_rep.motion_rep_dim
        self.dual_rep = None
        if stats is not None and stats.is_loaded():
            global_mean = stats.mean[self.indices["global_rep"]]
            global_std = stats.std[self.indices["global_rep"]]
            self.global_motion_rep.stats = Stats(eps=stats.eps, legacy=stats.legacy)
            self.global_motion_rep.stats.register_from_tensors(global_mean, global_std)
            local_mean = stats.mean[self.indices["local_rep"]]
            local_std = stats.std[self.indices["local_rep"]]
            self.local_motion_rep.stats = Stats(eps=stats.eps, legacy=stats.legacy)
            self.local_motion_rep.stats.register_from_tensors(local_mean, local_std)

    def _one_subset_to_the_other(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        mode: str,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if mode == "local_to_global":
            motion_rep_from = self.local_motion_rep
            motion_rep_to = self.global_motion_rep
        elif mode == "global_to_local":
            motion_rep_from = self.global_motion_rep
            motion_rep_to = self.local_motion_rep
        else:
            raise ValueError("Unrecognized mode.")
        from_rep, to_rep = mode.split("_to_")
        data_type = motion_rep_from.detect_all_body_or_root(features)
        if data_type == "all":
            body_motion = motion_rep_from.extract_body(features)
        else:
            assert data_type == "root"
        root_motion = motion_rep_from.extract_root(features)
        if is_normalized:
            root_motion = self.unnormalize(
                root_motion, index=self.indices[f"{from_rep}_root"]
            )
            if not to_normalize and data_type == "all":
                body_motion = self.unnormalize(body_motion, index=self.indices["body"])
        r_pos, r_rot_quat = motion_rep_from.compute_root_pos_and_rot(root_motion)
        new_root_motion = motion_rep_to.compute_root_rep_from_root_pos_and_rot(
            r_pos, r_rot_quat, lengths
        )
        if to_normalize:
            new_root_motion = self.normalize(
                new_root_motion, index=self.indices[f"{to_rep}_root"]
            )
            if not is_normalized and data_type == "all":
                body_motion = self.normalize(body_motion, index=self.indices["body"])
        if data_type == "root":
            assert new_root_motion.shape[-1] == motion_rep_to.motion_root_dim
            return new_root_motion
        new_features = motion_rep_to.concat_root_body(new_root_motion, body_motion)
        assert new_features.shape[-1] == motion_rep_to.motion_rep_dim
        return new_features

    def global_to_local(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert features from the global root layout to the local one."""
        return self._one_subset_to_the_other(
            features,
            is_normalized=is_normalized,
            to_normalize=to_normalize,
            mode="global_to_local",
            lengths=lengths,
        )

    def local_to_global(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Convert features from the local root layout to the global one."""
        return self._one_subset_to_the_other(
            features,
            is_normalized=is_normalized,
            to_normalize=to_normalize,
            mode="local_to_global",
            lengths=lengths,
        )


COMPUTE_KWARGS = {
    "local_vel_without_root": False,
    "local_root_vel_with_y": False,
    "compute_heading_method": "hips_pos",
    "removing_heading": False,
}
DEFAULT_JOINT_POSITIONS_FROM = "global_rot_data"


class DualRootGlobalJoints(DualRootLocalBody):
    """Dual layout whose body features hold joint positions and rotations in model space."""

    _name_ = "dual_root_global_joints"
    compute_kwargs = COMPUTE_KWARGS
    default_joint_positions_from = DEFAULT_JOINT_POSITIONS_FROM

    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            global_class=GlobalRootGlobalJoints,
            local_class=LocalRootGlobalJoints,
            **kwargs,
        )


def get_body_keys_dim(self, nbjoints: int):
    """Return the size of each body feature.

    The features are joint positions, 6D rotations, velocities, and foot contacts.
    """
    return {
        "ric_data": [(nbjoints - 1) * 3],
        "global_rot_data": [nbjoints * 6],
        "local_vel": [nbjoints * 3],
        "foot_contacts": [4],
    }


class GlobalRootGlobalJoints(GlobalRootLocalBody):
    """Global root layout with joint positions and rotations in model space."""

    dual_class = DualRootGlobalJoints
    get_body_keys_dim = get_body_keys_dim
    compute_kwargs = COMPUTE_KWARGS
    default_joint_positions_from = DEFAULT_JOINT_POSITIONS_FROM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class LocalRootGlobalJoints(LocalRootLocalBody):
    """Local root layout with joint positions and rotations in model space."""

    dual_class = DualRootGlobalJoints
    get_body_keys_dim = get_body_keys_dim
    compute_kwargs = COMPUTE_KWARGS
    default_joint_positions_from = DEFAULT_JOINT_POSITIONS_FROM

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def global_root_to_local_onnx_compatible(
    normalized_global_root: torch.Tensor, dual_motion_rep, lengths: torch.Tensor
) -> torch.Tensor:
    """Convert normalized global root features to local ones, with ONNX operations only.

    Velocities at each request's last frame repeat the frame before it.
    """
    global_motion_rep = dual_motion_rep.global_motion_rep
    local_motion_rep = dual_motion_rep.local_motion_rep
    if local_motion_rep.compute_kwargs.get("local_root_vel_with_y", False):
        raise NotImplementedError(
            "Dynamic ONNX root conversion does not support local Y velocity."
        )
    global_root = dual_motion_rep.unnormalize(
        normalized_global_root, index=dual_motion_rep.indices["global_root"]
    )
    root_positions, root_rotations = global_motion_rep.compute_root_pos_and_rot(
        global_root
    )
    root_angles = torch.atan2(root_rotations[..., 2], root_rotations[..., 0]) * 2
    angular_velocity = diff_angles(root_angles, local_motion_rep.fps)
    angular_velocity = torch.cat(
        [angular_velocity, torch.zeros_like(root_angles[:, :1])], dim=1
    )
    root_velocity = local_motion_rep.fps * (
        root_positions[:, 1:] - root_positions[:, :-1]
    )
    root_velocity = torch.cat(
        [root_velocity, torch.zeros_like(root_positions[:, :1])], dim=1
    )
    last_indices = lengths.reshape([-1, 1]).long() - 1
    angular_velocity = angular_velocity.scatter(
        1, last_indices, angular_velocity.gather(1, last_indices - 1)
    )
    root_last_indices = last_indices[..., None].expand_as(root_velocity[:, :1])
    root_velocity = root_velocity.scatter(
        1, root_last_indices, root_velocity.gather(1, root_last_indices - 1)
    )
    if local_motion_rep.compute_kwargs.get("removing_heading", True):
        root_velocity = quat_apply(quat_conjugate(root_rotations), root_velocity)
    local_root_velocity = root_velocity[..., [0, 2]]
    local_root = torch.cat(
        [angular_velocity[..., None], local_root_velocity, root_positions[..., [1]]],
        dim=-1,
    )
    return dual_motion_rep.normalize(
        local_root, index=dual_motion_rep.indices["local_root"]
    )
