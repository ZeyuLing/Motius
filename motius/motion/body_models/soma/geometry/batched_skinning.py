# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from ._utils import mask_1d, one_hot_1d, one_hot_2d, require_torch_tensors
from .lbs import lbs
from .rig_utils import (
    apply_joint_orient_local,
    compute_skeleton_levels,
    joint_local_to_world_levelorder,
    joint_world_to_local,
    precompute_joint_orient,
)
from .transforms import SE3_from_Rt


def topk_skinning(
    W,
    K=8,
    weight_eps=1e-12,
    sort_desc=True,
    pad_index=-1,
    dtype_idx=torch.int32,
    dtype_w=torch.float32,
):
    """
    Convert dense skinning weights (N, J) -> sparse top-K jointIndices/jointWeights.

    Args:
      W: (N, J) float tensor of skinning weights per vertex.
      K: number of influences per vertex to keep.
      weight_eps: prune tiny weights; also avoids div-by-zero in normalization.
      sort_desc: sort the chosen K influences by descending weight (stable output).
      pad_index: index used when fewer than K nonzero weights exist for a vertex.
      dtype_idx, dtype_w: dtypes for outputs.

    Returns:
      idx_mat: (N, K) int32
      w_mat:   (N, K) float32
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError("W must be a torch.Tensor.")
    N, J = W.shape
    K_eff = min(K, J)

    W_masked = torch.where(W > weight_eps, W, torch.zeros_like(W))

    w_topk, idx_topk = torch.topk(W_masked, K_eff, dim=1, largest=True, sorted=sort_desc)

    if K_eff < K:
        pad_cols = K - K_eff
        idx_pad = torch.full((N, pad_cols), pad_index, device=W.device, dtype=idx_topk.dtype)
        w_pad = torch.zeros((N, pad_cols), device=W.device, dtype=w_topk.dtype)
        idx_mat = torch.cat([idx_topk, idx_pad], dim=1)
        w_mat = torch.cat([w_topk, w_pad], dim=1)
    else:
        idx_mat, w_mat = idx_topk, w_topk

    s = w_mat.sum(dim=1, keepdim=True)
    nonzero = s > 0
    w_mat = torch.where(nonzero, w_mat / torch.clamp(s, min=1e-20), torch.zeros_like(w_mat))

    idx_mat = idx_mat.to(dtype_idx)
    w_mat = w_mat.to(dtype_w)

    return idx_mat, w_mat


class BatchedSkinning:
    def __init__(
        self,
        joint_parent_ids,
        skinning_weights,
        bind_world_transforms,
        bind_shapes,
        joint_orient=None,
        mode="warp",
    ):
        """Initialize a BatchedSkinning instance for posing meshes using Linear Blend Skinning (LBS).
        Args:
            joint_parent_ids: (J,) int array of joint parent indices across all characters
            skinning_weights: (V, J) array of skinning weights across all characters
            bind_world_transforms: (B, J, 4, 4) array of joint bind poses in world space for multiple characters
            bind_shapes: (B, V, 3) array of vertex positions in bind pose for multiple characters
            joint_orient: None or (J, M, M) where M > 3 array specifying the initial world space orientation of each joint. Poses are relative to this orientation if given.
            mode: "warp" for sparse warp-based LBS (default) or "dense" for dense LBS
        """
        self.dtype, self.device = require_torch_tensors(
            skinning_weights, bind_world_transforms, bind_shapes, name="BatchedSkinning inputs"
        )
        self.mode = mode

        batched = bind_shapes.ndim == 3
        num_joints = len(joint_parent_ids)
        if num_joints != bind_world_transforms.shape[1 if batched else 0]:
            raise ValueError(
                "joint_parent_ids and bind_world_transforms must have the same number of joints."
            )
        if batched and bind_world_transforms.shape[0] != bind_shapes.shape[0]:
            raise ValueError("bind_world_transforms and bind_shapes must have the same batch size.")
        self.bind_batched = batched
        self.num_joints = len(joint_parent_ids)
        self.joint_parent_ids = (
            joint_parent_ids
            if isinstance(joint_parent_ids, list)
            else joint_parent_ids.tolist()
            if hasattr(joint_parent_ids, "tolist")
            else list(joint_parent_ids)
        )
        self.skinning_weights = skinning_weights
        self.bind_world_transforms = bind_world_transforms
        bind_local_transforms, self.inverse_bind_transform = joint_world_to_local(
            bind_world_transforms, joint_parent_ids, return_inverse=True
        )
        self.local_translations = bind_local_transforms[..., :3, 3]
        self.bind_shapes = bind_shapes
        self.joint_orient = None
        self._orient_parent_T = None
        if joint_orient is not None:
            if num_joints != joint_orient.shape[0]:
                raise ValueError(
                    "joint_orient must have the same number of joints as joint_parent_ids."
                )
            jo = joint_orient.to(dtype=self.dtype, device=self.device)
            self.joint_orient, self._orient_parent_T = precompute_joint_orient(
                jo, self.joint_parent_ids
            )

        self._levels = compute_skeleton_levels(self.joint_parent_ids, device=self.device)

        self._bone_weights = None
        self._bone_indices = None
        if self.mode == "warp":
            self._prepare_warp_data()

    def rebind(self, bind_world_transforms, bind_shapes):
        """Rebind the skeleton to new bind poses and shapes
        Args:
            bind_world_transforms: (B, J, 4, 4) array of new joint bind poses in world space for multiple characters
            bind_shapes: (B, V, 3) array of new vertex positions in bind pose for multiple characters
        """
        batched = bind_shapes.ndim == 3
        self.bind_batched = batched
        self.bind_world_transforms = bind_world_transforms
        bind_local_transforms, self.inverse_bind_transform = joint_world_to_local(
            bind_world_transforms, self.joint_parent_ids, return_inverse=True
        )
        self.local_translations = bind_local_transforms[..., :3, 3]
        self.bind_shapes = bind_shapes

    def _prepare_warp_data(self):
        """Prepare sparse bone weights and indices for warp-based LBS."""
        bone_indices, bone_weights = topk_skinning(self.skinning_weights)
        self._bone_indices = bone_indices.to(device=self.device)
        self._bone_weights = bone_weights.to(dtype=self.dtype, device=self.device)

    def get_bone_weights(self):
        """Get bone weights. For warp mode, returns sparse weights (V, K). For dense mode, returns full weights (V, J)."""
        if self.mode == "warp":
            if self._bone_weights is None:
                self._prepare_warp_data()
            return self._bone_weights
        else:
            return self.skinning_weights

    def get_bone_indices(self):
        """Get bone indices. For warp mode, returns sparse indices (V, K). For dense mode, returns None."""
        if self.mode == "warp":
            if self._bone_indices is None:
                self._prepare_warp_data()
            return self._bone_indices
        else:
            return None

    def pose(
        self,
        local_rotations,
        hips_translations,
        align_translation=None,
        return_transforms=False,
        absolute_pose=False,
    ):
        """
        Pose the meshes using Linear Blend Skinning (LBS), autograd-safe (no in-place).

        Supports:
        - many characters x one pose
        - one character x many poses
        - N characters x N poses (i-th character with i-th pose)
        - one character x one pose

        Args:
            local_rotations: (J,3,3) or (B,J,3,3)
            hips_translations: (3,) or (B,3)
            align_translation: None or (3,)
            return_transforms: bool, whether to return world transforms
            absolute_pose: bool, whether local_rotations are absolute rotations (True) or relative to joint orient (False, default)
        Returns:
            posed_shapes: (..., V, 3)
            (optional) world_transforms: (..., J, 4, 4)
        """
        if local_rotations.shape[-3:] != (self.num_joints, 3, 3):
            raise ValueError(
                f"Expected local_rotations to have shape (...,{self.num_joints},3,3); got {local_rotations.shape}"
            )

        if local_rotations.ndim == 3:
            local_rotations = local_rotations[None, :, :, :]

        rot_batch = local_rotations.shape[0]
        bind_batch = self.bind_world_transforms.shape[0] if self.bind_batched else 1

        if hips_translations.shape not in [(3,), (rot_batch, 3)]:
            raise ValueError(
                f"Expected hips_translations to have shape (3,) or ({rot_batch},3); got {hips_translations.shape}"
            )

        if rot_batch == 1 and bind_batch > 1:
            batch_size = bind_batch
            local_rotations = local_rotations.to(dtype=self.dtype, device=self.device).expand(
                batch_size, self.num_joints, 3, 3
            )
        elif rot_batch >= 1 and bind_batch == 1:
            batch_size = rot_batch
            local_rotations = local_rotations.to(dtype=self.dtype, device=self.device)
        elif rot_batch == bind_batch:
            batch_size = rot_batch
            local_rotations = local_rotations.to(dtype=self.dtype, device=self.device)
        else:
            raise ValueError(
                f"Incompatible batches: rotations={rot_batch}, bind={bind_batch}. "
                "Provide (1xB), (Bx1), (1x1), or (BxB) with equal B."
            )

        if align_translation is not None:
            align_translation = align_translation.to(dtype=self.dtype, device=self.device)

        local_t = self.local_translations.to(dtype=self.dtype, device=self.device)
        inverse_bind_transform = self.inverse_bind_transform.to(
            dtype=self.dtype, device=self.device
        )
        if self.bind_batched:
            if rot_batch == 1 and bind_batch > 1:
                pass
            elif rot_batch >= 1 and bind_batch == 1:
                local_t = local_t.expand(batch_size, self.num_joints, 3)
                inverse_bind_transform = inverse_bind_transform.expand(
                    batch_size, self.num_joints, 4, 4
                )
        else:
            local_t = local_t.unsqueeze(0).expand(batch_size, self.num_joints, 3)
            inverse_bind_transform = inverse_bind_transform.unsqueeze(0).expand(
                batch_size, self.num_joints, 4, 4
            )

        j_mask = one_hot_1d(self.num_joints, 1, dtype=self.dtype, device=self.device)[None, :, None]
        if align_translation is not None:
            comp_m = mask_1d(3, [0, 2], dtype=self.dtype, device=self.device)[None, None, :]
            M = j_mask * comp_m
            local_t = local_t * (1 - M) + align_translation[None, None, :] * M
        else:
            comp_m = mask_1d(3, [0, 1, 2], dtype=self.dtype, device=self.device)[None, None, :]
            M = j_mask * comp_m
            hips_t = hips_translations.to(dtype=self.dtype, device=self.device)
            if hips_t.ndim == 1:
                hips_t = hips_t[None, :]
            local_t = local_t * (1 - j_mask) + hips_t[:, None, :] * j_mask

        if self.joint_orient is not None and not absolute_pose:
            local_rotations = apply_joint_orient_local(
                local_rotations, self.joint_orient, self._orient_parent_T
            )
        T_local = SE3_from_Rt(local_rotations, local_t)

        T_world = joint_local_to_world_levelorder(T_local, self._levels)

        if align_translation is not None:
            y_world = T_world[..., 1, 3]
            y_offset = y_world.min(dim=1, keepdim=True).values
            shift = y_offset + align_translation[1]

            new_y_world = y_world - shift
            delta_w = (new_y_world - y_world)[..., None, None]
            E13 = one_hot_2d(4, 4, 1, 3, dtype=self.dtype, device=self.device)[None, None, ...]
            T_world = T_world + delta_w * E13

            y_local = T_local[..., 1, 3]
            new_y_local = y_local - shift
            delta_l = (new_y_local - y_local)[..., None, None]
            j_only = one_hot_1d(self.num_joints, 1, dtype=self.dtype, device=self.device)[
                None, :, None, None
            ]
            T_local = T_local + delta_l * E13 * j_only

        bone_transforms = T_world @ inverse_bind_transform
        if self.mode == "warp":
            bind_verts = (
                self.bind_shapes
                if self.bind_batched
                else self.bind_shapes.unsqueeze(0).expand(batch_size, -1, -1)
            )
            posed_shapes = self._warp_skinning(bind_verts, bone_transforms)
        else:
            posed_shapes = lbs(
                self.bind_shapes,
                self.skinning_weights,
                bone_transforms,
            )

        if return_transforms:
            return posed_shapes, T_world
        return posed_shapes

    def _warp_skinning(self, bind_verts, bone_transforms):
        """Warp-based skinning."""
        from .lbs_warp import linear_blend_skinning

        return linear_blend_skinning(
            bind_verts, self._bone_weights, self._bone_indices, bone_transforms
        )
