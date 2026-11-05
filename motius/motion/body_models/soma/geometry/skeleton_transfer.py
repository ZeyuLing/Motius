# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch

from ._utils import one_hot_1d, require_torch_tensors
from .interpolate import RadialBasisFunction
from .rig_utils import get_joint_children_ids, joint_world_to_local
from .transforms import SE3_from_Rt, align_vectors

try:
    from .align_vectors_warp import (
        align_vectors_warp,
        parallel_rodrigues_kabsch_warp,
        rodrigues_rotation_warp,
    )
except ImportError:
    align_vectors_warp = None
    rodrigues_rotation_warp = None
    parallel_rodrigues_kabsch_warp = None


class SkeletonTransfer(torch.nn.Module):
    def __init__(
        self,
        joint_parent_ids,
        bind_world_transforms,
        bind_shape,
        skinning_weights,
        rbf_kernel="linear",
        vertex_ids_to_exclude=None,
        freeze_rotations=None,
        skip_endjoints=True,
        use_sparse_rbf_matrix=True,
        use_warp_for_rotations=True,
        rotation_method="kabsch",
        skip_inverse_lbs=False,
    ):
        """Initialize a SkeletonTransfer instance for fitting a skeleton to new shapes.
        Args:
            joint_parent_ids: (J,) int array of joint parent indices
            bind_world_transforms: (J, 4, 4) array of joint bind poses in world space
            bind_shape: (V, 3) array of vertex positions in bind pose
            skinning_weights: (V, J) array of skinning weights
            rbf_kernel: type of RBF kernel to use for joint position regression
            vertex_ids_to_exclude: (V,) int array of vertex ids to exclude from the joint position regressors
            freeze_rotations: list of joint ids to freeze to bind pose (for Warp mode)
            skip_endjoints: bool, whether to skip rotation fitting for end joints (for Warp mode)
            use_sparse_rbf_matrix: bool, whether to use a sparse RBF matrix for joint position regression
            use_warp_for_rotations: bool, whether to use Warp-based rotation fitting (requires Warp)
            rotation_method: str, rotation extraction method ('kabsch' or 'newton-schulz')
            skip_inverse_lbs: bool, whether to skip Inverse LBS (skinned vertex fitting) and use identity R_init
        """
        super().__init__()
        if freeze_rotations is None:
            freeze_rotations = []
        require_torch_tensors(
            bind_world_transforms, bind_shape, skinning_weights, name="SkeletonTransfer inputs"
        )

        assert len(joint_parent_ids) == len(bind_world_transforms) == skinning_weights.shape[1], (
            "joint_names, joint_parent_ids, bind_world_transforms, and skinning_weights.shape[1] must have the same length"
        )
        self.num_joints = len(joint_parent_ids)

        if isinstance(joint_parent_ids, torch.Tensor):
            self.joint_parent_ids = joint_parent_ids.detach().cpu().tolist()
        else:
            self.joint_parent_ids = joint_parent_ids
        self.joint_children_ids = get_joint_children_ids(self.joint_parent_ids)
        bind_world_transforms = bind_world_transforms.detach()
        bind_shape = bind_shape.detach()
        skinning_weights = skinning_weights.detach()
        self.register_buffer("bind_world_transforms", bind_world_transforms, persistent=False)
        self.register_buffer(
            "bind_local_transforms",
            joint_world_to_local(bind_world_transforms, joint_parent_ids),
            persistent=False,
        )
        self.register_buffer("bind_shape", bind_shape, persistent=False)
        self.register_buffer("skinning_weights", skinning_weights, persistent=False)
        self.register_buffer("regressor_mask", None, persistent=False)
        self.sparse_rbf_matrix = None
        self.joint_pos_regressors = None
        self.rbf_kernel = rbf_kernel
        if isinstance(vertex_ids_to_exclude, torch.Tensor):
            self.vertex_ids_to_exclude = vertex_ids_to_exclude.detach().cpu().tolist()
        else:
            self.vertex_ids_to_exclude = vertex_ids_to_exclude
        self.freeze_rotations = set(freeze_rotations) if freeze_rotations else set()
        self.skip_endjoints = skip_endjoints
        self.use_sparse_rbf_matrix = use_sparse_rbf_matrix
        self.rotation_method = rotation_method
        self.skip_inverse_lbs = skip_inverse_lbs
        self._precompute_regressors()

        # Warp-specific precomputed data (initialized lazily)
        self.register_buffer("_warp_stage1_offsets", None, persistent=False)
        self.register_buffer("_warp_stage1_counts", None, persistent=False)
        self.register_buffer("_warp_stage1_skinned_vids_flat", None, persistent=False)
        self.register_buffer("_warp_stage1_joint_indices", None, persistent=False)
        self.register_buffer("_warp_stage1_repeat_indices", None, persistent=False)
        self.register_buffer("_warp_stage1_skinned_orig_centered", None, persistent=False)
        self._warp_stage1_joint_to_batch_idx = None

        self.register_buffer("_warp_stage2_offsets", None, persistent=False)
        self.register_buffer("_warp_stage2_counts", None, persistent=False)
        self.register_buffer("_warp_stage2_child_flat", None, persistent=False)
        self.register_buffer("_warp_stage2_joint_indices", None, persistent=False)
        self._warp_stage2_joint_to_batch_idx = None
        self.register_buffer("_warp_stage2_repeat_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_R_repeat_indices", None, persistent=False)
        self.register_buffer("_warp_frozen_joints", None, persistent=False)
        self.register_buffer("_warp_unskinned_end_joints", None, persistent=False)
        self.register_buffer("_warp_unskinned_end_joint_parents", None, persistent=False)
        self.register_buffer("_warp_stage2_n1_joint_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_n1_child_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_n1_to_stage1_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_n1_children_orig_centered", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_offsets", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_counts", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_child_flat", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_joint_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_to_stage1_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_children_orig_centered", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_repeat_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_n2_R_repeat_indices", None, persistent=False)
        self.register_buffer("_warp_stage2_joint_indices", None, persistent=False)
        self.register_buffer("_warp_frozen_parents", None, persistent=False)
        self.use_warp_for_rotations = use_warp_for_rotations
        if use_warp_for_rotations:
            if align_vectors_warp is None or rodrigues_rotation_warp is None:
                raise ImportError("Warp-based rotation fitting requires Warp to be installed.")
            self._precompute_warp_data()

    def update_bind(self, bind_world_transforms, bind_shape):
        """Update bind-pose data without rebuilding structural caches.

        This is much faster than constructing a new SkeletonTransfer and is
        suitable when only the identity (shape) changes but the topology,
        skinning weights, and skeleton structure remain the same.
        """
        self.bind_world_transforms = bind_world_transforms.detach()
        self.bind_local_transforms = joint_world_to_local(
            bind_world_transforms, self.joint_parent_ids
        )
        self.bind_shape = bind_shape.detach()

    @property
    def device(self):
        return self.bind_world_transforms.device

    @property
    def dtype(self):
        return self.bind_world_transforms.dtype

    def _apply(self, fn):
        super()._apply(fn)
        self._precompute_regressors()
        if self.use_warp_for_rotations:
            self._precompute_warp_data()
        return self

    def _precompute_regressors(self):
        regressor_mask = self.skinning_weights > 0.0
        regressor_mask &= self.skinning_weights[:, self.joint_parent_ids] > 0.0
        zero_weight_ids = torch.where(regressor_mask.sum(dim=0) == 0.0)[0]

        joint_parent_ids = torch.as_tensor(
            self.joint_parent_ids, dtype=torch.long, device=self.device
        )
        joint_parent_ids_cur = joint_parent_ids.clone()
        regressor_mask[:, zero_weight_ids] = self.skinning_weights[:, zero_weight_ids] > 0.0
        while len(zero_weight_ids) > 1:
            regressor_mask[:, zero_weight_ids] |= (
                self.skinning_weights[:, joint_parent_ids_cur][:, zero_weight_ids] > 0.0
            )
            zero_weight_ids = torch.where(regressor_mask.sum(dim=0) == 0.0)[0]
            joint_parent_ids_cur_update = joint_parent_ids[joint_parent_ids_cur]
            if torch.equal(joint_parent_ids_cur_update, joint_parent_ids_cur):
                break
            joint_parent_ids_cur = joint_parent_ids_cur_update

        if torch.equal(zero_weight_ids, torch.tensor([0, 1], device=self.device)):
            print("Aggregating children of hips")
            child_ids = get_joint_children_ids(joint_parent_ids)[1]
            regressor_mask[:, 1] = regressor_mask[:, child_ids].any(axis=1)

        if self.vertex_ids_to_exclude is not None:
            regressor_mask[self.vertex_ids_to_exclude] = False
        self.regressor_mask = regressor_mask

        if self.dtype == torch.float16:
            bind_shape_rbf = self.bind_shape.to(torch.float32)
        else:
            bind_shape_rbf = self.bind_shape

        self.joint_pos_regressors = [
            RadialBasisFunction(
                bind_shape_rbf[regressor_mask[:, i]],
                kernel=self.rbf_kernel,
                include_polynomial=True,
            )
            if i != 0
            else None
            for i in range(self.num_joints)
        ]
        if self.use_sparse_rbf_matrix:
            all_weights = []
            all_col_indices = []
            crow_indices = [0]

            for i, rbf in enumerate(self.joint_pos_regressors):
                if rbf is None:
                    crow_indices.append(crow_indices[-1])
                    continue

                joint_query_position = self.bind_world_transforms[i, :3, 3]
                if rbf.dtype != self.dtype:
                    joint_query_position = joint_query_position.to(rbf.dtype)
                w = rbf.get_basis_weights(joint_query_position)

                all_weights.append(w)
                all_col_indices.append(torch.where(regressor_mask[:, i])[0])
                crow_indices.append(crow_indices[-1] + len(w))

            flat_values = torch.cat(all_weights)
            flat_indices = torch.cat(all_col_indices)
            crow_indices_tensor = torch.tensor(crow_indices, device=self.device, dtype=torch.int64)

            self.sparse_rbf_matrix = torch.sparse_csr_tensor(
                crow_indices=crow_indices_tensor,
                col_indices=flat_indices,
                values=flat_values.to(self.dtype),
                size=(len(self.bind_world_transforms), len(self.bind_shape)),
                device=self.device,
                dtype=self.dtype,
            )
        else:
            self.sparse_rbf_matrix = None

    def _precompute_warp_data(self):
        """Precompute offsets, counts, and indices for Warp-based rotation fitting."""
        frozen = self.freeze_rotations

        unskinned_end_joints = []
        unskinned_end_joint_parents = []

        # ===== Stage 1: Skinned vertex alignment =====
        stage1_skinned_vids_flat = []
        stage1_counts_list = []
        stage1_joint_indices_list = []
        stage1_joint_to_batch_idx = {}

        batch_idx = 0
        for i in range(1, self.num_joints):
            if i in frozen:
                continue

            children = self.joint_children_ids[i]
            is_end_joint = len(children) == 0

            skinned_vids = torch.where(self.skinning_weights[:, i] > 0.01)[0]
            num_skinned = len(skinned_vids)

            if is_end_joint:
                if self.skip_endjoints:
                    unskinned_end_joints.append(i)
                    unskinned_end_joint_parents.append(self.joint_parent_ids[i])
                    continue
                elif num_skinned < 1:
                    unskinned_end_joints.append(i)
                    unskinned_end_joint_parents.append(self.joint_parent_ids[i])
                    continue

            stage1_joint_indices_list.append(i)
            stage1_joint_to_batch_idx[i] = batch_idx
            stage1_skinned_vids_flat.extend(skinned_vids.tolist())
            stage1_counts_list.append(num_skinned)
            batch_idx += 1

        if stage1_counts_list:
            counts = torch.tensor(stage1_counts_list, dtype=torch.int32, device=self.device)
            offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)

            self._warp_stage1_offsets = offsets
            self._warp_stage1_counts = counts
            self._warp_stage1_skinned_vids_flat = torch.tensor(
                stage1_skinned_vids_flat, dtype=torch.long, device=self.device
            )
            self._warp_stage1_joint_indices = torch.tensor(
                stage1_joint_indices_list, dtype=torch.long, device=self.device
            )
            self._warp_stage1_joint_to_batch_idx = stage1_joint_to_batch_idx

            skinned_orig = self.bind_shape[self._warp_stage1_skinned_vids_flat]
            joint_positions_for_verts = self.bind_world_transforms[
                self._warp_stage1_joint_indices, :3, 3
            ]
            joint_positions_expanded = torch.repeat_interleave(
                joint_positions_for_verts, self._warp_stage1_counts, dim=0
            )
            self._warp_stage1_skinned_orig_centered = skinned_orig - joint_positions_expanded

            repeat_indices = torch.repeat_interleave(
                torch.arange(len(stage1_joint_indices_list), device=self.device),
                self._warp_stage1_counts,
            )
            self._warp_stage1_repeat_indices = repeat_indices
        else:
            self._warp_stage1_offsets = torch.tensor([], dtype=torch.int32, device=self.device)
            self._warp_stage1_counts = torch.tensor([], dtype=torch.int32, device=self.device)
            self._warp_stage1_skinned_vids_flat = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage1_joint_indices = torch.tensor([], dtype=torch.long, device=self.device)
            self._warp_stage1_joint_to_batch_idx = {}
            self._warp_stage1_skinned_orig_centered = torch.tensor(
                [], dtype=self.dtype, device=self.device
            )
            self._warp_stage1_repeat_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )

        # ===== Stage 2: Child joint alignment =====
        stage2_n1_child_list = []
        stage2_n1_joint_indices_list = []

        stage2_n2_child_flat = []
        stage2_n2_counts_list = []
        stage2_n2_joint_indices_list = []
        stage2_n2_joint_to_batch_idx = {}

        batch_idx_n2 = 0
        for i in range(1, self.num_joints):
            children = self.joint_children_ids[i]

            if not children:
                continue
            if i in frozen:
                continue

            if len(children) == 1:
                stage2_n1_joint_indices_list.append(i)
                stage2_n1_child_list.append(children[0])
            else:
                stage2_n2_joint_indices_list.append(i)
                stage2_n2_joint_to_batch_idx[i] = batch_idx_n2
                stage2_n2_child_flat.extend(children)
                stage2_n2_counts_list.append(len(children))
                batch_idx_n2 += 1

        if stage2_n1_joint_indices_list:
            self._warp_stage2_n1_joint_indices = torch.tensor(
                stage2_n1_joint_indices_list, dtype=torch.long, device=self.device
            )
            self._warp_stage2_n1_child_indices = torch.tensor(
                stage2_n1_child_list, dtype=torch.long, device=self.device
            )

            stage1_batch_indices_n1 = [
                stage1_joint_to_batch_idx[j] for j in stage2_n1_joint_indices_list
            ]
            self._warp_stage2_n1_to_stage1_indices = torch.tensor(
                stage1_batch_indices_n1, dtype=torch.long, device=self.device
            )

            bind_world = self.bind_world_transforms
            pos_children_n1 = bind_world[self._warp_stage2_n1_child_indices, :3, 3]
            parent_positions_n1 = bind_world[self._warp_stage2_n1_joint_indices, :3, 3]
            self._warp_stage2_n1_children_orig_centered = pos_children_n1 - parent_positions_n1
        else:
            self._warp_stage2_n1_joint_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage2_n1_child_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage2_n1_to_stage1_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage2_n1_children_orig_centered = torch.tensor(
                [], dtype=self.dtype, device=self.device
            )

        if stage2_n2_counts_list:
            counts = torch.tensor(stage2_n2_counts_list, dtype=torch.int32, device=self.device)
            offsets = torch.zeros_like(counts)
            if counts.numel() > 1:
                offsets[1:] = torch.cumsum(counts[:-1], dim=0)

            self._warp_stage2_n2_offsets = offsets
            self._warp_stage2_n2_counts = counts
            self._warp_stage2_n2_child_flat = torch.tensor(
                stage2_n2_child_flat, dtype=torch.long, device=self.device
            )
            self._warp_stage2_n2_joint_indices = torch.tensor(
                stage2_n2_joint_indices_list, dtype=torch.long, device=self.device
            )

            stage1_batch_indices_n2 = [
                stage1_joint_to_batch_idx[j] for j in stage2_n2_joint_indices_list
            ]
            self._warp_stage2_n2_to_stage1_indices = torch.tensor(
                stage1_batch_indices_n2, dtype=torch.long, device=self.device
            )

            bind_world = self.bind_world_transforms
            pos_children_n2 = bind_world[self._warp_stage2_n2_child_flat, :3, 3]
            parent_positions_n2 = bind_world[self._warp_stage2_n2_joint_indices, :3, 3]
            parent_positions_n2_expanded = torch.repeat_interleave(
                parent_positions_n2, self._warp_stage2_n2_counts, dim=0
            )
            self._warp_stage2_n2_children_orig_centered = (
                pos_children_n2 - parent_positions_n2_expanded
            )

            repeat_indices = torch.repeat_interleave(
                torch.arange(len(stage2_n2_joint_indices_list), device=self.device),
                self._warp_stage2_n2_counts,
            )
            self._warp_stage2_n2_repeat_indices = repeat_indices
            self._warp_stage2_n2_R_repeat_indices = repeat_indices
        else:
            self._warp_stage2_n2_offsets = torch.tensor([], dtype=torch.int32, device=self.device)
            self._warp_stage2_n2_counts = torch.tensor([], dtype=torch.int32, device=self.device)
            self._warp_stage2_n2_child_flat = torch.tensor([], dtype=torch.long, device=self.device)
            self._warp_stage2_n2_joint_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage2_n2_to_stage1_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage2_n2_children_orig_centered = torch.tensor(
                [], dtype=self.dtype, device=self.device
            )
            self._warp_stage2_n2_repeat_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )
            self._warp_stage2_n2_R_repeat_indices = torch.tensor(
                [], dtype=torch.long, device=self.device
            )

        all_stage2_joints = stage2_n1_joint_indices_list + stage2_n2_joint_indices_list
        if all_stage2_joints:
            self._warp_stage2_joint_indices = torch.tensor(
                all_stage2_joints, dtype=torch.long, device=self.device
            )
        else:
            self._warp_stage2_joint_indices = torch.tensor([], dtype=torch.long, device=self.device)

        self._warp_frozen_joints = torch.tensor(list(frozen), dtype=torch.long, device=self.device)

        if unskinned_end_joints:
            self._warp_unskinned_end_joints = torch.tensor(
                unskinned_end_joints, dtype=torch.long, device=self.device
            )
            self._warp_unskinned_end_joint_parents = torch.tensor(
                unskinned_end_joint_parents, dtype=torch.long, device=self.device
            )
        else:
            self._warp_unskinned_end_joints = torch.tensor([], dtype=torch.long, device=self.device)
            self._warp_unskinned_end_joint_parents = torch.tensor(
                [], dtype=torch.long, device=self.device
            )

        frozen_parents = [self.joint_parent_ids[i] for i in frozen if i > 0]
        self._warp_frozen_parents = torch.tensor(
            frozen_parents, dtype=torch.long, device=self.device
        )

    def fit(self, target_shapes):
        """Fit the skeleton to new shapes by adjusting joint positions and orientations.
        Args:
            target_shapes: (B, V, 3) or (V, 3) array of new vertex positions
        Returns:
            target_bind_world_transforms: (B, J, 4, 4) or (J, 4, 4) array of new bind poses in world space
        """
        new_joint_positions = self.fit_joint_positions(target_shapes)

        if self.use_warp_for_rotations:
            world_bind_pose = self.fit_rotations_warp(
                new_joint_positions,
                target_shapes,
            )
        else:
            world_bind_pose = self.fit_joint_rotations(
                new_joint_positions,
                target_shapes,
            )
        return world_bind_pose

    def fit_joint_positions(self, target_shapes):
        """Fit the skeleton to new shapes by adjusting joint positions.
        Args:
            target_shapes: (B, V, 3) or (V, 3) array of new vertex positions
        Returns:
            new_joint_positions: (B, J, 3) or (J, 3) array of new joint positions
        """
        dtype, device = self.dtype, self.device
        J = self.num_joints

        added_batch = False
        if target_shapes.ndim == 2:
            target_shapes = target_shapes[None, :, :]
            added_batch = True
        B = target_shapes.shape[0]

        if self.sparse_rbf_matrix is not None:
            target_shapes_flat = target_shapes.permute(1, 0, 2).reshape(target_shapes.shape[1], -1)
            new_joint_positions = torch.mm(self.sparse_rbf_matrix, target_shapes_flat)
            new_joint_positions = new_joint_positions.reshape(J, B, 3).permute(1, 0, 2)
        else:
            cols = []
            root_pos = self.bind_world_transforms[0, :3, 3].to(dtype=dtype, device=device)
            root_pos = root_pos.view(1, 1, 3).expand(B, 1, 3)
            cols.append(root_pos)
            for i in range(1, J):
                target_vertex_positions = target_shapes[:, self.regressor_mask[:, i]]
                joint_query_position = self.bind_world_transforms[i : i + 1, :3, 3]
                pred = self.joint_pos_regressors[i].interpolate(
                    target_vertex_positions, joint_query_position
                )
                if pred.ndim == 2:
                    pred = pred[:, None, :]
                cols.append(pred)
            new_joint_positions = torch.cat(cols, dim=1)
        return new_joint_positions[0] if added_batch else new_joint_positions

    def fit_joint_rotations(self, new_joint_positions, target_shapes):
        """Fit the skeleton to new positions by adjusting joint orientations.
        Args:
            new_joint_positions: (B, J, 3) or (J, 3) array of new joint positions
            target_shapes: (B, V, 3) or (V, 3) array of new vertex positions
        Returns:
            world_bind_pose: (B, J, 4, 4) or (J, 4, 4) array of new bind poses in world space
        """
        dtype, device = self.dtype, self.device
        J = self.num_joints
        freeze_rotations = self.freeze_rotations
        skip_endjoints = self.skip_endjoints

        added_batch = False
        if new_joint_positions.ndim == 2:
            new_joint_positions = new_joint_positions[None, :, :]
            added_batch = True
        if new_joint_positions.shape[-2:] != (J, 3):
            raise ValueError(
                f"Expected new_joint_positions to have shape (...,{J},3); got {new_joint_positions.shape}"
            )
        if target_shapes.ndim == 2:
            target_shapes = target_shapes[None, :, :]

        t = new_joint_positions
        B = t.shape[0]

        bind_world = self.bind_world_transforms[None, ...].expand(B, J, 4, 4)
        bind_local = self.bind_local_transforms[None, ...].expand(B, J, 4, 4)

        R0 = self.bind_world_transforms[..., :3, :3].clone()
        R = R0[None, ...].expand(B, J, 3, 3)

        for i in range(1, J):
            jmask = one_hot_1d(J, i, dtype=dtype, device=device)[None, :, None, None]
            children = self.joint_children_ids[i]

            if not children and skip_endjoints:
                p = self.joint_parent_ids[i]
                R_parent = R[:, p : p + 1, :, :]
                R = R * (1 - jmask) + R_parent * jmask
                continue

            if i in freeze_rotations:
                p = self.joint_parent_ids[i]
                R_parent = R[:, p : p + 1, :, :]
                R_i_new = R_parent @ bind_local[:, i : i + 1, :3, :3]
                R = R * (1 - jmask) + R_i_new * jmask
                continue

            if self.skip_inverse_lbs:
                R_init = (
                    torch.eye(3, dtype=dtype, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, 1, 3, 3)
                )
            else:
                skinned_vids = torch.where(self.skinning_weights[:, i] > 0.01)[0]
                skinned_orig = (
                    self.bind_shape[skinned_vids] - self.bind_world_transforms[i, :3, 3]
                )[None, :, :]
                skinned_new = target_shapes[:, skinned_vids, :] - t[:, i : i + 1]
                R_init = align_vectors(skinned_new, skinned_orig)

            if len(children) > 0:
                pos_children_orig = (
                    bind_world[:, :, :3, 3][:, children] - bind_world[:, i : i + 1, :3, 3]
                )
                R_init_squeezed = R_init.squeeze(1)
                pos_children_orig = (R_init_squeezed @ pos_children_orig.swapaxes(-2, -1)).swapaxes(
                    -2, -1
                )
                pos_children_new = t[:, children, :] - t[:, i : i + 1, :]

                align_rot = align_vectors(pos_children_new, pos_children_orig)

                R_i_new = align_rot @ R_init_squeezed @ R[:, i, :, :]
            else:
                R_i_new = R_init.squeeze(1) @ R[:, i, :, :]

            R = R * (1 - jmask) + R_i_new[:, None, :, :] * jmask

        world_bind_pose = SE3_from_Rt(R, t)
        return world_bind_pose[0] if added_batch else world_bind_pose

    def fit_rotations_warp(self, new_joint_positions, target_shapes):
        """Warp-accelerated version of fit_joint_rotations using GPU-parallel alignment.

        Args:
            new_joint_positions: (B, J, 3) or (J, 3) array of new joint positions
            target_shapes: (B, V, 3) or (V, 3) array of new vertex positions

        Returns:
            world_bind_pose: (B, J, 4, 4) or (J, 4, 4) array of new bind poses in world space
        """
        if self._warp_stage1_offsets is None:
            self._precompute_warp_data()

        dtype, device = self.dtype, self.device
        J = self.num_joints

        added_batch = False
        if new_joint_positions.ndim == 2:
            new_joint_positions = new_joint_positions[None, :, :]
            added_batch = True
        if new_joint_positions.shape[-2:] != (J, 3):
            raise ValueError(
                f"Expected new_joint_positions to have shape (...,{J},3); got {new_joint_positions.shape}"
            )
        if target_shapes.ndim == 2:
            target_shapes = target_shapes[None, :, :]

        t = new_joint_positions
        B = t.shape[0]

        bind_local = self.bind_local_transforms[None, ...].expand(B, J, 4, 4)

        R0 = self.bind_world_transforms[..., :3, :3].clone()
        R = R0[None, ...].expand(B, J, 3, 3)

        # ===== Inverse LBS: Skinned vertices (single Warp call) =====
        R_init_all = None
        if len(self._warp_stage1_joint_indices) > 0:
            if self.skip_inverse_lbs:
                num_joints_stage1 = len(self._warp_stage1_joint_indices)
                R_init_all = (
                    torch.eye(3, dtype=dtype, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, num_joints_stage1, 3, 3)
                )
            else:
                skinned_orig = self._warp_stage1_skinned_orig_centered

                skinned_new = target_shapes[:, self._warp_stage1_skinned_vids_flat, :]
                new_joint_positions_for_verts = t[:, self._warp_stage1_joint_indices, :]
                new_joint_positions_expanded = new_joint_positions_for_verts[
                    :, self._warp_stage1_repeat_indices, :
                ]
                skinned_new = skinned_new - new_joint_positions_expanded

                skinned_orig_batched = skinned_orig.unsqueeze(0).expand(B, -1, -1).reshape(-1, 3)
                skinned_new_flat = skinned_new.reshape(-1, 3)

                num_joints_stage1 = len(self._warp_stage1_joint_indices)
                offsets_batched = (
                    self._warp_stage1_offsets.unsqueeze(0)
                    + torch.arange(B, device=device, dtype=torch.int32).unsqueeze(1)
                    * skinned_orig.shape[0]
                )
                offsets_batched = offsets_batched.flatten()
                counts_batched = self._warp_stage1_counts.unsqueeze(0).expand(B, -1).flatten()

                R_init_all = align_vectors_warp(
                    skinned_new_flat,
                    skinned_orig_batched,
                    offsets_batched,
                    counts_batched,
                    method=self.rotation_method,
                )
                R_init_all = R_init_all.reshape(B, num_joints_stage1, 3, 3)

        # ===== Stage 2: Child alignment =====
        align_rot_n1 = None
        align_rot_n2 = None

        if R_init_all is not None and parallel_rodrigues_kabsch_warp is not None:
            num_n1 = len(self._warp_stage2_n1_joint_indices)
            if num_n1 > 0:
                pos_children_orig_n1 = self._warp_stage2_n1_children_orig_centered
                pos_children_new_n1 = (
                    t[:, self._warp_stage2_n1_child_indices, :]
                    - t[:, self._warp_stage2_n1_joint_indices, :]
                )

                pos_children_orig_n1_batched = pos_children_orig_n1.unsqueeze(0).expand(B, -1, -1)
                R_init_for_n1 = R_init_all[:, self._warp_stage2_n1_to_stage1_indices, :, :]
                pos_children_orig_n1_rotated = torch.bmm(
                    R_init_for_n1.reshape(-1, 3, 3), pos_children_orig_n1_batched.reshape(-1, 3, 1)
                ).reshape(B, num_n1, 3)

                src_vecs_n1 = pos_children_orig_n1_rotated.reshape(-1, 3)
                tgt_vecs_n1 = pos_children_new_n1.reshape(-1, 3)
            else:
                src_vecs_n1 = torch.empty((0, 3), dtype=dtype, device=device)
                tgt_vecs_n1 = torch.empty((0, 3), dtype=dtype, device=device)

            num_n2 = len(self._warp_stage2_n2_joint_indices)
            if num_n2 > 0:
                pos_children_orig_n2 = self._warp_stage2_n2_children_orig_centered
                pos_children_new_n2 = t[:, self._warp_stage2_n2_child_flat, :]
                parent_positions_new_n2 = t[:, self._warp_stage2_n2_joint_indices, :]
                parent_positions_new_n2_expanded = parent_positions_new_n2[
                    :, self._warp_stage2_n2_repeat_indices, :
                ]
                pos_children_new_n2 = pos_children_new_n2 - parent_positions_new_n2_expanded

                pos_children_orig_n2_batched = pos_children_orig_n2.unsqueeze(0).expand(B, -1, -1)
                R_init_for_n2 = R_init_all[:, self._warp_stage2_n2_to_stage1_indices, :, :]
                R_init_n2_expanded = R_init_for_n2[:, self._warp_stage2_n2_R_repeat_indices, :, :]
                pos_children_orig_n2_rotated = torch.bmm(
                    R_init_n2_expanded.reshape(-1, 3, 3),
                    pos_children_orig_n2_batched.reshape(-1, 3, 1),
                ).reshape(B, -1, 3)

                pos_children_orig_n2_flat = pos_children_orig_n2_rotated.reshape(-1, 3)
                pos_children_new_n2_flat = pos_children_new_n2.reshape(-1, 3)

                offsets_batched_n2 = self._warp_stage2_n2_offsets.unsqueeze(0) + torch.arange(
                    B, device=device, dtype=torch.int32
                ).unsqueeze(1) * len(self._warp_stage2_n2_child_flat)
                offsets_batched_n2 = offsets_batched_n2.flatten()
                counts_batched_n2 = self._warp_stage2_n2_counts.unsqueeze(0).expand(B, -1).flatten()
            else:
                pos_children_orig_n2_flat = torch.empty((0, 3), dtype=dtype, device=device)
                pos_children_new_n2_flat = torch.empty((0, 3), dtype=dtype, device=device)
                offsets_batched_n2 = torch.empty((0,), dtype=torch.int32, device=device)
                counts_batched_n2 = torch.empty((0,), dtype=torch.int32, device=device)

            align_rot_n1_flat, align_rot_n2_flat = parallel_rodrigues_kabsch_warp(
                tgt_vecs_n1,
                src_vecs_n1,
                pos_children_new_n2_flat,
                pos_children_orig_n2_flat,
                offsets_batched_n2,
                counts_batched_n2,
                method=self.rotation_method,
            )

            align_rot_n1 = align_rot_n1_flat.reshape(B, num_n1, 3, 3)
            align_rot_n2 = align_rot_n2_flat.reshape(B, num_n2, 3, 3)

        # ===== Combine rotations (no loops, fully vectorized) =====
        R_new = R.clone()

        if R_init_all is not None:
            stage1_joints = self._warp_stage1_joint_indices
            R_new[:, stage1_joints, :, :] = torch.bmm(
                R_init_all.reshape(B * len(stage1_joints), 3, 3),
                R[:, stage1_joints, :, :].reshape(B * len(stage1_joints), 3, 3),
            ).reshape(B, len(stage1_joints), 3, 3)

        if align_rot_n1 is not None:
            stage2_n1_joints = self._warp_stage2_n1_joint_indices
            R_init_for_n1 = R_init_all[:, self._warp_stage2_n1_to_stage1_indices, :, :]

            temp = torch.bmm(
                align_rot_n1.reshape(B * len(stage2_n1_joints), 3, 3),
                R_init_for_n1.reshape(B * len(stage2_n1_joints), 3, 3),
            ).reshape(B, len(stage2_n1_joints), 3, 3)

            R_new[:, stage2_n1_joints, :, :] = torch.bmm(
                temp.reshape(B * len(stage2_n1_joints), 3, 3),
                R[:, stage2_n1_joints, :, :].reshape(B * len(stage2_n1_joints), 3, 3),
            ).reshape(B, len(stage2_n1_joints), 3, 3)

        if align_rot_n2 is not None:
            stage2_n2_joints = self._warp_stage2_n2_joint_indices
            R_init_for_n2 = R_init_all[:, self._warp_stage2_n2_to_stage1_indices, :, :]

            temp = torch.bmm(
                align_rot_n2.reshape(B * len(stage2_n2_joints), 3, 3),
                R_init_for_n2.reshape(B * len(stage2_n2_joints), 3, 3),
            ).reshape(B, len(stage2_n2_joints), 3, 3)

            R_new[:, stage2_n2_joints, :, :] = torch.bmm(
                temp.reshape(B * len(stage2_n2_joints), 3, 3),
                R[:, stage2_n2_joints, :, :].reshape(B * len(stage2_n2_joints), 3, 3),
            ).reshape(B, len(stage2_n2_joints), 3, 3)

        if len(self._warp_unskinned_end_joints) > 0:
            R_new[:, self._warp_unskinned_end_joints, :, :] = R_new[
                :, self._warp_unskinned_end_joint_parents, :, :
            ]

        if len(self._warp_frozen_joints) > 0:
            R_parents = R_new[:, self._warp_frozen_parents, :, :]
            R_bind_local = bind_local[0, self._warp_frozen_joints, :3, :3]

            num_frozen = len(self._warp_frozen_joints)
            R_frozen = torch.bmm(
                R_parents.reshape(B * num_frozen, 3, 3),
                R_bind_local.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * num_frozen, 3, 3),
            ).reshape(B, num_frozen, 3, 3)

            R_new[:, self._warp_frozen_joints, :, :] = R_frozen

        world_bind_pose = SE3_from_Rt(R_new, t)
        return world_bind_pose[0] if added_batch else world_bind_pose
