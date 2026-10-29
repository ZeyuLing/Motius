# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimodo motion representation (``<skel>_kimodo``, formerly ``*_smooth_root``).

Per-frame features relative to a smoothed root trajectory: smooth root position, global root heading
(cos/sin), per-joint local positions (pelvis-origin plus the hips offset from the smooth root),
global joint rotations as cont6d, per-joint velocities, and foot contacts — see the
``FeatureLayout`` in ``KimodoMotionRep.__init__``. ``__call__`` builds features from local rotations
+ root trajectory via FK; ``inverse`` reconstructs posed joints; ``create_conditions`` maps
constraint sets onto the feature blocks for the denoiser. The root-local channels (root rot-vel /
xz-vel / y) support the canonicalized local-root form used in training.
"""

import einops
import torch
from torch import Tensor

from motius.utils.kimodo import to_numpy

from motius.motion.representation.kimodo.geometry import cont6d_to_matrix, matrix_to_cont6d
from motius.motion.body_models.kimodo.kinematics import fk, joints_from_global_rots
from motius.motion.body_models.kimodo.transforms import global_rots_to_local_rots
from motius.utils.kimodo import ensure_batched
from motius.motion.representation.kimodo.conditioning import ConditionWriters
from motius.motion.representation.kimodo.feature_utils import RotateFeatures, compute_heading_angle, compute_vel_xyz
from motius.motion.representation.kimodo.feet import foot_detect_from_pos_and_vel
from motius.motion.representation.kimodo.smooth_root import get_smooth_root_pos
from motius.motion.representation.kimodo.reps.base import FeatureLayout, MotionRepBase


class KimodoMotionRep(MotionRepBase):
    """Global root / global joints rotations representation, relative to a smooth root."""

    def __init__(
        self,
        skeleton,
        fps,
        stats_path: str | None = None,
        stats=None,
        name: str | None = None,
        **kwargs,
    ):
        # `stats`, `name`, and **kwargs let KimodoMotionRep be built straight from a training config
        # via Hydra instantiate(), which may pass a Stats object (stats=...) and a `name` in addition
        # to a bare `stats_path`. kimodo's own config passes `stats_path` directly.
        if stats_path is None and stats is not None:
            stats_path = getattr(stats, "folder", None)
        self.name = name
        nbjoints = skeleton.nbjoints

        layout = FeatureLayout(
            size_dict={
                "smooth_root_pos": torch.Size([3]),
                "global_root_heading": torch.Size([2]),
                "local_joints_positions": torch.Size([nbjoints, 3]),
                "global_rot_data": torch.Size([nbjoints, 6]),
                "velocities": torch.Size([nbjoints, 3]),
                "foot_contacts": torch.Size([4]),
            },
            last_root_feature="global_root_heading",
            local_root_size_dict={
                "local_root_rot_vel": torch.Size([1]),
                "local_root_vel": torch.Size([2]),
                "global_root_y": torch.Size([1]),
            },
        )
        super().__init__(skeleton, fps, stats_path, layout=layout)

    @ensure_batched(local_joint_rots=5, root_positions=3, lengths=1)
    def __call__(
        self,
        local_joint_rots: torch.Tensor,
        root_positions: torch.Tensor,
        to_normalize: bool,
        to_canonicalize: bool = False,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Convert local rotations and root trajectory into smooth-root features.

        Args:
            local_joint_rots: Local joint rotation matrices ``[B, T, J, 3, 3]``.
            root_positions: Root positions ``[B, T, 3]``.
            to_normalize: Whether to normalize output features.
            to_canonicalize: Whether to canonicalize output features (False by default).
            lengths: Optional valid lengths for variable-length batches.

        Returns:
            Motion features with shape ``[B, T, motion_rep_dim]``.
        """
        device = local_joint_rots.device
        if lengths is None:
            assert local_joint_rots.shape[0] == 1, "If lenghts is not provided, the input should not be batched."
            lengths = torch.tensor([local_joint_rots.shape[1]], device=device)

        (
            global_joints_rots,
            global_joints_positions,
            local_joints_positions_origin_is_pelvis,
        ) = fk(local_joint_rots, root_positions, self.skeleton)

        root_heading_angle = compute_heading_angle(global_joints_positions, self.skeleton)
        global_root_heading = torch.stack([torch.cos(root_heading_angle), torch.sin(root_heading_angle)], dim=-1)

        smooth_root_pos = get_smooth_root_pos(root_positions)
        hips_offset = root_positions - smooth_root_pos
        hips_offset[..., 1] = root_positions[..., 1]
        local_joints_positions = local_joints_positions_origin_is_pelvis + hips_offset[:, :, None]

        velocities = compute_vel_xyz(global_joints_positions, self.fps, lengths=lengths)
        foot_contacts = foot_detect_from_pos_and_vel(global_joints_positions, velocities, self.skeleton, 0.15, 0.10)
        global_rot_data = matrix_to_cont6d(global_joints_rots)

        features, _ = einops.pack(
            [
                smooth_root_pos,
                global_root_heading,
                local_joints_positions,
                global_rot_data,
                velocities,
                foot_contacts,
            ],
            "batch time *",
        )

        if to_canonicalize:
            features = self.canonicalize(features, normalized=False)

        if to_normalize:
            features = self.normalize(features)
        return features

    @ensure_batched(features=3, angle=1)
    def rotate(self, features: torch.Tensor, angle: torch.Tensor):
        """Rotate root/joint positional and rotational features by heading."""
        # assume it is not normalized
        bs = features.shape[0]
        device = features.device
        [
            smooth_root_pos,
            global_root_heading,
            local_joints_positions,
            global_rot_data,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        if not isinstance(angle, torch.Tensor):
            angle = torch.tensor(angle, device=device)
        if len(angle.shape) == 0:
            angle = angle.repeat(bs)

        RF = RotateFeatures(angle)
        new_features, _ = einops.pack(
            [
                RF.rotate_positions(smooth_root_pos),
                RF.rotate_2d_positions(global_root_heading),
                RF.rotate_positions(local_joints_positions),
                RF.rotate_6d_rotations(global_rot_data),
                RF.rotate_positions(velocities),
                foot_contacts,
            ],
            "batch time *",
        )
        return new_features

    @ensure_batched(features=3, translation_2d=2)
    def translate_2d(
        self,
        features: torch.Tensor,
        translation_2d: torch.Tensor,
    ) -> torch.Tensor:
        """Translate smooth root planar position by ``(dx, dz)``."""
        # only move on the ground
        # If we need a translate_3D function, we should not forget to move the local_joints_positions as well
        bs = features.shape[0]
        if len(translation_2d.shape) == 1:
            translation_2d = translation_2d.repeat(bs, 1)

        new_features = features.clone()
        new_smooth_root_pos = new_features[:, :, self.slice_dict["smooth_root_pos"]]
        new_smooth_root_pos[:, :, 0] += translation_2d[:, [0]]
        new_smooth_root_pos[:, :, 2] += translation_2d[:, [1]]
        return new_features

    @ensure_batched(features=3)
    def inverse(
        self,
        features: torch.Tensor,
        is_normalized: bool,
        posed_joints_from="rotations",
        return_numpy: bool = False,
        return_all: bool = False,
        need_local_rots: bool = True,
    ) -> torch.Tensor:
        """Decode smooth-root features into motion tensors.

        ``return_all`` adds an ``all_positions`` dict with the joints decoded from the position and
        rotation blocks (``{"positions": ..., "rotations": ...}``). ``need_local_rots=False`` skips
        the global->local conversion (``local_rot_mats`` is then ``None`` in the output) — the
        training losses only consume positions / global rotations.
        """
        assert posed_joints_from in [
            "rotations",
            "positions",
        ], "posed_joints_from should 'rotations' or 'positions'"

        if is_normalized:
            features = self.unnormalize(features)

        [
            smooth_root_pos,
            global_root_heading,
            local_joints_positions,
            global_rot_data,
            velocities,
            foot_contacts,
        ] = einops.unpack(features, self.ps, "batch time *")

        global_rot_mats = cont6d_to_matrix(global_rot_data)
        # skeletons with a baked rest pose re-bake it inside fk, so their position path still
        # needs the local rotations; everywhere else positions come straight from the globals
        has_baked_rest = getattr(self.skeleton, "rest_local_rots", None) is not None
        local_rot_mats = (
            global_rots_to_local_rots(global_rot_mats, self.skeleton) if (need_local_rots or has_baked_rest) else None
        )

        posed_joints_from_pos = local_joints_positions.clone()
        posed_joints_from_pos[..., 0] += smooth_root_pos[..., None, 0]
        posed_joints_from_pos[..., 2] += smooth_root_pos[..., None, 2]
        root_positions = posed_joints_from_pos[..., self.skeleton.root_idx, :]
        foot_contacts = foot_contacts > 0.5

        # root-local pose (root at the origin) returned by the FK; the FK position loss uses it
        # directly instead of re-subtracting the root
        posed_joints_from_rots = posed_joints_norootpos = None
        if return_all or posed_joints_from == "rotations":
            if has_baked_rest:
                _, posed_joints_from_rots, posed_joints_norootpos = self.skeleton.fk(
                    local_rot_mats,
                    root_positions,
                )
            else:
                posed_joints_from_rots, posed_joints_norootpos = joints_from_global_rots(
                    global_rot_mats, root_positions, self.skeleton, return_norootpos=True
                )
        posed_joints = posed_joints_from_rots if posed_joints_from == "rotations" else posed_joints_from_pos

        output_tensor_dict = {
            "local_rot_mats": local_rot_mats,
            "global_rot_mats": global_rot_mats,
            "posed_joints": posed_joints,
            "posed_joints_norootpos": posed_joints_norootpos,
            "root_positions": root_positions,
            "smooth_root_pos": smooth_root_pos,
            "foot_contacts": foot_contacts,
            "global_root_heading": global_root_heading,
        }
        if return_all:
            output_tensor_dict["all_positions"] = {
                "positions": posed_joints_from_pos,
                "rotations": posed_joints_from_rots,
            }
        if return_numpy:
            return to_numpy(output_tensor_dict)
        return output_tensor_dict

    def create_conditions(
        self,
        index_dict: dict[Tensor],
        data_dict: dict[Tensor],
        length: int,
        to_normalize: bool,
        device: str,
    ):
        """Build sparse conditioning tensors for smooth-root representation.

        Returns:
            ``(observed_motion, motion_mask)`` where ``motion_mask`` is **bool** and unmasked
            cells of ``observed_motion`` keep whatever ``normalize`` produced from the zero
            initialization. This deliberately differs from ``ArdyMotionRep.create_conditions``
            (float mask, unmasked cells zeroed) — callers are contract-specific.
        """
        # create empty features and mask to be filled in
        observed_motion = torch.zeros(length, self.motion_rep_dim, device=device)
        motion_mask = torch.zeros(length, self.motion_rep_dim, dtype=bool, device=device)

        writers = ConditionWriters(
            observed_motion,
            motion_mask,
            self.slice_dict,
            index_dict,
            data_dict,
            device,
            joints_per_block=lambda fname: self.nbjoints,
        )
        _gather, _write_frames, _write_joints = writers.gather, writers.write_frames, writers.write_joints

        if (gathered := _gather("smooth_root_2d")) is not None:
            indices, smooth_root_2d = gathered
            _write_frames("smooth_root_pos", indices, smooth_root_2d, components=[0, 2])

        if (gathered := _gather("root_y_pos")) is not None:
            indices, root_pos_y = gathered
            _write_frames("smooth_root_pos", indices, root_pos_y, components=[1])

        if (gathered := _gather("global_root_heading")) is not None:
            indices, global_root_heading = gathered
            _write_frames("global_root_heading", indices, global_root_heading)

        if (gathered := _gather("global_joints_rots")) is not None:
            indices_lst, global_joints_rots = gathered
            _write_joints("global_rot_data", indices_lst, matrix_to_cont6d(global_joints_rots))

        if (gathered := _gather("global_joints_positions")) is not None:
            indices_lst, global_joints_positions = gathered
            T_indices = indices_lst[:, 0].contiguous()
            root_mask = motion_mask[T_indices, self.slice_dict["smooth_root_pos"]]
            if not root_mask[:, [0, 2]].all():
                raise ValueError("For constraining global positions, the smooth root should also be constrained.")
            local_reference = observed_motion[T_indices, self.slice_dict["smooth_root_pos"]].clone()
            local_reference[..., 1] = 0.0
            local_joints_positions = global_joints_positions - local_reference
            _write_joints("local_joints_positions", indices_lst, local_joints_positions)

        if to_normalize:
            observed_motion = self.normalize(observed_motion)
        return observed_motion, motion_mask
