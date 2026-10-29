# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Base motion representation: feature layout, normalization, and conditioning helpers."""

import os
from dataclasses import dataclass, field

import einops
import numpy as np
import torch
from einops import repeat

from motius.utils.kimodo import ensure_batched
from motius.motion.representation.kimodo.conditioning import build_condition_dicts
from motius.motion.representation.kimodo.feature_utils import compute_vel_angle, compute_vel_xyz
from motius.motion.representation.kimodo.stats import Stats


def _require_split_stats_layout(stats_path: str) -> None:
    """Raise if stats_path does not contain the required global_root, local_root, body subdirs."""
    subdirs = ("global_root", "local_root", "body")
    missing = []
    for name in subdirs:
        subpath = os.path.join(stats_path, name)
        mean_path = os.path.join(subpath, "mean.npy")
        if not os.path.isfile(mean_path):
            missing.append(f"{subpath}/ (mean.npy)")
    if missing:
        raise FileNotFoundError(
            f"Checkpoint stats must use the split layout with subfolders "
            f"global_root/, local_root/, and body/ under '{stats_path}'. "
            f"Missing or incomplete: {', '.join(missing)}. "
        )


@dataclass(frozen=True)
class FeatureLayout:
    """Declarative feature layout of a motion representation.

    A representation is a concatenation of named feature blocks (root first, then body).
    Declaring the layout as data — instead of subclass attributes set before
    ``super().__init__()`` — means a new representation (e.g. one with extra scene or
    object-contact blocks) is a layout + the transform methods, not a base-class contract
    to reverse-engineer.

    Attributes:
        size_dict: Ordered mapping of feature block name -> per-frame ``torch.Size``.
            Block order defines the feature vector layout.
        last_root_feature: Name of the last block belonging to the (global) root; every
            block up to and including it forms the root slice, the rest is the body.
        local_root_size_dict: Layout of the derived local-root features (used by
            ``global_root_to_local_root``), not part of the stored feature vector.
    """

    size_dict: dict
    last_root_feature: str
    local_root_size_dict: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.last_root_feature not in self.size_dict:
            raise ValueError(
                f"last_root_feature {self.last_root_feature!r} is not a feature block; "
                f"available: {list(self.size_dict)}"
            )


class MotionRepBase(torch.nn.Module):
    """Base class for motion representations used in generation and conditioning.

    Subclasses pass a :class:`FeatureLayout` describing their feature blocks and implement
    transform-specific methods such as ``__call__``, ``inverse``, ``rotate``,
    ``translate_2d`` and ``create_conditions``.

    This is an ``nn.Module`` (holding no parameters) so that the ``Stats`` and skeleton
    buffers follow the model when it is moved to a device, instead of being re-transferred
    from CPU on every normalize/unnormalize/FK call.
    """

    def __init__(
        self,
        skeleton,
        fps,
        stats_path: str | None = None,
        layout: FeatureLayout | None = None,
    ):
        """Initialize feature slicing metadata and optional normalization stats."""
        super().__init__()
        self.skeleton = skeleton
        self.fps = fps
        self.nbjoints = skeleton.nbjoints

        if layout is not None:
            self.size_dict = dict(layout.size_dict)
            self.last_root_feature = layout.last_root_feature
            self.local_root_size_dict = dict(layout.local_root_size_dict)
        elif not hasattr(self, "size_dict"):
            raise TypeError(
                "MotionRepBase needs a feature layout: pass layout=FeatureLayout(...) to "
                "__init__ (preferred), or set size_dict / last_root_feature / "
                "local_root_size_dict before calling super().__init__() (legacy contract)."
            )

        self.feature_names = list(self.size_dict.keys())
        self.ps = list(self.size_dict.values())
        self.nfeats_dict = {key: val.numel() for key, val in self.size_dict.items()}
        feats_cumsum = np.cumsum([0] + list(self.nfeats_dict.values())).tolist()
        self.slice_dict = {key: slice(feats_cumsum[i], feats_cumsum[i + 1]) for i, key in enumerate(self.feature_names)}

        self.motion_rep_dim = sum(self.nfeats_dict.values())
        self.root_slice = slice(0, self.slice_dict[self.last_root_feature].stop)
        self.body_slice = slice(self.root_slice.stop, self.motion_rep_dim)
        self.body_dim = self.body_slice.stop - self.body_slice.start
        self.global_root_dim = self.root_slice.stop
        self.local_root_dim = sum(val.numel() for val in self.local_root_size_dict.values())

        # Aliases for the dual-root-lineage call sites migrated onto this base.
        self.motion_root_dim = self.global_root_dim
        self.motion_body_dim = self.body_dim
        self.local_motion_rep_dim = self.local_root_dim + self.body_dim
        self.global_motion_rep_dim = self.motion_rep_dim

        if stats_path:
            self._load_stats(stats_path)

    def _load_stats(self, stats_path: str) -> None:
        """Populate per-part and combined normalization stats from ``stats_path``.

        Two on-disk layouts are supported:

        - **split**: subfolders ``global_root/``, ``local_root/``, ``body/`` each with their own
          ``mean.npy``/``std.npy``.
        - **flat**: a single ``mean.npy``/``std.npy`` covering the concatenated
          ``[global_root | local_root | body]`` feature vector, sliced into parts here.

        In both layouts ``self.stats`` covers only the ``[global_root | body]`` features that make up
        the motion representation; the local-root block is used solely by
        ``global_root_to_local_root``. Slices come back as ``Stats`` submodules so their buffers
        follow the module across ``.to(device)``.
        """
        gr, lr, bd = self.global_root_dim, self.local_root_dim, self.body_dim
        if os.path.isfile(os.path.join(stats_path, "global_root", "mean.npy")):
            _require_split_stats_layout(stats_path)
            self.global_root_stats = Stats(os.path.join(stats_path, "global_root"))
            self.local_root_stats = Stats(os.path.join(stats_path, "local_root"))
            self.body_stats = Stats(os.path.join(stats_path, "body"))
            mean = torch.cat([self.global_root_stats.mean, self.body_stats.mean])
            std = torch.cat([self.global_root_stats.std, self.body_stats.std])
            self.stats = Stats()
            self.stats.register_from_tensors(mean, std)
        elif os.path.isfile(os.path.join(stats_path, "mean.npy")):
            all_stats = Stats(stats_path)
            self.global_root_stats = all_stats.sliced(list(range(0, gr)))
            self.local_root_stats = all_stats.sliced(list(range(gr, gr + lr)))
            self.body_stats = all_stats.sliced(list(range(gr + lr, gr + lr + bd)))
            self.stats = all_stats.sliced(list(range(0, gr)) + list(range(gr + lr, gr + lr + bd)))
        else:
            raise FileNotFoundError(
                f"No stats found under '{stats_path}': expected either split subfolders "
                f"global_root/, local_root/, body/ (each with mean.npy), or a flat mean.npy/std.npy pair."
            )
        assert self.stats.mean.shape[0] == self.motion_rep_dim, (
            f"Stats dim {self.stats.mean.shape[0]} != motion_rep_dim {self.motion_rep_dim}"
        )

    @property
    def root_pos_name(self) -> str:
        """Name of the 3D root-position feature block (``root_pos`` or ``smooth_root_pos``)."""
        if "root_pos" in self.slice_dict:
            return "root_pos"
        if "smooth_root_pos" not in self.slice_dict:
            raise TypeError("This motion rep should have either a root_pos or smooth_root_pos field")
        return "smooth_root_pos"

    def get_root_pos(self, features: torch.Tensor, fallback_to_smooth: bool = True):
        """Extract root positions from a feature tensor.

        Supports both ``root_pos`` and ``smooth_root_pos`` representations.
        """
        if self.root_pos_name == "root_pos":
            return features[..., self.slice_dict["root_pos"]]

        if fallback_to_smooth:
            return features[:, :, self.slice_dict["smooth_root_pos"]]

        # else compute the root pos from the smooth root and local joints offset
        smooth_root_pos = features[:, :, self.slice_dict["smooth_root_pos"]].clone()
        local_joints_positions_flatten = features[..., self.slice_dict["local_joints_positions"]]
        hips_offset = local_joints_positions_flatten[..., :3]  # root (joint 0) xyz offset
        root_pos = torch.stack(
            [
                smooth_root_pos[..., 0] + hips_offset[..., 0],
                smooth_root_pos[..., 1],
                smooth_root_pos[..., 2] + hips_offset[..., 2],
            ],
            axis=-1,
        )
        return root_pos

    def recenter_root_motion(
        self,
        root_motion: torch.Tensor,
        center_frame_index: torch.Tensor,
        is_normalized: bool,
        to_normalize: bool,
        return_center_pos: bool = False,
    ):
        """Translate root x/z so a selected frame becomes the local origin.

        Assumes the root feature block starts with a 3D position (``root_pos`` or
        ``smooth_root_pos``), like ``global_root_to_local_root``.
        """
        batch_idx = torch.arange(root_motion.shape[0], device=root_motion.device)

        if is_normalized and to_normalize:
            # Stay in normalized space: normalization is per-channel affine, so the raw-space
            # recentering x - c becomes x_n - c/std_eps, with c/std_eps = x_n[center] + mean/std_eps.
            # Skips the unnormalize/normalize round trip and leaves y/heading channels bit-exact.
            mean, std_eps = self.global_root_stats._stats_for(root_motion.device, root_motion.dtype)
            center_xz_n = root_motion[batch_idx, center_frame_index.long()][:, [0, 2]]
            root_motion = root_motion.clone()
            root_motion[:, :, [0, 2]] -= (center_xz_n + mean[..., [0, 2]] / std_eps[..., [0, 2]])[:, None, :]
            if return_center_pos:
                center_pos = torch.zeros(root_motion.shape[0], 3, device=root_motion.device, dtype=root_motion.dtype)
                center_pos[:, [0, 2]] = center_xz_n * std_eps[..., [0, 2]] + mean[..., [0, 2]]
                return root_motion, center_pos
            return root_motion

        if is_normalized:
            root_motion = self.global_root_stats.unnormalize(root_motion)

        center_pos = root_motion[batch_idx, center_frame_index.long(), :3].clone()
        center_pos[:, 1] = 0
        root_motion = root_motion.clone()
        root_motion[:, :, [0, 2]] -= center_pos[:, None, [0, 2]]

        if to_normalize:
            root_motion = self.global_root_stats.normalize(root_motion)
        if return_center_pos:
            return root_motion, center_pos
        return root_motion

    @ensure_batched(root_features=3, lengths=1)
    def global_root_to_local_root(
        self,
        root_features: torch.Tensor,
        normalized: bool,
        lengths: torch.Tensor | None,
    ):
        """Convert global root features to local-root motion features.

        Args:
            root_features: Root feature tensor containing root position and
                global heading, shaped ``[B, T, D_root]``.
            normalized: Whether ``root_features`` are normalized.
            lengths: Optional valid lengths per sequence.

        Returns:
            Tensor ``[B, T, 4]`` with local root rotational velocity, planar
            velocity, and global root height.
        """
        if normalized:
            root_features = self.global_root_stats.unnormalize(root_features)

        [root_pos, global_root_heading] = einops.unpack(root_features, self.ps[:2], "batch time *")
        cos, sin = global_root_heading.unbind(-1)
        heading_angle = torch.arctan2(sin, cos)

        local_root_rot_vel = compute_vel_angle(heading_angle, self.fps, lengths=lengths)
        local_root_vel = compute_vel_xyz(
            root_pos[..., None, :],
            self.fps,
            lengths=lengths,
        )[..., 0, [0, 2]]
        global_root_y = root_pos[..., 1]
        local_root_motion = torch.cat(
            [
                local_root_rot_vel[..., None],
                local_root_vel,
                global_root_y[..., None],
            ],
            axis=-1,
        )

        if normalized:
            local_root_motion = self.local_root_stats.normalize(local_root_motion)
        return local_root_motion

    @ensure_batched(local_root_features=3, lengths=1)
    def local_root_to_global_root(
        self,
        local_root_features: torch.Tensor,
        normalized: bool,
        lengths: torch.Tensor | None = None,
    ):
        """Convert local-root motion features back to global root features.

        Exact inverse of :meth:`global_root_to_local_root`, up to the integration
        constants that the local representation does not store: the recovered
        heading and planar (x, z) position are relative to frame 0 (zero heading
        and origin at frame 0).

        Args:
            local_root_features: Local-root tensor ``[B, T, 4]`` with rotational
                velocity, planar velocity, and global root height.
            normalized: Whether ``local_root_features`` are normalized (the
                returned global root features are then normalized as well).
            lengths: Optional valid lengths per sequence (unused; kept for
                symmetry with ``global_root_to_local_root``).

        Returns:
            Tensor ``[B, T, D_root]`` with root position and global heading.
        """
        if normalized:
            local_root_features = self.local_root_stats.unnormalize(local_root_features)

        [local_root_rot_vel, local_root_vel, global_root_y] = einops.unpack(
            local_root_features, [[], [2], []], "batch time *"
        )

        # multiply by dt (= divide by fps) to recover true per-frame differences
        local_root_rot_vel = local_root_rot_vel / self.fps
        local_root_vel = local_root_vel / self.fps

        # heading angle from rotational velocity
        heading_angle = torch.zeros_like(local_root_rot_vel)
        heading_angle[..., 1:] = local_root_rot_vel[..., :-1]
        heading_angle = torch.cumsum(heading_angle, dim=-1)

        # integrate the planar position from the linear velocity (xz only); the
        # velocities are world-frame, so they are not rotated by the heading
        root_pos = torch.zeros(
            (*global_root_y.shape, 3),
            device=local_root_features.device,
            dtype=local_root_features.dtype,
        )
        root_pos[..., 1:, [0, 2]] = local_root_vel[..., :-1, :]
        root_pos = torch.cumsum(root_pos, dim=-2)
        root_pos[..., 1] = global_root_y

        global_root_features = torch.cat(
            [
                root_pos,
                torch.cos(heading_angle)[..., None],
                torch.sin(heading_angle)[..., None],
            ],
            dim=-1,
        )

        if normalized:
            global_root_features = self.global_root_stats.normalize(global_root_features)
        return global_root_features

    def get_root_heading_angle(self, features: torch.Tensor, normalized: bool = False) -> torch.Tensor:
        """Compute root heading angle from cosine/sine heading features.

        With ``normalized=True`` only the two heading channels get unnormalized (arctan2 needs
        raw values), instead of unnormalizing the whole feature tensor first. ``features`` may be
        the full feature tensor or just the root block — the heading slice indices are valid for
        both since the root block leads the feature layout.
        """
        heading_slice = self.slice_dict["global_root_heading"]
        global_root_heading = features[:, :, heading_slice]
        if normalized:
            mean, std_eps = self.stats._stats_for(features.device, features.dtype)
            global_root_heading = global_root_heading * std_eps[..., heading_slice] + mean[..., heading_slice]
        cos, sin = global_root_heading.unbind(-1)
        return torch.arctan2(sin, cos)

    @ensure_batched(features=3)
    def get_first_heading_angle(self, features: torch.Tensor, normalized: bool) -> torch.Tensor:
        """Return the frame-0 heading angle for each sequence in the batch.

        Args:
            features: Motion features of shape ``[B, T, motion_rep_dim]``.
            normalized: Whether ``features`` is normalized.

        Returns:
            Tensor of shape ``[B]`` with the per-sequence heading at frame 0.
        """
        return self.get_root_heading_angle(features, normalized=normalized)[..., 0]

    @ensure_batched(features=3)
    def rotate_to(
        self,
        features: torch.Tensor,
        target_angle: torch.Tensor,
        return_delta_angle=False,
    ):
        """Rotate each sequence so frame-0 heading matches ``target_angle``."""
        # rotate so that the first frame angle is the target
        # it put the motion_rep to the angle
        current_first_angle = self.get_root_heading_angle(features)[:, 0]
        delta_angle = target_angle - current_first_angle
        rotated_features = self.rotate(features, delta_angle)
        if return_delta_angle:
            return rotated_features, delta_angle
        return rotated_features

    @ensure_batched(features=3)
    def rotate_to_zero(
        self,
        features: torch.Tensor,
        return_delta_angle=False,
    ):
        """Rotate each sequence so frame-0 heading becomes zero."""
        target_angle = torch.zeros(len(features), device=features.device)
        return self.rotate_to(features, target_angle, return_delta_angle=return_delta_angle)

    @ensure_batched(features=3)
    def canonicalize_random_heading(
        self,
        features: torch.Tensor,
        return_delta_angle: bool = False,
        return_delta_pos: bool = False,
    ):
        """Canonicalize each sequence to a RANDOM frame-0 heading: frame-0 root ``(x, z)`` goes
        to the origin (like ``canonicalize``) but the heading is randomized instead of zeroed —
        the training-time heading augmentation. Rotation and recentering belong together here:
        a rotate-only augmentation leaves the window's world offset in the root features, so
        models trained directly on loader output (kimodo) see a shifted root distribution."""
        target_heading_angle = torch.rand(features.shape[0], device=features.device) * 2 * np.pi
        rotated = self.rotate_to(features, target_heading_angle, return_delta_angle=return_delta_angle)
        delta_angle = None
        if return_delta_angle:
            rotated, delta_angle = rotated
        translated = self.translate_2d_to_zero(rotated, return_delta_pos=return_delta_pos)
        delta_pos = None
        if return_delta_pos:
            translated, delta_pos = translated
        extras = [x for x in (delta_angle, delta_pos) if x is not None]
        if extras:
            return (translated, *extras)
        return translated

    @ensure_batched(features=3, target_2d_pos=2)
    def translate_2d_to(
        self,
        features: torch.Tensor,
        target_2d_pos: torch.Tensor,
        return_delta_pos: bool = False,
    ) -> torch.Tensor:
        """Translate each sequence so frame-0 root ``(x, z)`` matches a target."""
        root_pos = self.get_root_pos(features)
        current_first_2d_pos = root_pos[:, 0, [0, 2]].clone()
        delta_2d_pos = target_2d_pos - current_first_2d_pos
        translated_features = self.translate_2d(features, delta_2d_pos)
        if return_delta_pos:
            return translated_features, delta_2d_pos
        return translated_features

    @ensure_batched(features=3)
    def translate_2d_to_zero(
        self,
        features: torch.Tensor,
        return_delta_pos: bool = False,
    ) -> torch.Tensor:
        """Translate each sequence so frame-0 root ``(x, z)`` is at the origin."""
        target_2d_pos = torch.zeros(len(features), 2, device=features.device)
        return self.translate_2d_to(features, target_2d_pos, return_delta_pos=return_delta_pos)

    @ensure_batched(features=3)
    def canonicalize(self, features: torch.Tensor, normalized: bool = False):
        """Canonicalize heading and planar position at frame 0."""
        if normalized:
            features = self.unnormalize(features)
        rotated_features = self.rotate_to_zero(features)
        canonicalized_features = self.translate_2d_to_zero(rotated_features)
        if normalized:
            canonicalized_features = self.normalize(canonicalized_features)
        return canonicalized_features

    def normalize(self, features):
        """Normalize features."""
        return self.stats.normalize(features)

    def unnormalize(self, features):
        """Undo feature normalization."""
        return self.stats.unnormalize(features)

    def feature_indices(self, feature_name: str) -> np.ndarray:
        """All flat feature indices of a feature block."""
        s = self.slice_dict[feature_name]
        return np.arange(s.start, s.stop, dtype=np.int64)

    def joint_feature_indices(
        self,
        feature_name: str,
        joint_names: list[str],
        joint_order: list[str] | None = None,
    ) -> np.ndarray:
        """Flat feature indices covering ``joint_names`` inside a per-joint feature block.

        The per-joint width (e.g. 3 for positions, 6 for cont6d rotations) is derived from the
        block size. ``joint_order`` is the joint ordering of the block; it defaults to the
        skeleton bone order.

        This is the single place mapping (feature block, joint) -> feature columns; both the
        training-time mask builders and the inference-time constraint conditioning rely on it.

        Results are memoized per instance (deterministic in ``slice_dict``/``skeleton``, which
        are fixed after init): the mask builders call this per keyframe per sample, and
        rebuilding the name->index dict each call dominated their cost. The cache stays tiny
        (one entry per distinct (block, joints, order) combination). Callers must not mutate
        the returned array.
        """
        cache = self.__dict__.setdefault("_joint_feature_indices_cache", {})
        key = (
            feature_name,
            tuple(joint_names),
            None if joint_order is None else tuple(joint_order),
        )
        cached = cache.get(key)
        if cached is not None:
            return cached

        s = self.slice_dict[feature_name]
        order = joint_order if joint_order is not None else self.skeleton.bone_order_names
        stride = (s.stop - s.start) // len(order)
        joint_index = {name: idx for idx, name in enumerate(order)}
        indices = np.array(
            [s.start + joint_index[name] * stride + j for name in joint_names for j in range(stride)],
            dtype=np.int64,
        )
        indices.setflags(write=False)  # shared across calls; guard against accidental mutation
        cache[key] = indices
        return indices

    def extract_root(self, features: torch.Tensor) -> torch.Tensor:
        """Extract the global-root features from a full motion tensor."""
        return features[..., self.root_slice]

    def extract_body(self, features: torch.Tensor) -> torch.Tensor:
        """Extract the body features from a full motion tensor."""
        return features[..., self.body_slice]

    def concat_root_body(self, root: torch.Tensor, body: torch.Tensor) -> torch.Tensor:
        """Concatenate root and body features into a full motion feature tensor."""
        return torch.cat([root, body], dim=-1)

    def create_conditions_from_constraints(
        self,
        constraints_lst: list,
        length: int,
        to_normalize: bool,
        device: str,
    ):
        """Create a conditioning tensor and mask from constraint objects."""
        index_dict, data_dict = build_condition_dicts(constraints_lst)
        return self.create_conditions(index_dict, data_dict, length, to_normalize, device)

    def create_conditions_from_constraints_batched(
        self,
        constraints_lst: list | list[list],
        lengths: torch.Tensor,
        to_normalize: bool,
        device: str,
    ):
        """Batched version of ``create_conditions_from_constraints``.

        Supports either one shared constraint list for all batch elements, or a per-sample list of
        constraint lists.
        """
        num_samples = len(lengths)
        if not constraints_lst or not isinstance(constraints_lst[0], list):
            # If no constraints, or constraints are shared across the batch,
            # build once and repeat.
            observed_motion, motion_mask = self.create_conditions_from_constraints(
                constraints_lst, int(lengths.max()), to_normalize, device
            )
            observed_motion = repeat(observed_motion, "t d -> b t d", b=num_samples)
            motion_mask = repeat(motion_mask, "t d -> b t d", b=num_samples)
            return observed_motion, motion_mask

        length = int(lengths.max())
        observed_motion_lst = []
        motion_mask_lst = []
        for constraints_lst_el in constraints_lst:
            observed_motion, motion_mask = self.create_conditions_from_constraints(
                constraints_lst_el,
                length,
                to_normalize,
                device,
            )
            observed_motion_lst.append(observed_motion)
            motion_mask_lst.append(motion_mask)
        observed_motion = torch.stack(observed_motion_lst, axis=0)
        motion_mask = torch.stack(motion_mask_lst, axis=0)
        return observed_motion, motion_mask
