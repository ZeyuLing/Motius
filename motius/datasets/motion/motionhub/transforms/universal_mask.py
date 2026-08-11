"""Universal mask sampling for M2M completion training.

Implements 7 mask strategies (M1-M7) covering 25+ application scenarios
for universal motion completion. Operates on a (T, 23) joint-group grid
(1 translation group + 22 SMPL joints), then expands to (T, D) where
D depends on the motion representation (default 135 dims = 3 abs transl
+ 22×6 rot6d).

Output keys are identical to :class:`PrepareM2MCompletion`:
  - ``src_motion``: full motion (all frames)
  - ``tgt_motion``: full motion (same, for loss computation)
  - ``src_mask``: binary mask ``(T, D)``, 1=needs generation, 0=keep
  - ``tgt_length``: number of valid frames
  - ``src_length``: same as tgt_length

Strategies
----------
M1 (random_cell)         : each (t, j) cell independently Bernoulli(p), p~U[0.01,0.95]
M2 (random_block)        : random time intervals x random joint subsets
M3 (temporal_contiguous) : 1+ contiguous temporal segments, all joints
M4 (joint_contiguous)    : random body part groups, all or partial frames
M5 (full_mask)           : entire grid = 1 (unconditional generation)
M6 (keyframe_sparse)     : K random keyframes preserved, rest masked
M7 (scattered_joint)     : scattered (frame, joint) spots + temporal dilation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import os

import numpy as np
import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS

# -----------------------------------------------------------------------
# Joint group constants
# -----------------------------------------------------------------------
# smpl_22 + rotation_6d + absolute translation = 135 dims per frame.
# 23 groups: dims[0:3] = translation (3 abs), dims[3+6*i : 3+6*(i+1)] for i=0..21.
NUM_JOINT_GROUPS = 23  # 1 translation + 22 joints
TRANSL_DIM = 3         # absolute translation dims
JOINT_ROT_DIM = 6      # rotation_6d per joint
TOTAL_DIM = TRANSL_DIM + 22 * JOINT_ROT_DIM  # 135

# Joint index mapping (into 23-group space, 0-indexed):
#  0  = translation (3 absolute)
#  1  = Pelvis (root orientation)
#  2  = L_Hip
#  3  = R_Hip
#  4  = Spine1
#  5  = L_Knee
#  6  = R_Knee
#  7  = Spine2
#  8  = L_Ankle
#  9  = R_Ankle
#  10 = Spine3
#  11 = L_Foot
#  12 = R_Foot
#  13 = Neck
#  14 = L_Collar
#  15 = R_Collar
#  16 = Head
#  17 = L_Shoulder
#  18 = R_Shoulder
#  19 = L_Elbow
#  20 = R_Elbow
#  21 = L_Wrist
#  22 = R_Wrist

# Body part groups (indices into 23-group space)
UPPER_BODY = [0, 7, 10, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
LOWER_BODY = [2, 3, 5, 6, 8, 9, 11, 12]
LEFT_ARM = [14, 17, 19, 21]    # L_Collar, L_Shoulder, L_Elbow, L_Wrist
RIGHT_ARM = [15, 18, 20, 22]   # R_Collar, R_Shoulder, R_Elbow, R_Wrist
LEFT_LEG = [2, 5, 8, 11]       # L_Hip, L_Knee, L_Ankle, L_Foot
RIGHT_LEG = [3, 6, 9, 12]      # R_Hip, R_Knee, R_Ankle, R_Foot
SPINE_HEAD = [4, 7, 10, 13, 16]  # Spine1, Spine2, Spine3, Neck, Head
FEET = [8, 9, 11, 12]          # L/R_Ankle, L/R_Foot
TRANSLATION = [0]
ALL_JOINTS_NO_TRANSL = list(range(1, 23))

BODY_PART_GROUPS: Dict[str, List[int]] = {
    'upper': UPPER_BODY,
    'lower': LOWER_BODY,
    'left_arm': LEFT_ARM,
    'right_arm': RIGHT_ARM,
    'left_leg': LEFT_LEG,
    'right_leg': RIGHT_LEG,
    'spine_head': SPINE_HEAD,
    'feet': FEET,
    'translation': TRANSLATION,
    'joints_only': ALL_JOINTS_NO_TRANSL,
}

# Default strategy weights: M1:20, M2:12, M3:23, M4:15, M5:5, M6:15, M7:10
DEFAULT_STRATEGY_WEIGHTS = {
    'm1_random_cell': 0.20,
    'm2_random_block': 0.12,
    'm3_temporal_contiguous': 0.23,
    'm4_joint_contiguous': 0.15,
    'm5_full_mask': 0.05,
    'm6_keyframe_sparse': 0.15,
    'm7_scattered_joint': 0.10,
}

VALID_STRATEGIES = set(DEFAULT_STRATEGY_WEIGHTS.keys())


def expand_grid_to_mask(grid: np.ndarray) -> torch.Tensor:
    """Expand joint-group grid ``(T, 23)`` to full mask ``(T, D)``.

    Translation group (col 0) expands to ``TRANSL_DIM`` dims (default 3).
    Each joint group (cols 1..22) expands to ``JOINT_ROT_DIM`` dims (default 6).

    Parameters
    ----------
    grid : np.ndarray
        Binary array of shape ``(T, 23)`` with values in {0, 1}.

    Returns
    -------
    torch.Tensor
        Float tensor of shape ``(T, TOTAL_DIM)``.
    """
    T = grid.shape[0]
    mask = torch.from_numpy(grid.astype(np.float32))
    # Expand translation group (col 0) to TRANSL_DIM dims
    transl_mask = mask[:, 0:1].repeat(1, TRANSL_DIM)  # (T, 3)
    # Expand joint groups (cols 1..22) to JOINT_ROT_DIM each
    joint_mask = mask[:, 1:].repeat_interleave(JOINT_ROT_DIM, dim=-1)  # (T, 132)
    return torch.cat([transl_mask, joint_mask], dim=-1)  # (T, 135)


# -----------------------------------------------------------------------
# Strategy implementations (operate on (T, 23) grid in-place)
# -----------------------------------------------------------------------

def m1_random_cell(T: int, grid: np.ndarray, rng: np.random.RandomState) -> None:
    """M1: Random cell — each (t, j) cell independently Bernoulli(p).

    ``p`` is sampled from U[0.01, 0.95] per call.
    """
    p = rng.uniform(0.01, 0.95)
    grid[:] = (rng.rand(T, NUM_JOINT_GROUPS) < p).astype(np.float32)


def m2_random_block(T: int, grid: np.ndarray, rng: np.random.RandomState) -> None:
    """M2: Random block — 1-3 rectangular blocks of random time x joints."""
    num_blocks = rng.randint(1, 4)  # 1-3
    for _ in range(num_blocks):
        # Sample time interval
        t1 = rng.randint(0, T)
        t2 = rng.randint(t1 + 1, T + 1)  # at least 1 frame

        # Sample random joint subset (1 to 10 joints)
        max_joints = min(10, NUM_JOINT_GROUPS)
        n_joints = rng.randint(1, max_joints + 1)
        joints = rng.choice(NUM_JOINT_GROUPS, size=n_joints, replace=False)

        grid[t1:t2, joints] = 1.0


def m3_temporal_contiguous(
    T: int, grid: np.ndarray, rng: np.random.RandomState
) -> None:
    """M3: Temporal contiguous — contiguous segment(s), all joints masked.

    Randomly picks one of five sub-modes:
      - inbetween:  [0..past | 1..middle | 0..future]
      - prediction: [0..past | 1..rest]
      - prefix:     [1..start | 0..rest]
      - outpainting:[1..start | 0..middle | 1..end]
      - multi_gap:  multiple alternating masked/unmasked segments
    """
    modes = ['inbetween', 'prediction', 'prefix', 'outpainting', 'multi_gap']
    mode = modes[rng.randint(0, len(modes))]

    if mode == 'inbetween':
        # Keep first and last portion, mask middle
        if T <= 2:
            grid[:] = 1.0
            return
        past_end = rng.randint(1, max(2, T // 2 + 1))
        future_start = rng.randint(max(past_end + 1, T // 2), T)
        grid[past_end:future_start, :] = 1.0

    elif mode == 'prediction':
        # Keep beginning, mask rest
        if T <= 1:
            grid[:] = 1.0
            return
        split = rng.randint(1, max(2, T * 2 // 3))
        grid[split:, :] = 1.0

    elif mode == 'prefix':
        # Mask beginning, keep rest
        if T <= 1:
            grid[:] = 1.0
            return
        split = rng.randint(max(1, T // 3), T)
        grid[:split, :] = 1.0

    elif mode == 'outpainting':
        # Mask beginning and end, keep middle
        if T <= 2:
            grid[:] = 1.0
            return
        start_end = rng.randint(1, max(2, T // 3 + 1))
        end_start = rng.randint(max(start_end + 1, T * 2 // 3), T)
        grid[:start_end, :] = 1.0
        grid[end_start:, :] = 1.0

    elif mode == 'multi_gap':
        # Multiple masked gaps between known segments
        if T <= 4:
            # Too short for multi-gap, fallback to single gap
            mid = T // 2
            grid[mid:, :] = 1.0
            return
        num_segments = rng.randint(2, min(5, T // 2 + 1))
        # Generate random boundaries and sort
        boundaries = sorted(rng.choice(range(1, T), size=min(num_segments * 2, T - 1), replace=False))
        # Alternate: first segment known, second masked, etc.
        for i in range(0, len(boundaries) - 1, 2):
            grid[boundaries[i]:boundaries[i + 1], :] = 1.0


def m4_joint_contiguous(
    T: int, grid: np.ndarray, rng: np.random.RandomState
) -> None:
    """M4: Joint contiguous — mask body part groups across all frames.

    With 60% probability, picks 1-3 named body part groups.
    With 40% probability, picks K=1..8 individual random joints.
    """
    use_body_parts = rng.rand() < 0.6

    if use_body_parts:
        part_names = list(BODY_PART_GROUPS.keys())
        num_parts = rng.randint(1, min(4, len(part_names) + 1))
        selected_parts = rng.choice(part_names, size=num_parts, replace=False)
        joints = set()
        for part in selected_parts:
            joints.update(BODY_PART_GROUPS[part])
        joints = list(joints)
    else:
        k = rng.randint(1, min(9, NUM_JOINT_GROUPS + 1))
        joints = rng.choice(NUM_JOINT_GROUPS, size=k, replace=False).tolist()

    # 30% probability: temporal partial — only mask selected joints on random
    # frame subset (10%-80%), producing (frame, joint) partial patterns.
    temporal_partial = rng.rand() < 0.3
    if temporal_partial:
        frame_ratio = rng.uniform(0.1, 0.8)
        n_frames = max(1, int(T * frame_ratio))
        masked_frames = rng.choice(T, size=n_frames, replace=False)
        grid[np.ix_(masked_frames, joints)] = 1.0
    else:
        grid[:, joints] = 1.0  # original: all frames


def m5_full_mask(T: int, grid: np.ndarray, rng: np.random.RandomState) -> None:
    """M5: Full mask — everything needs generation (unconditional)."""
    grid[:] = 1.0


def m6_keyframe_sparse(
    T: int, grid: np.ndarray, rng: np.random.RandomState
) -> None:
    """M6: Keyframe sparse — K random keyframes preserved, rest masked.

    With 70% probability, keyframes preserve all joints (full keyframe).
    With 30% probability, keyframes only preserve a random subset of joints
    (partial keyframe, covering scenario E2).
    """
    # Start with everything masked
    grid[:] = 1.0

    # Select K keyframes
    max_k = max(2, T // 8)
    k = rng.randint(1, max_k + 1)
    k = min(k, T)  # can't have more keyframes than frames
    keyframes = rng.choice(T, size=k, replace=False)

    partial_keyframe = rng.rand() < 0.3

    for kf in keyframes:
        if partial_keyframe:
            # Only preserve a random subset of joints
            n_preserve = rng.randint(1, NUM_JOINT_GROUPS + 1)
            preserved_joints = rng.choice(
                NUM_JOINT_GROUPS, size=n_preserve, replace=False
            )
            grid[kf, preserved_joints] = 0.0
        else:
            # Preserve all joints
            grid[kf, :] = 0.0


def m7_scattered_joint(
    T: int, grid: np.ndarray, rng: np.random.RandomState
) -> None:
    """M7: Scattered joint — simulate checker/adaptive repair masks.

    Randomly sample N_spots scattered (frame, joint) flag-points, then apply
    temporal dilation (1-8 frames each side).  Does NOT mask translation
    (col 0).  Grid ratio typically 1-20%.
    """
    n_spots = rng.randint(1, max(2, T // 5))  # 1 to T//5 spots
    for _ in range(n_spots):
        frame = rng.randint(0, T)
        # 1-3 joints per spot (from joint indices 1..22, skip translation)
        n_joints = rng.randint(1, min(4, NUM_JOINT_GROUPS))
        joints = rng.choice(
            range(1, NUM_JOINT_GROUPS), size=n_joints, replace=False
        )
        # Temporal dilation: 1-8 frames each side
        dilate = rng.randint(1, 9)
        f_start = max(0, frame - dilate)
        f_end = min(T, frame + dilate + 1)
        grid[f_start:f_end, joints] = 1.0


# Map strategy name → function
_STRATEGY_FN = {
    'm1_random_cell': m1_random_cell,
    'm2_random_block': m2_random_block,
    'm3_temporal_contiguous': m3_temporal_contiguous,
    'm4_joint_contiguous': m4_joint_contiguous,
    'm5_full_mask': m5_full_mask,
    'm6_keyframe_sparse': m6_keyframe_sparse,
    'm7_scattered_joint': m7_scattered_joint,
}


# -----------------------------------------------------------------------
# Transform
# -----------------------------------------------------------------------

@TRANSFORMS.register_module()
class PrepareM2MUniversalMask(BaseTransform):
    """Universal mask sampling for M2M completion training.

    Operates on ``(T, 23)`` joint-group grid, expands to ``(T, D)``.
    23 groups: dims[0:3]=translation(3 abs), dims[3+6*i:3+6*(i+1)] for i=0..21 joints.

    Output keys (identical to :class:`PrepareM2MCompletion`):
      - ``src_motion``: full motion
      - ``tgt_motion``: full motion
      - ``src_mask``: binary mask ``(T, D)``, 1=generate, 0=keep
      - ``tgt_length``: int, number of frames
      - ``src_length``: int, same as tgt_length

    Parameters
    ----------
    key : str
        Motion key in results dict.
    strategy_weights : dict or None
        Mapping ``{strategy_name: weight}``. Weights are normalized to sum
        to 1. If None, uses default weights (M1:20, M2:12, M3:23, M4:15,
        M5:5, M6:15, M7:10).
    min_mask_ratio : float
        Safety net: if sampled mask ratio < this, add random cells.
    max_mask_ratio : float
        Safety net: if sampled mask ratio > this (and not M5), remove cells.
    """

    # Expose constants at class level for testing convenience
    NUM_JOINT_GROUPS = NUM_JOINT_GROUPS
    TRANSL_DIM = TRANSL_DIM
    JOINT_ROT_DIM = JOINT_ROT_DIM
    TOTAL_DIM = TOTAL_DIM
    BODY_PART_GROUPS = BODY_PART_GROUPS

    def __init__(
        self,
        key: str = 'motion',
        strategy_weights: Optional[Dict[str, float]] = None,
        min_mask_ratio: float = 0.05,
        max_mask_ratio: float = 0.95,
        edit_repair_prob: float = 0.0,
        corruptor_names: Optional[List[str]] = None,
        max_corruptions: int = 2,
    ):
        self.key = key
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.edit_repair_prob = edit_repair_prob
        self.corruptor_names = corruptor_names or []
        self.max_corruptions = max_corruptions
        self._corruptor_cache: Dict[str, Any] = {}

        # Validate and normalize strategy weights
        if strategy_weights is None:
            strategy_weights = dict(DEFAULT_STRATEGY_WEIGHTS)
        else:
            strategy_weights = dict(strategy_weights)

        for name in strategy_weights:
            if name not in VALID_STRATEGIES:
                raise ValueError(
                    f'Unknown strategy {name!r}. '
                    f'Valid strategies: {sorted(VALID_STRATEGIES)}'
                )

        total = sum(strategy_weights.values())
        if total <= 0:
            raise ValueError('Strategy weights must sum to a positive number.')
        self.strategy_names = list(strategy_weights.keys())
        self.strategy_probs = np.array(
            [strategy_weights[k] / total for k in self.strategy_names]
        )

    def _enforce_mask_ratio(
        self,
        grid: np.ndarray,
        strategy: str,
        rng: np.random.RandomState,
    ) -> None:
        """Enforce min/max mask ratio by adding/removing random masked cells.

        Modifies ``grid`` in-place. M5 (full_mask) is exempt from max
        enforcement since ratio=1.0 is its intended behavior.
        """
        T, J = grid.shape
        total_cells = T * J
        if total_cells == 0:
            return

        mask_ratio = grid.sum() / total_cells

        # Enforce minimum
        if mask_ratio < self.min_mask_ratio:
            target_count = int(np.ceil(self.min_mask_ratio * total_cells))
            current_count = int(grid.sum())
            need = target_count - current_count
            if need > 0:
                zero_indices = np.argwhere(grid == 0)
                if len(zero_indices) > 0:
                    add_count = min(need, len(zero_indices))
                    chosen = rng.choice(len(zero_indices), size=add_count, replace=False)
                    for idx in chosen:
                        r, c = zero_indices[idx]
                        grid[r, c] = 1.0

        # Enforce maximum (skip for M5 full_mask)
        if strategy != 'm5_full_mask':
            mask_ratio = grid.sum() / total_cells
            if mask_ratio > self.max_mask_ratio:
                target_count = int(np.floor(self.max_mask_ratio * total_cells))
                current_count = int(grid.sum())
                excess = current_count - target_count
                if excess > 0:
                    one_indices = np.argwhere(grid == 1)
                    if len(one_indices) > 0:
                        remove_count = min(excess, len(one_indices))
                        chosen = rng.choice(
                            len(one_indices), size=remove_count, replace=False
                        )
                        for idx in chosen:
                            r, c = one_indices[idx]
                            grid[r, c] = 0.0

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), (
            f'Expected torch.Tensor for key {self.key!r}, got {type(motion)}'
        )

        T = motion.shape[-2]
        D = motion.shape[-1]

        rng = np.random.RandomState()

        # Sample strategy
        strategy_idx = rng.choice(len(self.strategy_names), p=self.strategy_probs)
        strategy = self.strategy_names[strategy_idx]

        # Initialize grid (all zeros = all known)
        grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)

        # Apply strategy
        _STRATEGY_FN[strategy](T, grid, rng)

        # Enforce min/max mask ratio
        self._enforce_mask_ratio(grid, strategy, rng)

        # Expand to full mask (T, TOTAL_DIM)
        src_mask = expand_grid_to_mask(grid)

        # Trim or pad if D != TOTAL_DIM (for robustness)
        if src_mask.shape[-1] < D:
            src_mask = torch.nn.functional.pad(
                src_mask, (0, D - src_mask.shape[-1]), value=1.0
            )
        elif src_mask.shape[-1] > D:
            src_mask = src_mask[..., :D]

        # tgt_length / src_length must be the number of VALID frames (pre-pad),
        # NOT the padded clip length. RandomCropPadding writes `num_frames` =
        # the real content length before right-padding; padded tail frames
        # (replicate of the last frame) must be excluded from loss and
        # attention. Falling back to T keeps backward compatibility when
        # RandomCropPadding is absent (short-clip datasets without padding).
        valid_length = int(results.get('num_frames', T))
        results['src_motion'] = motion.clone()
        results['tgt_motion'] = motion.clone()
        results['src_mask'] = src_mask
        results['tgt_length'] = valid_length
        results['src_length'] = valid_length
        results['mask_strategy'] = strategy  # P0: Task Instruction Modulation

        # Edit-repair mode: with probability edit_repair_prob, apply online corruption
        # to generate LQ motion. In this mode:
        #   src_motion = LQ (corrupted, mask regions have degraded values)
        #   tgt_motion = HQ (original clean motion)
        #   src_mask = from corruptor's joint_corrupted_mask
        #   Trainer will NOT zero mask regions → reactive = LQ values (editing mode)
        # The flag 'edit_mode' tells the trainer to skip the zeroing step.
        results['edit_mode'] = False
        if (self.edit_repair_prob > 0 and self.corruptor_names
                and rng.random() < self.edit_repair_prob):
            motion_path = results.get('motion_path', '')
            if motion_path and os.path.isfile(str(motion_path)):
                try:
                    lq_motion, lq_mask = self._apply_corruption(
                        str(motion_path), motion, T, D, rng
                    )
                    if lq_mask is not None:
                        results['src_motion'] = lq_motion
                        results['src_mask'] = lq_mask
                        results['edit_mode'] = True
                except Exception:
                    pass  # fallback to completion mode

        return results

    def _apply_corruption(
        self, npz_path: str, motion: torch.Tensor, T: int, D: int,
        rng: np.random.RandomState,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply random corruptor to generate LQ motion + mask.

        Returns (lq_motion_tensor, mask_tensor) or (motion, None) on failure.
        """
        import random as _random

        raw = dict(np.load(npz_path, allow_pickle=True))
        if 'transl' in raw and 'trans' not in raw:
            raw['trans'] = raw['transl']

        # Pick random corruptors
        names = [n for n in self.corruptor_names if n in self._get_corruptor_registry()]
        if not names:
            return motion, None

        num = rng.randint(1, min(self.max_corruptions, len(names)) + 1)
        chosen = list(rng.choice(names, size=num, replace=False))

        corrupted = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in raw.items()}
        merged_mask = None
        J = 22

        for name in chosen:
            corruptor = self._get_corruptor(name)
            if corruptor is None:
                continue
            result = corruptor.corrupt(corrupted)
            corrupted = result['corrupted_motion']
            jcm = result.get('joint_corrupted_mask')
            if jcm is not None:
                if jcm.shape[1] > J:
                    jcm = jcm[:, :J]
                if merged_mask is None:
                    merged_mask = np.zeros((T, J), dtype=np.float32)
                min_t = min(merged_mask.shape[0], jcm.shape[0])
                merged_mask[:min_t] = np.maximum(merged_mask[:min_t], jcm[:min_t])

        if merged_mask is None or merged_mask.sum() == 0:
            return motion, None

        # Convert corrupted dict to motion tensor (same transform as LoadSMPLX)
        from motius.datasets.motion.motionhub.transforms.load_smplx import (
            process_smplx_pose, process_transl,
        )
        poses = np.array(corrupted['poses'], dtype=np.float32)
        trans = np.array(corrupted.get('trans', corrupted.get('transl')), dtype=np.float32)
        if trans.ndim == 1:
            trans = trans.reshape(-1, 3)
        rot6d = process_smplx_pose(poses, rot_type='rotation_6d', out_type='smpl_22')
        transl = process_transl(trans, transl_type='abs')
        lq = np.concatenate([transl, rot6d], axis=-1)[:T]
        lq_tensor = torch.from_numpy(lq).float()
        if lq_tensor.shape[0] < T:
            lq_tensor = torch.nn.functional.pad(lq_tensor, (0, 0, 0, T - lq_tensor.shape[0]))

        # If target dim > 135 (e.g. 198-dim with position channels + global rotation),
        # apply the same transforms that were applied to the original motion.
        if D > 135:
            from motius.datasets.motion.motionhub.transforms.compute_198dim import (
                motion135_to_198, _DEFAULT_BONE_OFFSETS_PATH,
            )
            from motius.datasets.motion.motionhub.transforms.local_to_global import (
                LocalToGlobalRotation,
            )
            if os.path.isfile(_DEFAULT_BONE_OFFSETS_PATH):
                bone_offsets = torch.load(
                    _DEFAULT_BONE_OFFSETS_PATH, map_location='cpu'
                ).float()
                lq_tensor = motion135_to_198(lq_tensor, bone_offsets)
                # Convert local rotation to global rotation
                lq_tensor = LocalToGlobalRotation._convert(lq_tensor)
            else:
                # Cannot compute 198-dim; fall back to original motion
                return motion, None

        # Convert (T, J) joint mask to (T, D) mask via grid
        grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
        min_t = min(T, merged_mask.shape[0])
        # Map joint_corrupted_mask (T, 22) → grid (T, 23)
        # If any joint is corrupted, also flag translation group
        grid[:min_t, 1:23] = merged_mask[:min_t, :22]
        any_corrupted = merged_mask[:min_t].max(axis=1) > 0
        grid[:min_t, 0][any_corrupted] = 1.0
        mask_tensor = expand_grid_to_mask(grid)
        if mask_tensor.shape[-1] < D:
            mask_tensor = torch.nn.functional.pad(mask_tensor, (0, D - mask_tensor.shape[-1]), value=0.0)
        elif mask_tensor.shape[-1] > D:
            mask_tensor = mask_tensor[..., :D]

        return lq_tensor, mask_tensor

    def _get_corruptor_registry(self) -> Dict[str, type]:
        try:
            from motius.utils.data_corruptor import (
                JitterCorruptor, JointJumpCorruptor, SlidingCorruptor,
                LimbCandyWrapperCorruptor, WristCandyWrapperCorruptor,
            )
            return {
                'jitter': JitterCorruptor,
                'joint_jump': JointJumpCorruptor,
                'sliding': SlidingCorruptor,
                'limb_candy_wrapper': LimbCandyWrapperCorruptor,
                'wrist_candy_wrapper': WristCandyWrapperCorruptor,
            }
        except ImportError:
            return {}

    def _get_corruptor(self, name: str):
        if name in self._corruptor_cache:
            return self._corruptor_cache[name]
        registry = self._get_corruptor_registry()
        cls = registry.get(name)
        if cls is None:
            return None
        try:
            obj = cls()
            self._corruptor_cache[name] = obj
            return obj
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Unit tests — run with: python -m motius.datasets.motion.motionhub.transforms.universal_mask
# ---------------------------------------------------------------------------

def _test_expand_grid_to_mask():
    """Verify expand_grid_to_mask produces correct joint-group granularity."""
    import numpy as np

    T = 10
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    # Mask translation (col 0) at frame 0
    grid[0, 0] = 1.0
    # Mask joint 5 (col 6) at frame 3
    grid[3, 6] = 1.0

    mask = expand_grid_to_mask(grid)
    assert mask.shape == (T, TOTAL_DIM), f"Expected ({T}, {TOTAL_DIM}), got {mask.shape}"

    # Translation group: dims 0:3 must be all-or-nothing
    assert mask[0, 0] == 1.0 and mask[0, 1] == 1.0 and mask[0, 2] == 1.0, \
        "Translation dims 0:3 should all be 1 when grid col 0 = 1"
    assert mask[1, 0] == 0.0, "Frame 1 translation should be 0"

    # Joint group: 6 dims must be all-or-nothing
    j5_start = TRANSL_DIM + 5 * JOINT_ROT_DIM  # 3 + 5*6 = 33
    for d in range(j5_start, j5_start + JOINT_ROT_DIM):
        assert mask[3, d] == 1.0, f"Joint 5 dim {d} at frame 3 should be 1"
    assert mask[3, j5_start - 1] == 0.0, "Dim before joint 5 group should be 0"
    assert mask[3, j5_start + JOINT_ROT_DIM] == 0.0, "Dim after joint 5 group should be 0"

    # No partial joint masking: within any joint group, all 6 dims identical
    for t in range(T):
        # Check translation group
        trans_vals = mask[t, :TRANSL_DIM].unique()
        assert len(trans_vals) == 1, f"Frame {t}: translation dims not uniform: {trans_vals}"
        # Check each joint group
        for j in range(NUM_JOINT_GROUPS - 1):
            start = TRANSL_DIM + j * JOINT_ROT_DIM
            joint_vals = mask[t, start:start + JOINT_ROT_DIM].unique()
            assert len(joint_vals) == 1, f"Frame {t} joint {j}: dims not uniform: {joint_vals}"

    print("  ✅ expand_grid_to_mask: joint-group granularity correct")


def _test_m1_random_cell_mask_ratio():
    """Verify M1 random cell produces masks in expected ratio range."""
    import numpy as np

    T = 200
    rng = np.random.RandomState(0)
    grid = np.zeros((T, NUM_JOINT_GROUPS), dtype=np.float32)
    m1_random_cell(T, grid, rng)

    ratio = grid.sum() / grid.size
    assert 0.01 <= ratio <= 0.95, f"M1 ratio {ratio:.3f} outside [0.01, 0.95]"
    assert set(np.unique(grid)) == {0.0, 1.0}, "Grid should be binary"
    print(f"  ✅ m1_random_cell: ratio={ratio:.3f} in [0.01, 0.95]")


def _test_prepare_m2m_universal_mask():
    """Verify PrepareM2MUniversalMask output format and invariants."""
    import torch

    T, D = 100, TOTAL_DIM
    motion = torch.randn(T, D)
    results = {'motion': motion, 'num_frames': T}

    transform = PrepareM2MUniversalMask(key='motion')
    out = transform(results)

    assert 'src_motion' in out, "Missing src_motion"
    assert 'tgt_motion' in out, "Missing tgt_motion"
    assert 'src_mask' in out, "Missing src_mask"

    src = out['src_motion']
    tgt = out['tgt_motion']
    mask = out['src_mask']

    # src_motion and tgt_motion must be identical (completion task, no corruption)
    assert torch.allclose(src, tgt), "src_motion and tgt_motion must be identical"
    # src_motion must be the FULL motion (NOT zeroed in mask regions)
    # The zeroing happens in the trainer AFTER normalization, not here
    assert torch.allclose(src, motion[:T] if T <= motion.shape[0] else src), \
        "src_motion should be full motion (zeroing happens in trainer)"
    # mask must be binary
    assert set(mask.unique().tolist()) <= {0.0, 1.0}, f"Mask not binary: {mask.unique()}"
    # mask must have joint-group granularity
    for t in range(min(T, mask.shape[0])):
        trans_vals = mask[t, :TRANSL_DIM].unique()
        assert len(trans_vals) == 1, f"Frame {t}: translation dims not uniform"
        for j in range(NUM_JOINT_GROUPS - 1):
            start = TRANSL_DIM + j * JOINT_ROT_DIM
            joint_vals = mask[t, start:start + JOINT_ROT_DIM].unique()
            assert len(joint_vals) == 1, f"Frame {t} joint {j}: dims not uniform"

    print(f"  ✅ PrepareM2MUniversalMask: outputs correct, mask joint-group granularity verified")


def _test_vace_semantics():
    """Verify VACE inactive/reactive/mask semantics for completion task.

    Expected behavior (completion/inpainting):
    - After normalize, src_motion[mask=1] is zeroed in trainer
    - inactive = zeroed_src * (1-mask) = motion in keep regions, 0 in mask regions
    - reactive = zeroed_src * mask = 0 everywhere (completion: no prior info)
    - Model input = [x_t, inactive, reactive, mask]
    """
    import torch

    T, D = 50, TOTAL_DIM
    motion_norm = torch.randn(T, D)  # simulated normalized motion
    mask = torch.zeros(T, D)
    mask[10:20, :] = 1.0  # frames 10-19 fully masked

    # Simulate trainer's zeroing step
    src_motion_zeroed = motion_norm * (1 - mask)

    # VACE construction
    inactive = src_motion_zeroed * (1 - mask)
    reactive = src_motion_zeroed * mask

    # Checks
    mask_bool = mask > 0.5
    assert inactive[mask_bool].abs().max() == 0.0, "inactive must be 0 in mask=1 regions"
    assert reactive.abs().max() == 0.0, "reactive must be 0 everywhere for completion task"
    assert inactive[~mask_bool].std() > 0, "inactive must have motion values in mask=0 regions"

    print("  ✅ VACE semantics (completion): inactive correct, reactive=0 everywhere")


if __name__ == '__main__':
    print("Running universal_mask unit tests...")
    _test_expand_grid_to_mask()
    _test_m1_random_cell_mask_ratio()
    _test_prepare_m2m_universal_mask()
    _test_vace_semantics()
    print("All tests passed ✅")
