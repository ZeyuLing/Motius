"""Canonical motion-repair mask utilities (shared by the pipeline and eval).

This module is the single source of truth for the adaptive-mask post-processing
used by HyMotion-M2M motion repair. Both
:meth:`motius.pipelines.motioncanvas.MotionCanvasPipeline.infer_repair`
and the offline eval (``scripts/eval/eval_m2m_v2_all_tasks.py``) import from here,
so there is exactly one place that defines "how a raw defect mask is tightened".

All functions are pure numpy and operate on the 135/198-dim motion layout:

    dim 0:3     translation XYZ        (pelvis / joint 0 group)
    dim 3:135   rot6d, 22 joints * 6   (joints 0..21)
    dim 135:198 pos, 21 joints * 3     (joints 1..21, 198-dim only)

Mask convention everywhere: ``1 = generate/regenerate``, ``0 = keep (lock to LQ)``.
"""

from __future__ import annotations

from typing import List, Optional, Set

import numpy as np

# SMPL-22 kinematic parents (─1 = root). Used for the kinematic-neighbour
# spatial dilation of the defect mask.
SMPL22_PARENTS: List[int] = [
    -1,  # 0  pelvis
    0,   # 1  l_hip
    0,   # 2  r_hip
    0,   # 3  spine1
    1,   # 4  l_knee
    2,   # 5  r_knee
    3,   # 6  spine2
    4,   # 7  l_ankle
    5,   # 8  r_ankle
    6,   # 9  spine3
    7,   # 10 l_foot
    8,   # 11 r_foot
    9,   # 12 neck
    9,   # 13 l_collar
    9,   # 14 r_collar
    12,  # 15 head
    13,  # 16 l_shoulder
    14,  # 17 r_shoulder
    16,  # 18 l_elbow
    17,  # 19 r_elbow
    18,  # 20 l_wrist
    19,  # 21 r_wrist
]

# Upper-chain small joints excluded from BOTH emitting and receiving spatial
# propagation. The change detector often fires on the head from small pose
# noise; kinematic propagation would then drag the neck along and the model
# invents a whole new head/neck rotation not present in the LQ input.
DEFAULT_NO_PROPAGATE: Set[int] = {12, 13, 14, 15, 20, 21}  # neck, collars, head, wrists


def motion_135_to_198(
    motion_135: np.ndarray,           # (T, 135) local transl(3) + rot6d(132)
    bone_offsets: np.ndarray,         # (22, 3) SMPL-22 bone offsets
) -> np.ndarray:
    """Expand 135-dim motion to the 198-dim v2 representation by appending the
    63 FK joint-position channels (21 joints * 3, joints 1..21, no pelvis).

    Positions are pelvis-relative in XYZ, matching both
    ``Compute198DimPosition`` in the training pipeline and
    ``scripts/eval/eval_m2m_v2_all_tasks.motion_135_to_198``.  Keeping absolute
    Y here would add the pelvis height (roughly one metre) to every position-Y
    channel and send repair conditioning far outside the training statistics.
    """
    from motius.evaluation.motion.m2m_eval_metrics import (
        motion135_to_positions_np,
    )
    positions = motion135_to_positions_np(
        motion_135.astype(np.float32), bone_offsets.astype(np.float32))  # (T,22,3)
    T = positions.shape[0]
    pelvis = positions[:, 0:1, :]                   # (T,1,3)
    joint_pos = positions[:, 1:, :] - pelvis        # (T,21,3), XYZ-relative
    pos_flat = joint_pos.reshape(T, 63)
    return np.concatenate(
        [motion_135.astype(np.float32), pos_flat.astype(np.float32)], axis=-1)


def smpl22_neighbors() -> List[List[int]]:
    """Symmetric kinematic neighbourhood (parent AND children) per joint."""
    parents = SMPL22_PARENTS
    children: List[List[int]] = [[] for _ in range(len(parents))]
    for child, parent in enumerate(parents):
        if parent >= 0:
            children[parent].append(child)
    neigh: List[List[int]] = []
    for j in range(len(parents)):
        nbs = set()
        if parents[j] >= 0:
            nbs.add(parents[j])
        nbs.update(children[j])
        neigh.append(sorted(nbs))
    return neigh


def compute_strict_adaptive_mask(
    adaptive_raw: np.ndarray,         # (T, D) raw mask, 1=generate
    dilate: int = 2,                  # temporal dilation radius (frames)
    min_blob: int = 3,                # minimum temporal run (frames) to keep
    motion_dim: int = 135,
    lock_trans: bool = False,         # if True, never mask translation (M7 conv)
    no_propagate: Optional[Set[int]] = None,
) -> np.ndarray:
    """Tighten a raw defect mask into a reliable "definitely defective" mask.

    Steps (all in the ``1=generate`` convention):

    1. Per-joint aggregation: a pose joint at frame t is flagged iff ANY of
       its rotation/position channels in the raw mask are flagged (raw mask
       already has per-dim dropout, so OR de-noises sensibly). Root
       translation is kept as a separate flag and never enters this pose
       aggregation.
    2. Kinematic spatial dilation to parent+children joints (a bad joint
       usually drags its neighbours). Upper-chain small joints
       (``no_propagate``) neither emit nor receive propagation.
    3. Temporal dilation by ``±dilate`` frames.
    4. Blob filter: drop per-joint temporal runs shorter than ``min_blob``
       frames (isolated single-frame flags are noise).
    5. Map back to ``(T, D)`` with per-joint-group broadcasting.

    ``lock_trans`` zeroes the translation columns at the end (M7 convention:
    translation is never masked, so the global trajectory stays locked to LQ).
    When it is disabled, the raw translation flag is copied independently:
    neither translation defects nor pose defects cross-contaminate the other
    stream through kinematic spatial dilation.
    """
    if no_propagate is None:
        no_propagate = DEFAULT_NO_PROPAGATE
    T, D = adaptive_raw.shape

    # --- Step 1: per-joint aggregation to (T, 22) bool ----------------
    joint_flag = np.zeros((T, 22), dtype=bool)
    translation_flag = (adaptive_raw[:, :3] >= 0.5).any(axis=-1)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        joint_flag[:, j] |= (adaptive_raw[:, s:e] >= 0.5).any(axis=-1)
    if D >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            joint_flag[:, j] |= (adaptive_raw[:, ps:pe] >= 0.5).any(axis=-1)

    # --- Step 2: kinematic spatial dilation ---------------------------
    neigh = smpl22_neighbors()
    joint_flag_sp = joint_flag.copy()
    for j in range(22):
        if j in no_propagate:
            continue
        for nb in neigh[j]:
            if nb in no_propagate:
                continue
            joint_flag_sp[:, nb] |= joint_flag[:, j]
    joint_flag = joint_flag_sp

    # --- Step 3: temporal dilation ------------------------------------
    if dilate > 0:
        jf = joint_flag.copy()
        for s in range(1, dilate + 1):
            jf[s:] |= joint_flag[:-s]
            jf[:-s] |= joint_flag[s:]
        joint_flag = jf

    # --- Step 4: blob filter ------------------------------------------
    if min_blob > 1:
        for j in range(22):
            col = joint_flag[:, j]
            if not col.any():
                continue
            i = 0
            while i < T:
                if col[i]:
                    k = i
                    while k < T and col[k]:
                        k += 1
                    if (k - i) < min_blob:
                        col[i:k] = False
                    i = k
                else:
                    i += 1
            joint_flag[:, j] = col

    # --- Step 5: map back to (T, D) -----------------------------------
    out_mask = np.zeros((T, D), dtype=np.float32)
    out_mask[:, :3] = translation_flag[:, None].astype(np.float32)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        out_mask[:, s:e] = joint_flag[:, j:j + 1].astype(np.float32)
    if D >= 198:
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            out_mask[:, ps:pe] = joint_flag[:, j:j + 1].astype(np.float32)

    if lock_trans:
        out_mask[:, :3] = 0.0

    return out_mask


def compute_ada_keep_mask(
    motion_norm_lq: np.ndarray,       # (T, D) normalized LQ
    denoised_stage1: np.ndarray,      # (T, D) normalized stage-1 projection
    threshold_mode: str = 'abs',      # 'abs' or 'topk_pct'
    threshold: float = 0.1,           # abs threshold OR top-k fraction
) -> np.ndarray:
    """Self-detection (ada_denoise Stage-2): build a defect mask from the
    model's own change pattern.

        change      = |motion_lq - denoised_stage1|     (normalized space)
        high_change = change > threshold                → "this cell is defective"

    Per-joint aggregation: a joint is "clean at frame t" iff ALL of its
    channels (rot6d 6 [+ pos 3]) are low-change; otherwise it is flagged.
    Translation (dims 0:3) is treated as a single group.

    Returns ``(T, D)`` mask with 1=generate, 0=keep.
    """
    D = motion_norm_lq.shape[-1]
    change = np.abs(motion_norm_lq - denoised_stage1)

    if threshold_mode == 'abs':
        thr = float(threshold)
    elif threshold_mode == 'topk_pct':
        thr = float(np.quantile(change.ravel(), 1.0 - float(threshold)))
    else:
        raise ValueError(f'Unknown threshold_mode: {threshold_mode!r}')

    low_change_chan = (change <= thr)  # True = clean

    out_mask = np.ones((motion_norm_lq.shape[0], D), dtype=np.float32)

    trans_clean = low_change_chan[:, :3].all(axis=-1)
    out_mask[trans_clean, :3] = 0.0

    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        j_clean = low_change_chan[:, s:e].all(axis=-1)
        out_mask[j_clean, s:e] = 0.0

    if D >= 198:
        for j in range(1, 22):
            rot_s, rot_e = 3 + j * 6, 3 + (j + 1) * 6
            j_clean_rot = low_change_chan[:, rot_s:rot_e].all(axis=-1)
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            j_clean_pos = low_change_chan[:, ps:pe].all(axis=-1)
            j_clean = j_clean_rot & j_clean_pos
            out_mask[j_clean, ps:pe] = 0.0
            out_mask[j_clean, rot_s:rot_e] = 0.0

    return out_mask


def joint_mask_to_dim_mask(
    joint_flag: np.ndarray,           # (T, 22) bool, 1=defective
    motion_dim: int = 135,
    translation_mode: str = 'lock',   # 'lock' | 'detected' | 'all'
    translation_flag: Optional[np.ndarray] = None,  # (T,), independent root defect
    position_flag: Optional[np.ndarray] = None,  # (T,22|21), independent position defect
    position_mask_mode: str = 'fk_descendant',  # 'fk_descendant'|'joint_coupled'
    valid_len: Optional[int] = None,  # frames [valid_len:] forced to keep (0)
) -> np.ndarray:
    """Expand a per-joint defect flag to a ``(T, D)`` generate-mask, applying
    the translation policy.

    For the 198-D representation, ``position_flag`` is authoritative when
    supplied. Otherwise ``position_mask_mode='fk_descendant'`` derives the
    physically affected strict-descendant support, while ``'joint_coupled'``
    gives every non-root joint the same rotation/position decision. The latter
    is the BrokenAMASS-v5 repair contract and matches a joint-coupled detector.

    translation_mode:
      - 'lock'     : translation never regenerated (cols 0:3 = 0).
      - 'detected' : translation regenerated from ``translation_flag`` when
                     supplied; the pelvis joint flag is only a backward-
                     compatible fallback.  Root rotation and translation are
                     deliberately independent defect channels.
      - 'all'      : translation regenerated on every valid frame.
    """
    T = joint_flag.shape[0]
    out = np.zeros((T, motion_dim), dtype=np.float32)
    for j in range(22):
        s, e = 3 + j * 6, 3 + (j + 1) * 6
        out[:, s:e] = joint_flag[:, j:j + 1].astype(np.float32)
    if motion_dim >= 198:
        if position_flag is not None:
            resolved_position = np.asarray(position_flag, dtype=bool)
            if resolved_position.shape == (T, 22):
                resolved_position = resolved_position[:, 1:]
            elif resolved_position.shape != (T, 21):
                raise ValueError(
                    'position_flag must have shape '
                    f'{(T, 22)} or {(T, 21)}, got {resolved_position.shape}'
                )
        elif position_mask_mode == 'fk_descendant':
            resolved_position = joint_rotation_mask_to_fk_position_mask(
                joint_flag
            )
        elif position_mask_mode == 'joint_coupled':
            resolved_position = np.asarray(joint_flag[:, 1:], dtype=bool)
        else:
            raise ValueError(
                'position_mask_mode must be fk_descendant|joint_coupled, '
                f'got {position_mask_mode!r}'
            )
        for j in range(1, 22):
            ps, pe = 135 + (j - 1) * 3, 135 + j * 3
            out[:, ps:pe] = resolved_position[
                :, j - 1:j
            ].astype(np.float32)

    if translation_mode == 'lock':
        out[:, :3] = 0.0
    elif translation_mode == 'detected':
        if translation_flag is None:
            translation_flag = joint_flag[:, 0]
        translation_flag = np.asarray(translation_flag, dtype=bool).reshape(-1)
        if len(translation_flag) != T:
            raise ValueError(
                f'translation_flag length {len(translation_flag)} != mask length {T}'
            )
        out[:, :3] = translation_flag[:, None].astype(np.float32)
    elif translation_mode == 'all':
        out[:, :3] = 1.0
    else:
        raise ValueError(f'Unknown translation_mode: {translation_mode!r}')

    if valid_len is not None and valid_len < T:
        out[valid_len:] = 0.0
    return out


def joint_rotation_mask_to_fk_position_mask(
    joint_flag: np.ndarray,
) -> np.ndarray:
    """Map SMPL-22 local-rotation defects to affected FK positions.

    Returns ``(T, 21)`` flags for the appended positions of joints 1..21.
    Position ``d`` is generated iff at least one *strict ancestor* of ``d``
    has a generated local rotation.  In particular, root rotation affects all
    21 positions, a leaf rotation affects none, and a joint never masks its
    own position merely because its own rotation is masked.
    """
    rotation_flag = np.asarray(joint_flag, dtype=bool)
    if rotation_flag.ndim != 2 or rotation_flag.shape[1] != 22:
        raise ValueError(
            f'joint_flag must have shape (T, 22), got {rotation_flag.shape}'
        )

    position_flag = np.zeros((rotation_flag.shape[0], 21), dtype=bool)
    for descendant in range(1, 22):
        ancestor = SMPL22_PARENTS[descendant]
        while ancestor >= 0:
            position_flag[:, descendant - 1] |= rotation_flag[:, ancestor]
            ancestor = SMPL22_PARENTS[ancestor]
    return position_flag


def global_rotation_mask_to_local_rotation_mask(
    joint_flag: np.ndarray,
) -> np.ndarray:
    """Conservatively map global-rotation support to local rotations.

    For a non-root joint ``j``, ``L_j = G_parent(j)^T G_j``.  Therefore a
    local rotation is unknown whenever either endpoint of that relative
    rotation is unknown.  Equivalently, one flagged global rotation activates
    the same joint and each of its direct children, but not all descendants.

    This Boolean mapping is a conservative fallback.  When the continuous
    input/probe global rotations are available, converting both through IK and
    thresholding their local-rotation residual is tighter because adjacent
    global changes may cancel in the relative rotation.
    """
    global_flag = np.asarray(joint_flag, dtype=bool)
    if global_flag.ndim != 2 or global_flag.shape[1] != 22:
        raise ValueError(
            f'joint_flag must have shape (T, 22), got {global_flag.shape}'
        )
    local_flag = global_flag.copy()
    for joint in range(1, 22):
        local_flag[:, joint] |= global_flag[
            :, SMPL22_PARENTS[joint]
        ]
    return local_flag


def adapt_morediff_201d_mask_to_hymotion198(
    channel_mask: np.ndarray,
    *,
    local_pose_channel_mask: np.ndarray | None = None,
    position_rotation_support: str = "none",
) -> dict[str, np.ndarray]:
    """Adapt a MoreDiff/Occam 201-D repair mask to OURS' local 198-D layout.

    MoreDiff orders channels as ``pose6d[22] + pelvis-relative position[22]
    + translation[3]``.  HyMotion-M2M orders them as ``translation[3] +
    local-pose6d[22] + FK-position[1:22]``.  Copying the first 198 MoreDiff
    cells is therefore invalid: it interprets root-pose cells as translation
    and shifts every remaining semantic group.

    MoreDiff pose channels are **global** rotations because
    ``OccamMotionRep(global_pose=True)`` applies FK before encoding.  OURS pose
    channels are local rotations.  A direct per-joint copy is invalid:
    ``L_j = G_parent(j)^T G_j``.  When an exact local residual mask is supplied
    by the MoreDiff probe, it is used directly.  Otherwise the adapter uses
    the conservative Boolean closure ``local[j] = global[j] | global[parent]``.

    The adapter also closes OURS' local-rotation/FK-position dependency:

    * a measured position defect remains position-only by default; optional
      ``parent``/``ancestors`` modes provide an explicitly conservative
      position-to-rotation expansion for ablations;
    * every FK-position channel affected by a regenerated local rotation is
      made unknown, so the model is never conditioned on a position that
      conflicts with a generated rotation;
    * translation stays an independent per-axis decision and never enters the
      rotation closure.

    MoreDiff's redundant pelvis-relative position for joint 0 is discarded.
    Rotation and position vector components are grouped per physical joint:
    one flagged component regenerates all six rotation or all three position
    components of that group.

    Args:
        channel_mask: Boolean-like array with shape ``(T, 201)`` and
            ``True/1 = regenerate``.
        local_pose_channel_mask: Optional exact local-pose residual mask with
            shape ``(T,22,6)`` or ``(T,132)``.  This must come from applying IK
            to both the normalized MoreDiff input and denoise probe before
            thresholding.
        position_rotation_support: ``"none"`` (default), ``"parent"``, or
            ``"ancestors"``.

    Returns:
        A dictionary containing the exact 198-D generate mask plus separate
        rotation, position, effective-joint, and translation-axis supports.
    """
    mask = np.asarray(channel_mask, dtype=bool)
    if mask.ndim != 2 or mask.shape[1] != 201:
        raise ValueError(
            f"MoreDiff channel_mask must have shape (T, 201), got {mask.shape}"
        )
    if position_rotation_support not in {"none", "parent", "ancestors"}:
        raise ValueError(
            "position_rotation_support must be 'none', 'parent', or "
            "'ancestors', got "
            f"{position_rotation_support!r}"
        )

    frame_count = mask.shape[0]
    raw_global_pose = mask[:, :132].reshape(
        frame_count, 22, 6
    ).any(axis=-1)
    if local_pose_channel_mask is None:
        raw_local_pose = global_rotation_mask_to_local_rotation_mask(
            raw_global_pose
        )
        local_pose_source = "global_boolean_parent_endpoint_closure"
    else:
        local_channels = np.asarray(local_pose_channel_mask, dtype=bool)
        if local_channels.shape == (frame_count, 132):
            local_channels = local_channels.reshape(frame_count, 22, 6)
        if local_channels.shape != (frame_count, 22, 6):
            raise ValueError(
                "local_pose_channel_mask must have shape (T,22,6) or "
                f"(T,132), got {local_channels.shape}"
            )
        raw_local_pose = local_channels.any(axis=-1)
        local_pose_source = "ik_local_residual"
    raw_position = mask[:, 132:198].reshape(
        frame_count, 22, 3
    ).any(axis=-1)
    # pelvis-relative pelvis position is identically zero and has no 198-D
    # counterpart.  It must not be reinterpreted as root rotation/translation.
    raw_position[:, 0] = False
    translation_axis = mask[:, 198:201].copy()

    rotation = raw_local_pose.copy()
    if position_rotation_support != "none":
        for joint in range(1, 22):
            parent = SMPL22_PARENTS[joint]
            rotation[:, parent] |= raw_position[:, joint]
            if position_rotation_support == "ancestors":
                while parent > 0:
                    parent = SMPL22_PARENTS[parent]
                    rotation[:, parent] |= raw_position[:, joint]

    position = np.zeros((frame_count, 22), dtype=bool)
    position[:, 1:] = raw_position[:, 1:]
    position[:, 1:] |= joint_rotation_mask_to_fk_position_mask(rotation)
    # This root cell is descriptive/visual only. HyMotion-198 has no root FK
    # position channel; actual root generation is governed by translation_axis.
    position[:, 0] = translation_axis.any(axis=-1)

    mask_198 = np.zeros((frame_count, 198), dtype=bool)
    mask_198[:, :3] = translation_axis
    mask_198[:, 3:135] = np.repeat(
        rotation[:, :, None], 6, axis=-1
    ).reshape(frame_count, 132)
    mask_198[:, 135:198] = np.repeat(
        position[:, 1:, None], 3, axis=-1
    ).reshape(frame_count, 63)

    return {
        "mask_198": mask_198,
        "raw_pose_joint_mask": raw_global_pose,
        "raw_global_pose_joint_mask": raw_global_pose,
        "raw_local_pose_joint_mask": raw_local_pose,
        "raw_position_joint_mask": raw_position,
        "rotation_joint_mask": rotation,
        "position_joint_mask": position,
        "effective_joint_mask": rotation | position,
        "translation_axis_mask": translation_axis,
        "translation_mask": translation_axis.any(axis=-1),
        "position_rotation_support": np.asarray(
            position_rotation_support
        ),
        "local_pose_source": np.asarray(local_pose_source),
    }


def expand_motion_mask_135_to_198(mask_135: np.ndarray) -> np.ndarray:
    """Append FK-consistent position support to an existing 135-D mask.

    The original translation and rotation channels are copied exactly.  This
    matters for channel-granularity masks: position support is derived from an
    OR over each local-rotation group, without broadening or otherwise changing
    the six rotation channels themselves.
    """
    mask = np.asarray(mask_135)
    if mask.ndim != 2 or mask.shape[1] != 135:
        raise ValueError(f'mask_135 must have shape (T, 135), got {mask.shape}')

    joint_flag = (mask[:, 3:135].reshape(-1, 22, 6) >= 0.5).any(axis=-1)
    position_flag = joint_rotation_mask_to_fk_position_mask(joint_flag)
    position_mask = np.repeat(position_flag[:, :, None], 3, axis=-1).reshape(-1, 63)
    return np.concatenate(
        [mask.astype(np.float32, copy=True), position_mask.astype(np.float32)],
        axis=-1,
    )


def project_motion135_fixed_support(
    condition_motion: np.ndarray,
    candidate_motion: np.ndarray,
    joint_flag: np.ndarray,
    translation_flag: np.ndarray,
):
    """Restore every unmasked 135-D condition cell after FPS conversion.

    ``condition_motion`` and ``candidate_motion`` use the canonical raw layout
    ``translation(3) + 22 * rot6d(6)``. The projection is intentionally done
    in this representation, before conversion back to SMPL axis-angle, so an
    unmasked rotation is copied as one complete 6-D joint group.

    Returns ``(projected_motion, stats)``. ``stats`` reports leakage before and
    after projection separately for translation and rotation support and is
    suitable for persisting in evaluation artifacts.
    """
    condition = np.asarray(condition_motion)
    candidate = np.asarray(candidate_motion)
    joints = np.asarray(joint_flag, dtype=bool)
    translation = np.asarray(translation_flag, dtype=bool).reshape(-1)

    if condition.shape != candidate.shape:
        raise ValueError(
            f"condition/candidate shape mismatch: {condition.shape} vs {candidate.shape}"
        )
    if condition.ndim != 2 or condition.shape[1] != 135:
        raise ValueError(f"motion must have shape (T, 135), got {condition.shape}")
    if joints.shape != (len(condition), 22):
        raise ValueError(
            f"joint_flag must have shape {(len(condition), 22)}, got {joints.shape}"
        )
    if len(translation) != len(condition):
        raise ValueError(
            f"translation_flag length {len(translation)} != motion length {len(condition)}"
        )

    generate_mask = joint_mask_to_dim_mask(
        joints,
        motion_dim=135,
        translation_mode="detected",
        translation_flag=translation,
        valid_len=len(condition),
    ).astype(bool)
    known_mask = ~generate_mask
    projected = np.where(known_mask, condition, candidate)

    pre_delta = np.abs(candidate - condition)
    post_delta = np.abs(projected - condition)

    def _support_stats(mask: np.ndarray, prefix: str) -> dict:
        values_pre = pre_delta[mask]
        values_post = post_delta[mask]
        return {
            f"{prefix}_dim_count": int(mask.sum()),
            f"{prefix}_pre_projection_max_abs": (
                float(values_pre.max()) if values_pre.size else 0.0
            ),
            f"{prefix}_pre_projection_mean_abs": (
                float(values_pre.mean()) if values_pre.size else 0.0
            ),
            f"{prefix}_post_projection_max_abs": (
                float(values_post.max()) if values_post.size else 0.0
            ),
            f"{prefix}_post_projection_mean_abs": (
                float(values_post.mean()) if values_post.size else 0.0
            ),
        }

    known_translation = known_mask.copy()
    known_translation[:, 3:] = False
    known_rotation = known_mask.copy()
    known_rotation[:, :3] = False
    stats = {
        **_support_stats(known_mask, "known"),
        **_support_stats(known_translation, "known_translation"),
        **_support_stats(known_rotation, "known_rotation"),
        "generated_dim_count": int(generate_mask.sum()),
    }
    return projected.astype(candidate.dtype, copy=False), stats
