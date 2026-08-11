"""Load real source motion for editing pairs (e.g. PerMo Neutral->Emotion).

This transform replaces `src_motion` from PrepareM2MCondition with a real
source motion loaded from disk. It is
designed for editing datasets where source and target are different
recordings of the same action (e.g. Neutral vs Emotion style).

Pipeline placement: AFTER PrepareM2MCondition.

When `source_motion_path` is present in results:
  1. Load source npz -> 198-dim motion tensor
  2. (Optional) Apply KIMODO root conversion if kimodo_root_cfg is set
  3. Resample to the target's valid length and pad to its tensor length
  4. Override src_motion, preserve the sampled src_mask, set edit_mode=True

When `source_motion_path` is absent:
  Pass through unchanged (regular completion from PrepareM2MCondition).

KIMODO root conversion:
  In the E4 (KIMODO Root) pipeline, SmplTransToKimodoRootOnline converts
  the target motion BEFORE PrepareM2MCondition splits it into src/tgt.
  Since this transform loads source motion AFTER the split, the source
  would remain raw SMPL — creating a representation mismatch. Setting
  ``kimodo_root_cfg`` applies the same ADMM smoothing + reference frame
  adjustment on the loaded source motion.

Example pipeline (E2 / SMPL Root — no KIMODO conversion needed):
    dict(type='LoadCompatibleCaption', allow_none=False),
    dict(type='LoadPreExtractedTextEmbedding', ...),
    dict(type='LoadSmplx55', key='motion', ...),       # loads TARGET motion
    dict(type='Compute198DimPosition', key='motion'),
    dict(type='RandomCropPadding', clip_len=360, ...),
    dict(type='PrepareM2MCondition', ...),
    dict(type='LoadEditingSourceMotion'),               # <-- HERE
    dict(type='PackInputs', ...),

Example pipeline (E4 / KIMODO Root — with KIMODO conversion):
    ...
    dict(type='SmplTransToKimodoRootOnline', key='motion', admm_margin_m=0.06),
    dict(type='RandomCropPadding', clip_len=360, ...),
    dict(type='PrepareM2MCondition', ...),
    dict(type='LoadEditingSourceMotion',                # <-- HERE
         kimodo_root_cfg=dict(admm_margin_m=0.06)),
    dict(type='PackInputs', ...),
"""
from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


def _load_motion_198_from_npz(path: str, bone_offsets: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Load a 198-dim motion tensor from an npz file.

    Supports two formats:
      1. Pre-computed: npz has 'motion_198' key -> use directly
      2. Pre-computed 135: npz has 'motion_135' key -> run FK to get 198
      3. Raw SMPL: npz has 'poses' + 'trans' -> convert to 135 -> 198

    Returns:
        (T, 198) float tensor.
    """
    data = np.load(path, allow_pickle=True)

    # Format 1: pre-computed 198-dim (e.g. PerMo)
    if 'motion_198' in data:
        return torch.from_numpy(np.asarray(data['motion_198'], dtype=np.float32))

    # Format 2: pre-computed 135-dim -> needs FK
    if 'motion_135' in data and 'poses' not in data:
        motion_135 = torch.from_numpy(np.asarray(data['motion_135'], dtype=np.float32))
        if bone_offsets is not None:
            from motius.datasets.motion.motionhub.transforms.compute_198dim import motion135_to_198
            return motion135_to_198(motion_135, bone_offsets)
        else:
            # Fallback: zero-pad position channels
            return torch.nn.functional.pad(motion_135, (0, 63))

    # Format 3: raw SMPL params
    if 'poses' in data and ('trans' in data or 'transl' in data):
        from motius.datasets.motion.motionhub.transforms.load_smplx import (
            process_smplx_pose, process_transl,
        )
        poses = np.asarray(data['poses'], dtype=np.float32)
        trans_key = 'trans' if 'trans' in data else 'transl'
        trans = np.asarray(data[trans_key], dtype=np.float32)
        if trans.ndim == 1:
            trans = trans.reshape(-1, 3)

        rot6d = process_smplx_pose(poses, rot_type='rotation_6d', out_type='smpl_22')
        transl = process_transl(trans, transl_type='abs')
        motion_135 = torch.from_numpy(
            np.concatenate([transl, rot6d], axis=-1).astype(np.float32)
        )
        if bone_offsets is not None:
            from motius.datasets.motion.motionhub.transforms.compute_198dim import motion135_to_198
            return motion135_to_198(motion_135, bone_offsets)
        else:
            return torch.nn.functional.pad(motion_135, (0, 63))

    raise ValueError(
        f"Cannot load motion from {path}: expected 'motion_198', 'motion_135', "
        f"or 'poses'+'trans' keys, got {list(data.keys())}"
    )


def _resample_motion_to_length(motion: torch.Tensor, target_len: int) -> torch.Tensor:
    """Resample paired edit context to the target's normalized timeline."""
    if motion.ndim != 2 or motion.shape[-1] != 198:
        raise ValueError(f"Expected motion with shape (T, 198), got {tuple(motion.shape)}")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if motion.shape[0] == target_len:
        return motion
    if motion.shape[0] == 0:
        raise ValueError("Cannot resample an empty source motion")

    resampled = F.interpolate(
        motion.transpose(0, 1).unsqueeze(0),
        size=target_len,
        mode='linear',
        align_corners=True,
    ).squeeze(0).transpose(0, 1)

    # Project interpolated HYMotion row-major 6D rotations back onto SO(3).
    # The representation is the row-wise flattening of the first two matrix
    # columns: [R00, R01, R10, R11, R20, R21].  The generic geometry helpers
    # split a 6D vector into its first/last three values and therefore decode a
    # different layout; using them here corrupts every resampled edit source.
    from motius.motion.skeleton.fk import (
        rot6d_to_rotmat_row_major,
        rotmat_to_rot6d_row_major,
    )
    rot6d = resampled[:, 3:135].reshape(target_len, 22, 6)
    resampled[:, 3:135] = rotmat_to_rot6d_row_major(
        rot6d_to_rotmat_row_major(rot6d)
    ).reshape(target_len, 132)
    return resampled


def _pad_motion_to_length(motion: torch.Tensor, padded_len: int) -> torch.Tensor:
    """Replicate-pad valid source frames to the model's fixed tensor length."""
    if motion.shape[0] > padded_len:
        raise ValueError(
            f"Valid source length {motion.shape[0]} exceeds padded length {padded_len}"
        )
    if motion.shape[0] == padded_len:
        return motion
    tail = motion[-1:].expand(padded_len - motion.shape[0], -1)
    return torch.cat([motion, tail], dim=0)


@TRANSFORMS.register_module()
class LoadEditingSourceMotion(BaseTransform):
    """Load real source motion for editing pairs.

    When ``source_motion_path`` exists in results, loads it as the
    ``src_motion`` for editing training, replacing the copied target motion
    created by ``PrepareM2MCondition``.

    The source and target motions can have different durations (e.g. Neutral
    vs Emotion style of the same action). The source is resampled onto the
    target's normalized timeline instead of being truncated or tail-padded.

    Parameters
    ----------
    source_path_key : str
        Key in results dict for the source motion file path.
    bone_offsets_path : str or None
        Path to bone offsets .pt for FK computation (135->198).
        If None, uses default path. Only needed when source npz lacks
        pre-computed motion_198.
    kimodo_root_cfg : dict or None
        If set, apply KIMODO Root conversion (ADMM smoothing) on the
        loaded source motion. This is REQUIRED when the pipeline uses
        ``SmplTransToKimodoRootOnline`` on the target motion — otherwise
        the target is KIMODO-converted but the source stays raw SMPL,
        creating a representation mismatch.

        Example: ``kimodo_root_cfg=dict(admm_margin_m=0.06)``

        Keys:
          - admm_margin_m (float): Max frame-to-frame XZ displacement
            in meters (default 0.06). Should match the value used in
            ``SmplTransToKimodoRootOnline`` for the target motion.
    require_source : bool
        Return ``None`` when the source path is absent or unreadable. Editing
        datasets should enable this together with dataset refetch so a broken
        pair can never silently degrade into a completion sample.
    """

    def __init__(
        self,
        source_path_key: str = 'source_motion_path',
        bone_offsets_path: Optional[str] = None,
        kimodo_root_cfg: Optional[dict] = None,
        require_source: bool = False,
    ):
        self.source_path_key = source_path_key
        self._bone_offsets_path = bone_offsets_path
        self._bone_offsets: Optional[torch.Tensor] = None
        self.kimodo_root_cfg = kimodo_root_cfg
        self.require_source = bool(require_source)

    def _get_bone_offsets(self) -> Optional[torch.Tensor]:
        """Lazy-load bone offsets for FK computation."""
        if self._bone_offsets is not None:
            return self._bone_offsets

        import os.path as osp
        path = self._bone_offsets_path
        if path is None:
            path = osp.join(
                osp.dirname(osp.dirname(osp.dirname(osp.dirname(
                    osp.dirname(osp.dirname(__file__)))))),
                'data', 'hymotion_m2m_data', 'bone_offsets_22.pt',
            )
        if osp.isfile(path):
            self._bone_offsets = torch.load(path, map_location='cpu', weights_only=True).float()
        return self._bone_offsets

    def _apply_kimodo_root_conversion(self, motion_198: torch.Tensor) -> torch.Tensor:
        """Apply KIMODO Root conversion (ADMM smoothing) on a (T, 198) tensor.

        Reuses the same logic as SmplTransToKimodoRootOnline._convert_motion_198
        to ensure identical conversion for source and target motions.
        """
        from motius.datasets.motion.motionhub.transforms.smpl_trans_to_kimodo_root import (
            admm_smooth_translation_xz_simple,
        )

        admm_margin_m = self.kimodo_root_cfg.get('admm_margin_m', 0.06)

        raw_trans = motion_198[..., 0:3]       # (T, 3)
        rotation = motion_198[..., 3:135]      # (T, 132)
        pos_rel_raw = motion_198[..., 135:198]  # (T, 63)

        # Step 1: Smooth translation (XZ only, Y raw)
        smooth_trans = admm_smooth_translation_xz_simple(
            raw_trans,
            margin_m=admm_margin_m,
        )  # (T, 3)

        # Step 2: Adjust position reference frame
        # pos_rel_smooth = pos_rel_raw + (raw_trans - smooth_trans).
        # The KIMODO smoother keeps Y raw, so only XZ change in practice.
        trans_diff = raw_trans - smooth_trans  # (T, 3)
        trans_diff_expanded = trans_diff.unsqueeze(-2).expand(-1, 21, -1).reshape(-1, 63)
        pos_rel_smooth = pos_rel_raw + trans_diff_expanded  # (T, 63)

        # Reconstruct 198-dim KIMODO Root motion
        return torch.cat([smooth_trans, rotation, pos_rel_smooth], dim=-1)

    def transform(self, results: Dict) -> Optional[Dict]:
        source_path = results.get(self.source_path_key)

        # No source motion path -> pass through (regular completion/corruption)
        if source_path is None or not os.path.isfile(str(source_path)):
            return None if self.require_source else results

        # Load source motion as 198-dim tensor
        try:
            src_motion_198 = _load_motion_198_from_npz(
                str(source_path),
                bone_offsets=self._get_bone_offsets(),
            )
        except Exception:
            return None if self.require_source else results

        # Apply KIMODO Root conversion if configured (E4 pipeline)
        if self.kimodo_root_cfg is not None:
            src_motion_198 = self._apply_kimodo_root_conversion(src_motion_198)

        # Match target length
        tgt_motion = results.get('tgt_motion')
        if tgt_motion is None:
            return None if self.require_source else results

        padded_target_len = int(tgt_motion.shape[0])
        valid_target_len = int(results.get('tgt_length', padded_target_len))
        valid_target_len = max(1, min(valid_target_len, padded_target_len))
        src_motion_198 = _resample_motion_to_length(
            src_motion_198, valid_target_len
        )
        src_motion_198 = _pad_motion_to_length(
            src_motion_198, padded_target_len
        )

        # Compose the VACE source by role. Known coordinates must carry target
        # evidence, while generated coordinates carry the paired source as edit
        # context. Replacing the whole tensor with the paired source would make
        # x_t[known] and the edit-context block disagree whenever source != target.
        src_mask = results.get('src_mask')
        if src_mask is None:
            src_mask = torch.ones(padded_target_len, 198)
            results['src_mask'] = src_mask
        src_mask = src_mask.to(dtype=src_motion_198.dtype)
        results['src_motion'] = (
            tgt_motion.to(dtype=src_motion_198.dtype) * (1.0 - src_mask)
            + src_motion_198 * src_mask
        )
        results['src_length'] = valid_target_len
        results['edit_mode'] = True

        return results
