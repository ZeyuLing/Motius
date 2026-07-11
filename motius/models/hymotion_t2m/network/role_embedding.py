"""Role Embedding for DSCF v3.

Injects per-frame "role" information into the denoising input via learnable
embeddings. Each frame/joint-group is assigned one of three roles based on the
condition mask, providing explicit semantic signal about what the model should
do at each position.

Roles:
    KEEP (0): This joint-group at this frame is known (mask=0). The model
        should preserve/condition on this information.
    GENERATE (1): This joint-group at this frame is unknown (mask=1, completion
        mode). The model should generate plausible motion here.
    EDIT (2): This joint-group at this frame has pre-edit values (mask=1,
        editing mode). The model should modify while respecting the hint.

Design choices:
    - 3 roles × feat_dim embedding table (not per-joint-group): joint identity
      is already captured by the position within the 198-dim representation.
      Adding per-group embeddings (23 × 3 × feat_dim) would be 69 separate
      embeddings, overparameterized for limited training signal.
    - Per-frame role aggregation: the mask is (B, L, D_motion), but we reduce
      to per-frame role by majority vote across dims. When a frame has mixed
      mask values (some joints KEEP, some GENERATE), the predominant role wins.
      This is acceptable because:
        a) Most training masks are frame-coherent (M3/M5/M6) or joint-coherent (M4)
        b) The cross-attention from MotionCondEncoder provides fine-grained
           per-joint condition info; RoleEmbedding provides coarse frame-level intent.
    - Alternatively, per-joint-group role assignment is supported via
      `mode='per_group'`, which produces (B, L, 23, feat_dim) role embeddings
      and projects them to match the input dimension.
    - Zero initialization: role embeddings start at zero so the pretrained
      backbone is unperturbed at initialization.

Usage:
    role_emb = self.role_embedding(mask, mode='per_frame')  # (B, L, feat_dim)
    x_input = x_input + role_emb
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor


# Number of joint groups in 198-dim representation:
# 1 (translation) + 22 (joints) = 23 groups
NUM_JOINT_GROUPS = 23

# Dimension sizes per group in 198-dim:
# translation: 3, each joint: 6 (rot6d) for joints 0-21,
# + 3 (position) for joints 1-21 (21 position groups)
# Total: 3 + 22*6 = 135 + 21*3 = 63 → 198
# For mask, the groups are still defined on the full 198-dim layout.

# Role indices
ROLE_KEEP = 0
ROLE_GENERATE = 1
ROLE_EDIT = 2
NUM_ROLES = 3


class RoleEmbedding(nn.Module):
    """Learnable role embedding for DSCF v3.

    Assigns KEEP/GENERATE/EDIT roles to each frame (or per-joint-group per-frame)
    based on the condition mask, and produces an additive embedding for the input.

    Args:
        feat_dim: Output embedding dimension (must match transformer feat_dim, e.g. 1024).
        motion_dim: Raw motion dimension (e.g. 198).
        num_roles: Number of role categories (default 3: KEEP/GENERATE/EDIT).
        mode: Role assignment mode.
            'per_frame': majority vote across joints → (B, L, feat_dim).
            'per_group': per-joint-group roles, projected → (B, L, feat_dim).
        zero_init: If True (default), initialize embeddings to zero so model
            starts unperturbed. Recommended for fine-tuning from pretrained.
    """

    def __init__(
        self,
        feat_dim: int = 1024,
        motion_dim: int = 198,
        num_roles: int = NUM_ROLES,
        mode: Literal['per_frame', 'per_group'] = 'per_frame',
        zero_init: bool = True,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.motion_dim = motion_dim
        self.num_roles = num_roles
        self.mode = mode

        if mode == 'per_frame':
            # Simple: 3 embeddings, each of dim feat_dim
            self.role_embed = nn.Embedding(num_roles, feat_dim)
            if zero_init:
                nn.init.zeros_(self.role_embed.weight)

        elif mode == 'per_group':
            # Per-group: 3 embeddings × num_groups, then project to feat_dim
            self.role_embed = nn.Embedding(num_roles, feat_dim)
            # For per-group mode, we first get per-group embeddings (B, L, 23, feat_dim)
            # then aggregate: mean over groups + project
            self.group_proj = nn.Linear(feat_dim, feat_dim, bias=False)
            if zero_init:
                nn.init.zeros_(self.role_embed.weight)
                nn.init.zeros_(self.group_proj.weight)
        else:
            raise ValueError(f"Unknown mode: {mode!r}. Use 'per_frame' or 'per_group'.")

        # Build group boundaries for mask → role conversion
        self._register_group_boundaries(motion_dim)

    def _register_group_boundaries(self, motion_dim: int) -> None:
        """Compute joint-group boundaries for mask→role conversion.

        For 198-dim: group 0 = translation (dims 0:3),
                     groups 1-22 = rot6d joints (dims 3:135, each 6 wide),
                     groups 23-43 = position joints (dims 135:198, each 3 wide)
        For 135-dim: group 0 = translation (dims 0:3),
                     groups 1-22 = rot6d joints (dims 3:135, each 6 wide)
        """
        boundaries = []
        # Translation group
        boundaries.append((0, 3))

        # Rot6d joints (22 joints × 6 dims)
        for j in range(22):
            start = 3 + j * 6
            boundaries.append((start, start + 6))

        # Position joints (21 joints × 3 dims, starting at dim 135) — only for 198-dim
        if motion_dim >= 198:
            for j in range(21):
                start = 135 + j * 3
                boundaries.append((start, start + 3))

        self.register_buffer(
            '_group_starts',
            torch.tensor([b[0] for b in boundaries], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            '_group_ends',
            torch.tensor([b[1] for b in boundaries], dtype=torch.long),
            persistent=False,
        )
        self._num_groups = len(boundaries)

    def _mask_to_frame_roles(self, mask: Tensor) -> Tensor:
        """Convert (B, L, D) binary mask to per-frame roles (B, L).

        Role assignment: if mean(mask[b,l,:]) > 0.5 → GENERATE, else KEEP.
        EDIT role is assigned externally (see forward's `edit_mask` argument).
        """
        # mask: (B, L, D), 1=generate, 0=known
        frame_density = mask.mean(dim=-1)  # (B, L)
        roles = torch.where(
            frame_density > 0.5,
            torch.full_like(frame_density, ROLE_GENERATE, dtype=torch.long),
            torch.full_like(frame_density, ROLE_KEEP, dtype=torch.long),
        )
        return roles  # (B, L) of int64 in {0, 1}

    def _mask_to_group_roles(self, mask: Tensor) -> Tensor:
        """Convert (B, L, D) binary mask to per-group roles (B, L, num_groups).

        For each joint group, check if its dims are majority-masked.
        """
        B, L, D = mask.shape
        num_groups = self._num_groups
        roles = torch.zeros(B, L, num_groups, dtype=torch.long, device=mask.device)

        for g in range(num_groups):
            start = self._group_starts[g].item()
            end = self._group_ends[g].item()
            group_density = mask[:, :, start:end].mean(dim=-1)  # (B, L)
            roles[:, :, g] = torch.where(
                group_density > 0.5,
                torch.ones_like(group_density, dtype=torch.long) * ROLE_GENERATE,
                torch.zeros_like(group_density, dtype=torch.long),  # ROLE_KEEP
            )

        return roles  # (B, L, num_groups)

    def forward(
        self,
        mask: Tensor,
        edit_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute role embeddings from the condition mask.

        Args:
            mask: (B, L, motion_dim) — binary mask (1=generate, 0=known).
            edit_mask: (B, L) — optional boolean, True where the frame is in
                editing mode (has pre-edit values in reactive channel). If provided,
                frames with edit_mask=True AND mask>0.5 get ROLE_EDIT instead of
                ROLE_GENERATE.

        Returns:
            role_emb: (B, L, feat_dim) — additive embedding for the input.
        """
        if self.mode == 'per_frame':
            roles = self._mask_to_frame_roles(mask)  # (B, L)

            # Override GENERATE → EDIT where applicable
            if edit_mask is not None:
                edit_positions = edit_mask & (roles == ROLE_GENERATE)
                roles = torch.where(
                    edit_positions,
                    torch.full_like(roles, ROLE_EDIT),
                    roles,
                )

            role_emb = self.role_embed(roles)  # (B, L, feat_dim)

        elif self.mode == 'per_group':
            roles = self._mask_to_group_roles(mask)  # (B, L, num_groups)

            # Override GENERATE → EDIT where applicable
            if edit_mask is not None:
                # edit_mask: (B, L) → expand to (B, L, num_groups)
                edit_expanded = edit_mask.unsqueeze(-1).expand_as(roles)
                edit_positions = edit_expanded & (roles == ROLE_GENERATE)
                roles = torch.where(
                    edit_positions,
                    torch.full_like(roles, ROLE_EDIT),
                    roles,
                )

            # (B, L, num_groups) → embed → (B, L, num_groups, feat_dim)
            group_embs = self.role_embed(roles)

            # Aggregate: mean across groups → (B, L, feat_dim)
            role_emb = group_embs.mean(dim=2)

            # Project
            role_emb = self.group_proj(role_emb)

        return role_emb
