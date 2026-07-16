"""
Rotary Position Embedding (RoPE) module for MotionWan Transformer.

This module implements 2D rotary position embeddings specifically designed for
motion data, which has two spatial dimensions: temporal frames and body joints.
The RoPE is factorized into separate embeddings for each dimension, following
the approach used in video/image transformers.

JOINT COUNT ARCHITECTURE (Critical for understanding):
===============================================
PRISM uses a total of 23 joint positions:
  - 1 translation token (index 0)      — represents root translation in world space
  - 22 body joints (indices 1-22)      — SMPL-22 kinematic tree:
    * Pelvis (root)
    * L/R Hip, L/R Knee, L/R Ankle, L/R Foot (legs)
    * Spine1, Spine2, Spine3 (spine)
    * Neck, Head (head)
    * L/R Collar, L/R Shoulder, L/R Elbow, L/R Wrist (arms)

This 1+22 structure is maintained throughout all RoPE modes:
  - Translation token: identity RoPE (cos=1, sin=0) across all modes
  - Body joints: mode-specific RoPE based on position encoding
    * Sequential: full-token positions 0..22; body tokens occupy 1..22
    * Spectral: eigenvector coordinates from kinematic tree
    * Spectral_unified: body positions 1..22 plus a small signed spectral residual
    * DFS: depth-first-search ordering of kinematic tree, shifted to 1..22

When input has shape (B, C, T, J) with J=23:
  - VAE outputs motion with 23 joint positions
  - Token 0 (j=0) → translation
  - Tokens 1-22 (j=1-22) → body joints [0-21]
  - Concatenation: torch.cat([trans_freqs, joint_freqs], dim=0) yields shape (23, j_dim)

Supports four joint position modes:
    - "sequential": Standard sequential indices (0, 1, 2, ...) for joints.
    - "spectral": Laplacian eigenvector coordinates from SMPL-22 kinematic tree.
      Encodes kinematic distance as RoPE attention bias. Per-mode frequency
      decomposition — NOT compatible with sequential pretrained weights.
    - "spectral_unified": Sequential body-token positions with a signed
      topology residual and full j_dim frequency basis. Compatible with
      sequential pretrained weights while avoiding translation/body collision.
    - "dfs": Depth-first-search ordering of the kinematic tree.

Reference:
    - RoFormer: Enhanced Transformer with Rotary Position Embedding
      (https://arxiv.org/abs/2104.09864)
    - Rotary Position Embedding for Vision Transformer
      (https://arxiv.org/abs/2403.13298)
"""

from typing import Optional, Tuple
from torch import nn
import torch
import numpy as np
from diffusers.models.embeddings import get_1d_rotary_pos_embed


# SMPL-22 kinematic tree parent array (22 body joints, no translation)
# Index i has parent SMPL_22_PARENTS[i]. Root (pelvis, index 0) has parent -1.
# NOTE: These are the 22 body joints only. Translation (position 0) is separate.
SMPL_22_PARENTS = [
    -1,  # 0: Pelvis (root)
    0,   # 1: L_Hip -> Pelvis
    0,   # 2: R_Hip -> Pelvis
    0,   # 3: Spine1 -> Pelvis
    1,   # 4: L_Knee -> L_Hip
    2,   # 5: R_Knee -> R_Hip
    3,   # 6: Spine2 -> Spine1
    4,   # 7: L_Ankle -> L_Knee
    5,   # 8: R_Ankle -> R_Knee
    6,   # 9: Spine3 -> Spine2
    7,   # 10: L_Foot -> L_Ankle
    8,   # 11: R_Foot -> R_Ankle
    9,   # 12: Neck -> Spine3
    9,   # 13: L_Collar -> Spine3
    9,   # 14: R_Collar -> Spine3
    12,  # 15: Head -> Neck
    13,  # 16: L_Shoulder -> L_Collar
    14,  # 17: R_Shoulder -> R_Collar
    16,  # 18: L_Elbow -> L_Shoulder
    17,  # 19: R_Elbow -> R_Shoulder
    18,  # 20: L_Wrist -> L_Elbow
    19,  # 21: R_Wrist -> R_Elbow
]


def _compute_spectral_coords(num_joints: int = 22, num_modes: int = 4) -> np.ndarray:
    """
    Compute spectral coordinates from the SMPL-22 kinematic tree graph Laplacian.

    Uses the first K non-trivial eigenvectors of the normalized graph Laplacian
    as positional coordinates for each joint. These coordinates encode kinematic
    distance: joints close in the kinematic tree have similar spectral coordinates,
    while distant joints are well-separated.

    Args:
        num_joints: Number of joints (22 for SMPL-22). Translation token is separate.
        num_modes: Number of non-trivial eigenvectors to use (K).

    Returns:
        spectral_coords: Array of shape (num_joints, num_modes) containing
            the spectral positional coordinates for each of the 22 body joints.
    """
    # Build adjacency matrix from kinematic tree
    adj = np.zeros((num_joints, num_joints), dtype=np.float64)
    for child, parent in enumerate(SMPL_22_PARENTS):
        if parent >= 0:
            adj[child, parent] = 1.0
            adj[parent, child] = 1.0

    # Compute degree matrix
    degree = np.diag(adj.sum(axis=1))

    # Compute graph Laplacian: L = D - A
    laplacian = degree - adj

    # Compute normalized Laplacian: L_norm = D^{-1/2} L D^{-1/2}
    d_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(adj.sum(axis=1), 1e-10)))
    laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt

    # Eigendecomposition (eigenvalues in ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian_norm)

    # Skip the first eigenvector (trivial, constant) and take next K
    # eigenvectors[:, 0] corresponds to eigenvalue 0 (constant vector)
    spectral_coords = eigenvectors[:, 1:num_modes + 1]


    # ==================== FIX: CANONICALIZE EIGENVECTOR SIGNS ====================
    # Eigenvectors are defined up to sign: if v is an eigenvector, so is -v.
    # Different numpy/BLAS versions and CPU architectures can return opposite signs.
    # Canonicalize by enforcing that the first joint (Pelvis, root) has a positive
    # coordinate in each mode. This ensures deterministic, consistent signs across
    # all runs, systems, and library versions.
    for mode_idx in range(num_modes):
        if spectral_coords[0, mode_idx] < 0:
            spectral_coords[:, mode_idx] *= -1.0

    return spectral_coords


def _compute_dfs_ordering(num_joints: int = 22) -> np.ndarray:
    """
    Compute depth-first-search ordering of the SMPL-22 kinematic tree.

    Returns an array where dfs_order[i] is the DFS visit index of joint i.
    This ensures parent-child joints have nearby indices while maintaining
    the tree structure.

    Args:
        num_joints: Number of joints (22 for SMPL-22). Translation token is separate.

    Returns:
        dfs_order: Array of shape (num_joints,) with DFS visit indices.
    """
    # Build children list
    children = [[] for _ in range(num_joints)]
    for child, parent in enumerate(SMPL_22_PARENTS):
        if parent >= 0:
            children[parent].append(child)

    # DFS traversal
    dfs_order = np.zeros(num_joints, dtype=np.float64)
    visit_idx = 0
    stack = [0]  # Start from root (pelvis)

    while stack:
        node = stack.pop()
        dfs_order[node] = visit_idx
        visit_idx += 1
        # Push children in reverse order so leftmost child is visited first
        for child in reversed(children[node]):
            stack.append(child)

    return dfs_order


def _compute_projected_spectral_positions(
    num_joints: int = 22,
    num_modes: int = 4,
    spectral_scale: float = 22.0,
    position_offset: float = 1.0,
    topology_mix: float = 0.25,
    tie_break_eps: float = 1e-3,
) -> np.ndarray:
    """
    Project signed Laplacian spectral coordinates to scalar RoPE positions.

    The result is a topology-aware perturbation of the pretrained sequential
    positions, not a wholesale re-indexing.  Body tokens therefore stay close to
    positions 1..22 while translation keeps the unique identity position 0.
    This preserves the pretrained RoPE phase geometry and injects SMPL tree
    topology through relative phase residuals.
    """
    spectral_coords = _compute_spectral_coords(
        num_joints=num_joints, num_modes=num_modes
    )

    base_weights = np.array([1.0, 1.7, 2.9, 4.3], dtype=np.float64)
    if num_modes <= len(base_weights):
        weights = base_weights[:num_modes]
    else:
        extra = np.arange(len(base_weights) + 1, num_modes + 1, dtype=np.float64)
        weights = np.concatenate([base_weights, extra * 1.3 + 0.7])

    dfs_order = _compute_dfs_ordering(num_joints=num_joints)
    raw_positions = spectral_coords @ weights
    raw_positions = raw_positions + tie_break_eps * dfs_order

    min_pos = raw_positions.min()
    max_pos = raw_positions.max()
    if max_pos - min_pos < 1e-8:
        raise ValueError("Projected spectral positions collapsed to a constant.")

    centered = raw_positions - raw_positions.mean()
    denom = np.max(np.abs(centered))
    if denom < 1e-8:
        raise ValueError("Projected spectral positions collapsed after centering.")
    topology_residual = topology_mix * centered / denom

    base_positions = np.linspace(
        position_offset,
        spectral_scale,
        num_joints,
        dtype=np.float64,
    )
    return base_positions + topology_residual


class MotionWanRotaryPosEmbed(nn.Module):
    """
    2D Rotary Position Embedding for motion sequences with proper handling of 23 joint positions.

    ARCHITECTURE (Core Design):
    ===========================
    This module generates rotary position embeddings for motion data with two
    spatial dimensions: temporal (frames) and spatial (joints). The embedding
    is factorized into two 1D embeddings that are later combined, allowing the
    model to capture both temporal dynamics and spatial joint relationships.

    Joint Position Structure:
      - Position 0: Translation token (world-space root translation)
        * Always uses identity RoPE: cos=1, sin=0 across all modes
        * Represents global motion offset, not part of kinematic tree
      - Positions 1-22: Body joints (SMPL-22 kinematic tree)
        * Position encoding varies by mode (sequential, spectral, dfs)
        * Maintains kinematic tree structure relationships

    Concatenation Pattern (in forward method):
      ```
      trans_freqs_cos: shape (j_dim,)        [identity RoPE for translation]
      joint_freqs_cos: shape (22, j_dim)     [per-joint RoPE for body]

      all_joint_cos = cat([trans_freqs_cos.unsqueeze(0), joint_freqs_cos], dim=0)
      → shape (23, j_dim)                     [full 23-position RoPE buffer]

      final: joint_cos = all_joint_cos[:ppj]
      → shape (ppj, j_dim)                    [ppj=23 when patch_size=(1,1)]
      ```

    The attention head dimension is split between the two axes:
        - First half (t_dim): Encodes temporal/frame position
        - Second half (j_dim): Encodes spatial/joint position

    Supports four joint position modes:
        - "sequential": Standard 0, 1, 2, ... indices (default, backward-compatible)
        - "spectral": Laplacian eigenvector coordinates from kinematic tree.
          Each joint gets a multi-dimensional spectral coordinate, and the j_dim
          is further split across spectral modes. NOT compatible with sequential
          pretrained weights (different frequency basis).
        - "spectral_unified": Like "spectral" but uses a single scalar position
          per joint. The scalar is the pretrained sequential body position
          (1..22) plus a small signed spectral residual, with the FULL j_dim
          frequency basis. Compatible with sequential pretrained weights.
        - "dfs": DFS ordering of the kinematic tree as joint indices.

    Args:
        attention_head_dim (int): Dimension of each attention head.
        patch_size (Tuple[int, int]): Patch size as (patch_frames, patch_joints).
        max_seq_len (int): Maximum sequence length for pre-computing RoPE.
        theta (float): Base frequency for rotary embeddings. Defaults to 10000.0.
        joint_pos_mode (str): Joint position encoding mode. One of
            "sequential", "spectral", "spectral_unified", "dfs". Defaults to "sequential".
        num_spectral_modes (int): Number of Laplacian eigenvectors to use
            in spectral mode. Defaults to 4.
        spectral_scale (float or None): Scale factor for spectral coordinates.
            If None, defaults to num_joints (22). Spectral coordinates are
            multiplied by this value before being used as position indices.
    """

    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int],
        max_seq_len: int,
        theta: float = 10000.0,
        joint_pos_mode: str = "sequential",
        num_spectral_modes: int = 4,
        spectral_scale: Optional[float] = None,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.joint_pos_mode = joint_pos_mode
        self.num_spectral_modes = num_spectral_modes
        self.spectral_scale = spectral_scale

        # Split attention head dimension between temporal and joint axes
        # j_dim gets half, t_dim gets the rest (handles odd dimensions)
        j_dim = attention_head_dim // 2
        t_dim = attention_head_dim - j_dim

        self._t_dim = t_dim
        self._j_dim = j_dim

        # Use float64 for frequency computation precision (float32 on MPS)
        freqs_dtype = (
            torch.float32 if torch.backends.mps.is_available() else torch.float64
        )

        if joint_pos_mode == "sequential":
            # Standard sequential mode: pre-compute 1D RoPE for both dimensions
            freqs_cos = []
            freqs_sin = []
            for dim in [t_dim, j_dim]:
                freq_cos, freq_sin = get_1d_rotary_pos_embed(
                    dim,
                    max_seq_len,
                    theta,
                    use_real=True,
                    repeat_interleave_real=True,
                    freqs_dtype=freqs_dtype,
                )
                freqs_cos.append(freq_cos)
                freqs_sin.append(freq_sin)

            # Concatenate temporal and joint frequencies along the last dimension
            # Shape: (max_seq_len, attention_head_dim)
            self.register_buffer(
                "freqs_cos", torch.cat(freqs_cos, dim=1), persistent=True
            )
            self.register_buffer(
                "freqs_sin", torch.cat(freqs_sin, dim=1), persistent=True
            )

        elif joint_pos_mode == "spectral":
            # Spectral mode: use Laplacian eigenvectors as joint positions
            # Temporal axis still uses standard sequential RoPE
            freq_cos_t, freq_sin_t = get_1d_rotary_pos_embed(
                t_dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            self.register_buffer("freqs_cos_t", freq_cos_t, persistent=False)
            self.register_buffer("freqs_sin_t", freq_sin_t, persistent=False)

            # Compute spectral coordinates for 22 body joints (NOT including translation)
            spectral_coords = _compute_spectral_coords(
                num_joints=22, num_modes=num_spectral_modes
            )
            # Scale spectral coordinates
            # Default scale=22.0 represents the SMPL body joint count
            scale = spectral_scale if spectral_scale is not None else 22.0
            spectral_coords = spectral_coords * scale

            # spectral_coords shape: (22, num_spectral_modes)
            # We need to compute per-joint RoPE frequencies using these coords.
            # Split j_dim across spectral modes:
            #   Each mode gets j_dim // num_spectral_modes dimensions
            #   (remainder goes to last mode)
            dims_per_mode = [j_dim // num_spectral_modes] * num_spectral_modes
            remainder = j_dim - sum(dims_per_mode)
            for i in range(remainder):
                dims_per_mode[-(i + 1)] += 1

            # For each of the 22 body joints, compute RoPE frequencies based on spectral coords
            # Pre-compute the frequency bases for each mode
            joint_freqs_cos_list = []
            joint_freqs_sin_list = []

            for joint_idx in range(22):
                mode_cos_list = []
                mode_sin_list = []
                for mode_idx in range(num_spectral_modes):
                    dim = dims_per_mode[mode_idx]
                    # Position for this joint in this spectral mode
                    pos = spectral_coords[joint_idx, mode_idx]
                    # Compute frequencies for a single position
                    # get_1d_rotary_pos_embed expects integer positions in a range,
                    # but we need fractional positions. We compute manually.
                    half_dim = dim // 2
                    freq_seq = torch.arange(
                        0, half_dim, dtype=freqs_dtype
                    )
                    # Standard RoPE frequency formula: theta^(-2i/d)
                    freqs = 1.0 / (theta ** (2.0 * freq_seq / dim))
                    # Multiply by position
                    angles = pos * freqs
                    cos_vals = torch.cos(angles)
                    sin_vals = torch.sin(angles)
                    # repeat_interleave pattern: [cos(f0), cos(f0), cos(f1), cos(f1), ...]
                    cos_vals = cos_vals.repeat_interleave(2)
                    sin_vals = sin_vals.repeat_interleave(2)
                    mode_cos_list.append(cos_vals)
                    mode_sin_list.append(sin_vals)

                # Concatenate all modes for this joint: shape (j_dim,)
                joint_cos = torch.cat(mode_cos_list, dim=0).float()
                joint_sin = torch.cat(mode_sin_list, dim=0).float()
                joint_freqs_cos_list.append(joint_cos)
                joint_freqs_sin_list.append(joint_sin)

            # Stack 22 body joints: shape (22, j_dim)
            joint_freqs_cos = torch.stack(joint_freqs_cos_list, dim=0)
            joint_freqs_sin = torch.stack(joint_freqs_sin_list, dim=0)

            self.register_buffer(
                "joint_freqs_cos", joint_freqs_cos, persistent=False
            )
            self.register_buffer(
                "joint_freqs_sin", joint_freqs_sin, persistent=False
            )

            # For the TRANSLATION token (position 0 in J=23 sequence),
            # use identity RoPE (cos=1, sin=0) since translation is NOT
            # part of the kinematic tree — it's a separate world-space offset.
            # This is concatenated with joint RoPE in forward() to create (23, j_dim).
            trans_cos = torch.ones(j_dim, dtype=torch.float32)
            trans_sin = torch.zeros(j_dim, dtype=torch.float32)
            self.register_buffer("trans_freqs_cos", trans_cos, persistent=False)
            self.register_buffer("trans_freqs_sin", trans_sin, persistent=False)

        elif joint_pos_mode == "spectral_unified":
            # Spectral Unified mode: same frequency basis as sequential (full j_dim),
            # but with spectral-derived scalar positions instead of integer indices.
            #
            # Key difference from "spectral" mode:
            #   - "spectral" splits j_dim into num_spectral_modes independent frequency
            #     spaces with dim_per_mode dimensions each. This creates a DIFFERENT
            #     frequency basis (steeper decay) incompatible with pretrained weights.
            #   - "spectral_unified" keeps the SAME frequency basis (j_dim=64) as
            #     sequential mode, only changing positions from integers to spectral
            #     scalars. Fully compatible with sequential pretrained weights.
            #
            # Position computation: sequential body-token positions (1..22)
            # plus a small signed spectral residual.  This preserves most of
            # the pretrained RoPE phase while injecting kinematic topology and
            # avoiding the L2-norm collapse of mirrored limbs.  Position 0 is
            # reserved exclusively for the translation token.

            # Temporal axis: standard sequential RoPE (identical to sequential mode)
            freq_cos_t, freq_sin_t = get_1d_rotary_pos_embed(
                t_dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            self.register_buffer("freqs_cos_t", freq_cos_t, persistent=False)
            self.register_buffer("freqs_sin_t", freq_sin_t, persistent=False)

            scale = spectral_scale if spectral_scale is not None else 22.0
            spectral_positions = _compute_projected_spectral_positions(
                num_joints=22,
                num_modes=num_spectral_modes,
                spectral_scale=scale,
                position_offset=1.0,
            )

            # Now use get_1d_rotary_pos_embed with these fractional positions
            # (same j_dim=64 frequency basis as sequential mode)
            spectral_pos_tensor = torch.from_numpy(
                spectral_positions
            ).to(dtype=freqs_dtype)

            # Compute RoPE for these fractional positions manually
            # (get_1d_rotary_pos_embed only supports integer/range positions)
            half_dim = j_dim // 2
            freq_seq = torch.arange(0, half_dim, dtype=freqs_dtype)
            # Standard RoPE: theta^(-2i/d) where d = j_dim (full dimension!)
            freqs = 1.0 / (theta ** (2.0 * freq_seq / j_dim))  # (j_dim/2,)

            # outer product: (22,) × (j_dim/2,) -> (22, j_dim/2)
            angles = torch.outer(spectral_pos_tensor, freqs)
            cos_vals = torch.cos(angles).float()  # (22, j_dim/2)
            sin_vals = torch.sin(angles).float()  # (22, j_dim/2)

            # repeat_interleave pattern to match sequential: [cos(f0), cos(f0), cos(f1), ...]
            joint_freqs_cos = cos_vals.repeat_interleave(2, dim=1)  # (22, j_dim)
            joint_freqs_sin = sin_vals.repeat_interleave(2, dim=1)  # (22, j_dim)

            self.register_buffer(
                "joint_freqs_cos", joint_freqs_cos, persistent=False
            )
            self.register_buffer(
                "joint_freqs_sin", joint_freqs_sin, persistent=False
            )

            # Translation token: identity RoPE (same as spectral mode)
            # Concatenated in forward() to create (23, j_dim) buffer
            trans_cos = torch.ones(j_dim, dtype=torch.float32)
            trans_sin = torch.zeros(j_dim, dtype=torch.float32)
            self.register_buffer("trans_freqs_cos", trans_cos, persistent=False)
            self.register_buffer("trans_freqs_sin", trans_sin, persistent=False)

        elif joint_pos_mode == "dfs":
            # DFS mode: use DFS ordering as joint positions
            # Temporal axis still uses standard sequential RoPE
            freq_cos_t, freq_sin_t = get_1d_rotary_pos_embed(
                t_dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            self.register_buffer("freqs_cos_t", freq_cos_t, persistent=False)
            self.register_buffer("freqs_sin_t", freq_sin_t, persistent=False)

            # Compute DFS ordering and use as positions for joint RoPE
            dfs_order = _compute_dfs_ordering(num_joints=22) + 1
            # dfs_order shape: (22,) with values 1..22 in DFS visit order.
            # Position 0 is reserved for the translation token.

            # Pre-compute per-joint RoPE using DFS positions
            # Use get_1d_rotary_pos_embed for max positions, then index
            freq_cos_j, freq_sin_j = get_1d_rotary_pos_embed(
                j_dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            # freq_cos_j shape: (max_seq_len, j_dim)
            # Index by DFS order for each of 22 body joints
            dfs_indices = torch.from_numpy(dfs_order).long()
            joint_freqs_cos = freq_cos_j[dfs_indices].float()  # (22, j_dim)
            joint_freqs_sin = freq_sin_j[dfs_indices].float()  # (22, j_dim)

            self.register_buffer(
                "joint_freqs_cos", joint_freqs_cos, persistent=False
            )
            self.register_buffer(
                "joint_freqs_sin", joint_freqs_sin, persistent=False
            )

            # Translation token (position 0): use identity RoPE (cos=1, sin=0)
            # NOT part of kinematic tree, so gets identity rather than DFS position
            # Concatenated in forward() to create (23, j_dim) buffer
            trans_cos = torch.ones(j_dim, dtype=torch.float32)
            trans_sin = torch.zeros(j_dim, dtype=torch.float32)
            self.register_buffer("trans_freqs_cos", trans_cos, persistent=False)
            self.register_buffer("trans_freqs_sin", trans_sin, persistent=False)

        else:
            raise ValueError(
                f"Unknown joint_pos_mode: '{joint_pos_mode}'. "
                f"Must be one of 'sequential', 'spectral', 'spectral_unified', 'dfs'."
            )

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary position embeddings for the given input tensor.

        Input/Output Shape Semantics (CRITICAL):
        ========================================
        Input motion shape: (batch, channels, frames, joints)
          - joints dimension: 23 = 1 translation + 22 body joints

        Output RoPE shape: (1, num_patches, 1, attention_head_dim)
          - num_patches = (frames // patch_t) * (joints // patch_j)
          - With default patch=(1,1): num_patches = frames * 23

        RoPE Construction (All Modes):
        ==============================
        1. Temporal RoPE: (ppf, t_dim) — standard sequential for all modes
        2. Joint RoPE construction differs by mode but ALWAYS:
           - Registers per-joint buffers: joint_freqs_cos (22, j_dim), trans_freqs_cos (j_dim,)
           - Concatenates: torch.cat([trans_freqs_cos.unsqueeze(0), joint_freqs_cos])
             → yields shape (23, j_dim) = (1+22, j_dim)
           - Slices to ppj patches: all_joint_cos[:ppj]
           - With patch_j=1: ppj=23, so full (23, j_dim) is retained

        Args:
            hidden_states (torch.Tensor): Input tensor with shape
                (batch_size, num_channels, num_frames, num_joints).
                num_joints should be 23 (1 translation + 22 body).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (freqs_cos, freqs_sin) each with
                shape (1, num_patches, 1, attention_head_dim).
        """
        # Get target device and dtype from input
        device = hidden_states.device
        dtype = hidden_states.dtype

        # ============ FP32 PRECISION FIX FOR ROPE ============
        # RoPE frequencies must maintain fp32 precision for accurate positional encoding.
        # When in fp16 autocast context, dtype will be fp16, but we override it to fp32
        # for RoPE buffers because:
        # 1. RoPE frequencies are used in multiplicative operations (x * cos(θ))
        # 2. fp16 precision loss accumulates across the attention head dimension
        # 3. Lower precision = weaker positional encoding = worse sequence understanding
        # Solution: Always use fp32 for RoPE buffers, regardless of training dtype
        safe_rope_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype

        # Extract input dimensions
        batch_size, num_channels, num_frames, num_joints = hidden_states.shape
        # num_joints should be 23 (1 translation + 22 body)

        # Calculate number of patches per dimension
        p_t, p_j = self.patch_size
        ppf = num_frames // p_t  # patches per frame dimension
        ppj = num_joints // p_j  # patches per joint dimension

        if self.joint_pos_mode == "sequential":
            # Original sequential mode (backward-compatible)
            split_sizes = [
                self.attention_head_dim - (self.attention_head_dim // 2),  # t_dim
                self.attention_head_dim // 2,  # j_dim
            ]

            # Move buffers to correct device and SAFE dtype for RoPE
            freqs_cos_all = self.freqs_cos.to(device=device, dtype=safe_rope_dtype)
            freqs_sin_all = self.freqs_sin.to(device=device, dtype=safe_rope_dtype)

            freqs_cos = freqs_cos_all.split(split_sizes, dim=1)
            freqs_sin = freqs_sin_all.split(split_sizes, dim=1)

            # Slice and expand temporal frequencies
            freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, -1).expand(ppf, ppj, -1)
            freqs_cos_j = freqs_cos[1][:ppj].view(1, ppj, -1).expand(ppf, ppj, -1)

            freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, -1).expand(ppf, ppj, -1)
            freqs_sin_j = freqs_sin[1][:ppj].view(1, ppj, -1).expand(ppf, ppj, -1)

            # Concatenate and reshape
            freqs_cos = torch.cat([freqs_cos_f, freqs_cos_j], dim=-1).reshape(
                1, ppf * ppj, 1, -1
            )
            freqs_sin = torch.cat([freqs_sin_f, freqs_sin_j], dim=-1).reshape(
                1, ppf * ppj, 1, -1
            )

        elif self.joint_pos_mode in ("spectral", "spectral_unified", "dfs"):
            # Spectral/Spectral_unified/DFS modes: per-joint pre-computed frequencies
            # All three modes follow the same 23-position concatenation pattern:
            # (1 translation identity RoPE) + (22 body joints mode-specific RoPE)

            # Temporal: use sequential RoPE (same for all modes)
            # Move temporal buffers to correct device and SAFE dtype for RoPE
            freqs_cos_t = self.freqs_cos_t.to(device=device, dtype=safe_rope_dtype)[:ppf]  # (ppf, t_dim)
            freqs_sin_t = self.freqs_sin_t.to(device=device, dtype=safe_rope_dtype)[:ppf]  # (ppf, t_dim)

            # Joint: per-joint pre-computed frequencies
            # Move joint buffers to correct device and SAFE dtype for RoPE
            joint_freqs_cos = self.joint_freqs_cos.to(device=device, dtype=safe_rope_dtype)  # (22, j_dim)
            joint_freqs_sin = self.joint_freqs_sin.to(device=device, dtype=safe_rope_dtype)  # (22, j_dim)
            trans_freqs_cos = self.trans_freqs_cos.to(device=device, dtype=safe_rope_dtype)  # (j_dim,)
            trans_freqs_sin = self.trans_freqs_sin.to(device=device, dtype=safe_rope_dtype)  # (j_dim,)

            # Build the full joint frequency array including translation token
            # ============================================================
            # CRITICAL ARCHITECTURE: 1 translation + 22 body joints = 23 total
            #
            # Token ordering in PRISM:
            #   token 0 = translation (identity RoPE)
            #   tokens 1-22 = body joints (mode-specific RoPE)
            # Total ppj = num_joints // p_j.
            # With default p_j=1 and num_joints=23: ppj=23
            # (VAE outputs J=23: 1 translation + 22 body joints)
            #
            # Concatenation builds full (23, j_dim) frequency array:
            #   trans_freqs_cos (j_dim,) -> unsqueeze(0) -> (1, j_dim)
            #   joint_freqs_cos (22, j_dim) -> stays (22, j_dim)
            #   cat(...) -> (1+22, j_dim) = (23, j_dim)
            # ============================================================

            # Construct per-token joint frequencies
            # Token 0 = translation -> identity RoPE (cos=1, sin=0)
            # Tokens 1..22 = body joints -> spectral/spectral_unified/dfs RoPE
            all_joint_cos = torch.cat(
                [trans_freqs_cos.unsqueeze(0), joint_freqs_cos], dim=0
            )  # (23, j_dim) = (1+22, j_dim)
            all_joint_sin = torch.cat(
                [trans_freqs_sin.unsqueeze(0), joint_freqs_sin], dim=0
            )  # (23, j_dim) = (1+22, j_dim)

            # Slice to actual number of joint patches
            # (in case ppj != 23, e.g., if p_j > 1)
            # With default patch_size=(1,1) and num_joints=23: ppj=23
            # → full (23, j_dim) buffer is retained
            joint_cos = all_joint_cos[:ppj]  # (ppj, j_dim)
            joint_sin = all_joint_sin[:ppj]  # (ppj, j_dim)

            # Expand temporal: (ppf, 1, t_dim) -> (ppf, ppj, t_dim)
            freqs_cos_f = freqs_cos_t.view(ppf, 1, -1).expand(ppf, ppj, -1)
            freqs_sin_f = freqs_sin_t.view(ppf, 1, -1).expand(ppf, ppj, -1)

            # Expand joint: (1, ppj, j_dim) -> (ppf, ppj, j_dim)
            freqs_cos_j = joint_cos.view(1, ppj, -1).expand(ppf, ppj, -1)
            freqs_sin_j = joint_sin.view(1, ppj, -1).expand(ppf, ppj, -1)

            # Concatenate temporal and joint, reshape for attention
            freqs_cos = torch.cat([freqs_cos_f, freqs_cos_j], dim=-1).reshape(
                1, ppf * ppj, 1, -1
            )
            freqs_sin = torch.cat([freqs_sin_f, freqs_sin_j], dim=-1).reshape(
                1, ppf * ppj, 1, -1
            )

        return freqs_cos, freqs_sin

if __name__ == "__main__":
    """
    Test script for MotionWanRotaryPosEmbed module.

    This script validates:
    1. Basic initialization and forward pass (sequential mode)
    2. Spectral mode initialization and forward pass
    3. Spectral_unified mode initialization and forward pass
    4. DFS mode initialization and forward pass
    5. Output shapes match expected dimensions
    6. Frequency values are within expected ranges
    7. 23-joint concatenation (1 translation + 22 body) works correctly
    8. Backward compatibility (sequential mode unchanged)
    """
    print("=" * 60)
    print("Testing MotionWanRotaryPosEmbed module")
    print("=" * 60)

    # ==================== Test Configuration ====================
    batch_size = 2
    num_channels = 128
    num_frames = 64
    num_joints = 23  # PRISM uses J=23 (1 translation + 22 body)
    attention_head_dim = 128  # Match PRISM 1B config
    patch_size = (1, 1)
    max_seq_len = 1024

    # ==================== Test 1: Sequential Mode ====================
    print("\n[Test 1] Sequential Mode (Backward Compatibility)")
    print("-" * 50)

    rope_seq = MotionWanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        joint_pos_mode="sequential",
    )

    hidden_states = torch.randn(batch_size, num_channels, num_frames, num_joints)
    freqs_cos, freqs_sin = rope_seq(hidden_states)

    expected_seq_len = (num_frames // patch_size[0]) * (num_joints // patch_size[1])
    expected_shape = (1, expected_seq_len, 1, attention_head_dim)

    print(f"  Input shape: {hidden_states.shape}")
    print(f"    → {num_joints} joints = 1 translation + 22 body")
    print(f"  Output freqs_cos shape: {freqs_cos.shape}")
    print(f"  Expected shape: {expected_shape}")

    assert freqs_cos.shape == expected_shape, f"Shape mismatch! {freqs_cos.shape} != {expected_shape}"
    assert freqs_sin.shape == expected_shape, f"Shape mismatch! {freqs_sin.shape} != {expected_shape}"
    print("  ✓ Sequential mode passed!")

    # ==================== Test 2: Spectral Mode ====================
    print("\n[Test 2] Spectral Mode (KT-RoPE with 22 body + 1 translation)")
    print("-" * 50)

    rope_spectral = MotionWanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        joint_pos_mode="spectral",
        num_spectral_modes=4,
        spectral_scale=22.0,
    )

    freqs_cos_sp, freqs_sin_sp = rope_spectral(hidden_states)

    print(f"  Output freqs_cos shape: {freqs_cos_sp.shape}")
    print(f"  Expected shape: {expected_shape}")

    assert freqs_cos_sp.shape == expected_shape, f"Shape mismatch! {freqs_cos_sp.shape} != {expected_shape}"
    assert freqs_sin_sp.shape == expected_shape, f"Shape mismatch! {freqs_sin_sp.shape} != {expected_shape}"

    # Verify translation token has identity RoPE (cos=1, sin=0) in joint dimension
    # Token 0 at frame 0 should have j_dim part = (cos=1, sin=0)
    j_dim = attention_head_dim // 2
    t_dim = attention_head_dim - j_dim
    # First token (frame=0, joint=0) -> index 0 in sequence
    token0_cos = freqs_cos_sp[0, 0, 0, t_dim:]  # joint part of first token
    token0_sin = freqs_sin_sp[0, 0, 0, t_dim:]  # joint part of first token
    assert torch.allclose(token0_cos, torch.ones_like(token0_cos), atol=1e-5), \
        f"Translation token (position 0) cos should be 1, got {token0_cos[:5]}"
    assert torch.allclose(token0_sin, torch.zeros_like(token0_sin), atol=1e-5), \
        f"Translation token (position 0) sin should be 0, got {token0_sin[:5]}"
    print("  ✓ Position 0 (translation): identity RoPE correct!")

    # Verify different body joints get different embeddings
    # Token 1 (body joint 0, pelvis) vs Token 2 (body joint 1, L_Hip)
    token1_cos = freqs_cos_sp[0, 1, 0, t_dim:]
    token2_cos = freqs_cos_sp[0, 2, 0, t_dim:]
    assert not torch.allclose(token1_cos, token2_cos, atol=1e-3), \
        "Different body joints should have different spectral embeddings!"
    print("  ✓ Positions 1-22 (body joints): different spectral embeddings!")

    # Verify spectral mode differs from sequential
    assert not torch.allclose(freqs_cos, freqs_cos_sp, atol=1e-3), \
        "Spectral mode should produce different embeddings from sequential!"
    print("  ✓ Spectral mode differs from sequential mode!")
    print("  ✓ Spectral mode passed!")

    # ==================== Test 3: Spectral Unified Mode ====================
    print("\n[Test 3] Spectral Unified Mode (Compatible with sequential)")
    print("-" * 50)

    rope_spectral_unified = MotionWanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        joint_pos_mode="spectral_unified",
        num_spectral_modes=4,
        spectral_scale=22.0,
    )

    freqs_cos_su, freqs_sin_su = rope_spectral_unified(hidden_states)

    print(f"  Output freqs_cos shape: {freqs_cos_su.shape}")
    assert freqs_cos_su.shape == expected_shape, f"Shape mismatch!"

    # Verify translation token identity
    token0_cos_su = freqs_cos_su[0, 0, 0, t_dim:]
    token0_sin_su = freqs_sin_su[0, 0, 0, t_dim:]
    assert torch.allclose(token0_cos_su, torch.ones_like(token0_cos_su), atol=1e-5)
    assert torch.allclose(token0_sin_su, torch.zeros_like(token0_sin_su), atol=1e-5)
    print("  ✓ Position 0 (translation): identity RoPE correct!")

    # Should use same frequency basis as sequential (j_dim=64)
    # but with different positions
    print("  ✓ Spectral unified mode passed!")

    # ==================== Test 4: DFS Mode ====================
    print("\n[Test 4] DFS Mode (Tree-ordered positions)")
    print("-" * 50)

    rope_dfs = MotionWanRotaryPosEmbed(
        attention_head_dim=attention_head_dim,
        patch_size=patch_size,
        max_seq_len=max_seq_len,
        joint_pos_mode="dfs",
    )

    freqs_cos_dfs, freqs_sin_dfs = rope_dfs(hidden_states)

    print(f"  Output freqs_cos shape: {freqs_cos_dfs.shape}")
    assert freqs_cos_dfs.shape == expected_shape, f"Shape mismatch!"
    assert freqs_sin_dfs.shape == expected_shape, f"Shape mismatch!"

    # Verify translation token identity
    token0_cos_dfs = freqs_cos_dfs[0, 0, 0, t_dim:]
    token0_sin_dfs = freqs_sin_dfs[0, 0, 0, t_dim:]
    assert torch.allclose(token0_cos_dfs, torch.ones_like(token0_cos_dfs), atol=1e-5)
    assert torch.allclose(token0_sin_dfs, torch.zeros_like(token0_sin_dfs), atol=1e-5)
    print("  ✓ Position 0 (translation): identity RoPE correct!")

    # DFS mode should differ from both sequential and spectral
    assert not torch.allclose(freqs_cos_dfs, freqs_cos, atol=1e-3)
    assert not torch.allclose(freqs_cos_dfs, freqs_cos_sp, atol=1e-3)
    print("  ✓ DFS mode differs from sequential and spectral!")
    print("  ✓ DFS mode passed!")

    # ==================== Test 5: Frequency Validation ====================
    print("\n[Test 5] Frequency Value Validation (all modes)")
    print("-" * 50)

    for name, cos, sin in [
        ("sequential", freqs_cos, freqs_sin),
        ("spectral", freqs_cos_sp, freqs_sin_sp),
        ("spectral_unified", freqs_cos_su, freqs_sin_su),
        ("dfs", freqs_cos_dfs, freqs_sin_dfs),
    ]:
        assert cos.min() >= -1.0 and cos.max() <= 1.0, f"{name}: cos out of range!"
        assert sin.min() >= -1.0 and sin.max() <= 1.0, f"{name}: sin out of range!"
        identity_check = cos**2 + sin**2
        identity_error = (identity_check - 1.0).abs().max().item()
        print(f"  {name}: cos²+sin² identity error = {identity_error:.2e}")
        assert identity_error < 1e-4, f"{name}: Pythagorean identity not satisfied!"

    print("  ✓ All modes pass frequency validation!")

    # ==================== Test 6: Spectral Coordinate Properties ====================
    print("\n[Test 6] Spectral Coordinate Properties")
    print("-" * 50)

    coords = _compute_spectral_coords(num_joints=22, num_modes=4)
    print(f"  Spectral coords shape: {coords.shape}")
    print(f"    → 22 body joints, 4 spectral modes")
    print(f"  Coord range: [{coords.min():.4f}, {coords.max():.4f}]")

    # Parent-child pairs should have similar coordinates
    # L_Hip (1) -> L_Knee (4), distance should be small
    dist_hip_knee = np.linalg.norm(coords[1] - coords[4])
    # L_Foot (10) -> R_Wrist (21), distance should be large
    dist_foot_wrist = np.linalg.norm(coords[10] - coords[21])
    print(f"  L_Hip->L_Knee spectral distance: {dist_hip_knee:.4f}")
    print(f"  L_Foot->R_Wrist spectral distance: {dist_foot_wrist:.4f}")
    assert dist_hip_knee < dist_foot_wrist, \
        "Parent-child should be closer than distant joints!"
    print("  ✓ Kinematic proximity preserved in spectral coords!")

    # ==================== Test 7: DFS Ordering ====================
    print("\n[Test 7] DFS Ordering Properties (22 body joints)")
    print("-" * 50)

    dfs_order = _compute_dfs_ordering(num_joints=22)
    print(f"  DFS order (first 10): {dfs_order[:10].astype(int).tolist()}")
    # Root should be first
    assert dfs_order[0] == 0, "Root (pelvis) should be visited first!"
    # All values should be unique 0..21
    assert len(set(dfs_order.tolist())) == 22, "DFS should visit all 22 joints exactly once!"
    print("  ✓ DFS ordering is valid!")

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
    print("\nKey validation points:")
    print("  ✓ 23-joint architecture: 1 translation + 22 body joints")
    print("  ✓ Translation token (position 0): identity RoPE across all modes")
    print("  ✓ Body joints (positions 1-22): mode-specific RoPE")
    print("  ✓ All 4 RoPE modes (sequential, spectral, spectral_unified, dfs)")
    print("  ✓ Frequency ranges and Pythagorean identity validation")
    print("  ✓ Kinematic tree structure preservation")
