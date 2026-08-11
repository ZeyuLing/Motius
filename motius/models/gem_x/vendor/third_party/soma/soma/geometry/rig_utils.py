# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import torch

from .transforms import SE3_inverse


def joint_world_to_local(joint_world_transforms, joint_parent_ids, return_inverse=False):
    """Convert world space joint transforms to local space (relative to parent).

    Args:
        joint_world_transforms: (J, M, M) or (B, J, M, M) where M=4 or 3
        joint_parent_ids: (J,) int array
    Returns:
        joint_local_transforms: same shape as joint_world_transforms
    """
    if not isinstance(joint_world_transforms, torch.Tensor):
        raise TypeError("joint_world_transforms must be a torch.Tensor.")
    added_batch = False
    if joint_world_transforms.ndim == 3:
        joint_world_transforms = joint_world_transforms[None, :, :, :]
        added_batch = True
    if joint_world_transforms.shape[-2:] == (3, 3):
        inverse_world_transforms = joint_world_transforms.swapaxes(-2, -1)
    elif joint_world_transforms.shape[-2:] == (4, 4):
        inverse_world_transforms = SE3_inverse(joint_world_transforms)
    else:
        raise ValueError(
            f"Expected joint_world_transforms to have shape (...,4,4) or (...,3,3); got {joint_world_transforms.shape}"
        )
    joint_local_transforms = inverse_world_transforms[:, joint_parent_ids] @ joint_world_transforms
    if return_inverse:
        return (
            joint_local_transforms[0] if added_batch else joint_local_transforms,
            inverse_world_transforms[0] if added_batch else inverse_world_transforms,
        )
    return joint_local_transforms[0] if added_batch else joint_local_transforms


def joint_local_to_world(joint_local_transforms, joint_parent_ids):
    """Convert local space joint transforms (relative to parent) to world space.

    Args:
        joint_local_transforms: (J, M, M) or (B, J, M, M) where M=4 or 3
        joint_parent_ids: (J,) int array or Python list[int] (list preferred for torch.export)
    Returns:
        joint_world_transforms: same shape as joint_local_transforms
    """
    if not isinstance(joint_local_transforms, torch.Tensor):
        raise TypeError("joint_local_transforms must be a torch.Tensor.")
    added_batch = False
    if joint_local_transforms.ndim == 3:
        joint_local_transforms = joint_local_transforms[None, :, :, :]
        added_batch = True

    B, J = joint_local_transforms.shape[:2]
    results = [joint_local_transforms[:, 0]]

    parent_ids_list = (
        joint_parent_ids
        if isinstance(joint_parent_ids, list)
        else joint_parent_ids.tolist()
        if hasattr(joint_parent_ids, "tolist")
        else list(joint_parent_ids)
    )

    for i, parent_id in enumerate(parent_ids_list[1:], 1):
        parent_world = results[parent_id]
        local = joint_local_transforms[:, i]
        results.append(parent_world @ local)

    joint_world_transforms = torch.stack(results, dim=1)
    return joint_world_transforms[0] if added_batch else joint_world_transforms


def compute_skeleton_levels(joint_parent_ids, device=None):
    """Group joints by tree depth for level-order forward kinematics.

    Args:
        joint_parent_ids: (J,) int array, tensor, or Python list of parent indices.
        device: torch device for the returned index tensors.  Defaults to the
            device of *joint_parent_ids* when it is a tensor, otherwise ``"cpu"``.

    Returns:
        List of ``(joint_ids, parent_ids)`` tensor pairs, one per depth level.
        Level 0 contains the root(s); levels 1+ contain joints whose parents
        are at strictly lower levels.
    """
    parent_ids = (
        joint_parent_ids.tolist() if hasattr(joint_parent_ids, "tolist") else list(joint_parent_ids)
    )
    num_joints = len(parent_ids)

    depth = [0] * num_joints
    for i in range(1, num_joints):
        depth[i] = depth[parent_ids[i]] + 1
    max_depth = max(depth) if num_joints > 0 else 0

    if device is None and hasattr(joint_parent_ids, "device"):
        device = joint_parent_ids.device

    levels = []
    for d in range(max_depth + 1):
        jids = [i for i in range(num_joints) if depth[i] == d]
        pids = [parent_ids[i] for i in jids]
        levels.append(
            (
                torch.tensor(jids, dtype=torch.long, device=device),
                torch.tensor(pids, dtype=torch.long, device=device),
            )
        )
    return levels


def joint_local_to_world_levelorder(joint_local_transforms, levels):
    """Batched level-order forward kinematics.

    Equivalent to :func:`joint_local_to_world` but processes all joints at the
    same tree depth in a single batched ``bmm``, reducing ~J sequential kernel
    launches to ~D (number of depth levels, typically 10-12 for humanoids).

    Args:
        joint_local_transforms: (J, M, M) or (B, J, M, M) where M = 3 or 4.
        levels: precomputed output of :func:`compute_skeleton_levels`.
    Returns:
        joint_world_transforms: same shape as *joint_local_transforms*.
    """
    if not isinstance(joint_local_transforms, torch.Tensor):
        raise TypeError("joint_local_transforms must be a torch.Tensor.")
    added_batch = False
    if joint_local_transforms.ndim == 3:
        joint_local_transforms = joint_local_transforms[None, :, :, :]
        added_batch = True

    B = joint_local_transforms.shape[0]
    M = joint_local_transforms.shape[-1]
    world = joint_local_transforms.clone()

    for joint_ids, parent_ids in levels[1:]:
        n = joint_ids.shape[0]
        world[:, joint_ids] = torch.bmm(
            world[:, parent_ids].reshape(B * n, M, M),
            joint_local_transforms[:, joint_ids].reshape(B * n, M, M),
        ).reshape(B, n, M, M)

    return world[0] if added_batch else world


def precompute_joint_orient(joint_orient, joint_parent_ids):
    """Precompute tensors for :func:`apply_joint_orient_local`.

    Args:
        joint_orient: (J, 3, 3) world-space orientation per joint.
        joint_parent_ids: (J,) int array or Python list of parent indices.

    Returns:
        ``(orient, orient_parent_T)`` -- both (J, 3, 3) tensors ready to be
        passed to :func:`apply_joint_orient_local`.
    """
    orient = joint_orient[..., :3, :3]
    orient_parent_T = orient[joint_parent_ids].transpose(-2, -1)
    return orient, orient_parent_T


def apply_joint_orient_local(local_rotations, orient, orient_parent_T):
    """Apply joint orient as a per-joint local operation (no FK loop).

    Mathematically equivalent to::

        world = joint_local_to_world(local_rotations, parent_ids)
        world = world @ orient
        local = joint_world_to_local(world, parent_ids)

    but computed as a single parallel matmul:

        R_out[j] = orient_parent_T[j] @ R_in[j] @ orient[j]

    Args:
        local_rotations: (B, J, 3, 3) local-space rotation matrices.
        orient: (J, 3, 3) from :func:`precompute_joint_orient`.
        orient_parent_T: (J, 3, 3) from :func:`precompute_joint_orient`.

    Returns:
        (B, J, 3, 3) oriented local rotations.
    """
    return orient_parent_T[None] @ local_rotations @ orient[None]


def remove_joint_orient_local(local_rotations, orient, orient_parent_T):
    """Remove joint orient — inverse of :func:`apply_joint_orient_local`.

    Converts absolute local rotations (with orient baked in) back to
    T-pose-relative rotations::

        R_rel[j] = orient_parent_T[j]^T @ R_abs[j] @ orient[j]^T

    This is used to convert PoseInversion output (absolute) into the
    relative convention expected by ``SOMALayer.pose(absolute_pose=False)``.

    Args:
        local_rotations: (B, J, 3, 3) absolute local rotations.
        orient: (J, 3, 3) from :func:`precompute_joint_orient`.
        orient_parent_T: (J, 3, 3) from :func:`precompute_joint_orient`.

    Returns:
        (B, J, 3, 3) T-pose-relative rotations.
    """
    orient_parent = orient_parent_T.transpose(-2, -1)  # (J, 3, 3)
    orient_T = orient.transpose(-2, -1)  # (J, 3, 3)
    return orient_parent[None] @ local_rotations @ orient_T[None]


def get_joint_children_ids(joint_parent_ids):
    """Given a list of joint parent IDs, return a list of lists of joint children IDs."""
    parent_ids = (
        joint_parent_ids.tolist() if hasattr(joint_parent_ids, "tolist") else list(joint_parent_ids)
    )
    joint_children_ids = [[] for _ in range(len(parent_ids))]
    for i in range(1, len(parent_ids)):
        joint_children_ids[parent_ids[i]].append(i)
    return joint_children_ids


def get_joint_descendents(joint_parent_ids, joint_id):
    """Given a list of joint parent IDs, return a list of all descendents of a given joint ID."""
    children = get_joint_children_ids(joint_parent_ids)
    descendents = []

    def _gather_descendents(joint_id):
        for child_id in children[joint_id]:
            descendents.append(child_id)
            _gather_descendents(child_id)

    _gather_descendents(joint_id)
    return descendents


def get_body_part_vertex_ids(
    skinning_weights, joint_parent_ids, root_joint_id, include_root=True, weight_threshold=0.01
):
    """Get the vertex IDs influenced by a body part defined by a root joint and its descendents.

    Args:
        skinning_weights: (V, J) array of skinning weights
        joint_parent_ids: (J,) int array of joint parent indices
        root_joint_id: int, joint ID of the body part root
        include_root: bool, whether to include vertices influenced by the root joint itself
        weight_threshold: float, minimum skinning weight to consider a vertex influenced by a joint
    Returns:
        vertex_ids: list of int, vertex IDs influenced by the body part
    """
    if not isinstance(skinning_weights, torch.Tensor):
        raise TypeError("skinning_weights must be a torch.Tensor.")
    descendents = get_joint_descendents(joint_parent_ids, root_joint_id)
    if include_root:
        descendents = [root_joint_id] + descendents
    vertex_mask = torch.zeros(
        skinning_weights.shape[0], dtype=torch.bool, device=skinning_weights.device
    )
    for joint_id in descendents:
        vertex_mask |= skinning_weights[:, joint_id] > weight_threshold
    vertex_ids = torch.where(vertex_mask)[0].tolist()
    return vertex_ids


class PoseMirror_SOMA:
    """
    Mirrors world-space SOMA skeletal poses across the sagittal plane (YZ plane).

    This class reflects joint positions and orientations to swap the "Left" and "Right"
    sides of the body, effectively creating a mirror image of the pose. It handles the
    complexities of retaining a valid Right-Handed Coordinate System (RHCS) after reflection.

    Rig Assumptions:
        - **World Up:** +Y
        - **World Forward:** +Z
        - **Bone Axis:** Local +X axis points toward the child joint (along the bone).
        - **Naming Convention:** Symmetrical joints must start with "Left" or "Right".

    Mathematical Logic:
        1. **Global Reflection:** Reflects world positions across the YZ plane ($$x \to -x$$).
        2. **Chirality Correction:** Flips the local Y-axis to restore the coordinate system
           determinant to +1 (fixing the "inside-out" matrix caused by reflection).
        3. **Axis Realignment:** Applies 180-degree rotations to realign the bone vectors:
           - **Limbs:** Rotated 180° about Y to ensure Local +X points to the child.
           - **Center:** Rotated 180° about X to correct roll/twist on the mirror plane.
        4. **Swap:** Swaps the memory of Left and Right joint indices.

    Features:
        - Supports both NumPy arrays and PyTorch tensors (CPU/GPU).
        - Auto-detects backend and caches device-specific matrices.
        - Vectorized operations (no loops during execution).
    """

    def __init__(self, joint_names, root_name="Root"):
        """
        Args:
            joint_names (list[str]): List of joint names in the order of the pose array.
            root_name (str): Name of the root joint. Defaults to "Root".
        """
        self.num_joints = len(joint_names)
        self._cache = {}  # Cache for device-specific tensors

        # --- 1. Pre-compute Permutation Indices (Pure Python) ---
        # Map indices: Left -> Right, Right -> Left, Center -> Center
        perm_indices = list(range(self.num_joints))
        left_indices = []
        right_indices = []
        center_indices = []
        root_index = -1

        for i, name in enumerate(joint_names):
            if name == root_name:
                root_index = i
                # Root is processed separately, so we don't add it to center_indices
                # to avoid overwriting the fix later.
            elif name.startswith("Left"):
                try:
                    right_name = name.replace("Left", "Right")
                    j = joint_names.index(right_name)
                    perm_indices[i] = j
                    perm_indices[j] = i
                    left_indices.append(i)
                except ValueError:
                    pass
            elif name.startswith("Right"):
                right_indices.append(i)
            else:
                center_indices.append(i)

        self._perm_indices_raw = perm_indices

        # --- 2. Pre-compute Matrices (NumPy Base) ---

        # A. Global Reflection (Reflects World X)
        # Matrix: diag(-1, 1, 1, 1)
        self._global_ref_raw = np.diag([-1, 1, 1, 1]).astype(np.float32)

        # B. Local Adjustments (Applied on Right side)
        # Initialize with Identity
        self._local_adjust_raw = np.tile(np.eye(4), (self.num_joints, 1, 1)).astype(np.float32)

        # --- Matrix Logic ---

        # 1. Limbs: Combined = Ref_Y @ Rot_180y
        # Fixes chirality and ensures X-axis points to child
        limb_fix = np.diag([-1, -1, -1, 1]).astype(np.float32)
        self._local_adjust_raw[left_indices + right_indices] = limb_fix

        # 2. Center (Spine/Head): Combined = Ref_Y @ Rot_180x
        # Fixes chirality and roll/twist on the mirror plane
        center_fix = np.diag([1, 1, -1, 1]).astype(np.float32)
        self._local_adjust_raw[center_indices] = center_fix

        # 3. Root Fix
        # We want the root to face Forward (+Z), not Backward.
        # This matrix cancels the Global Reflection rotation effect while keeping position mirrored.
        # Global(-1, 1, 1) * Local(-1, 1, 1) = Identity(1, 1, 1)
        if root_index != -1:
            root_fix = np.diag([-1, 1, 1, 1]).astype(np.float32)
            self._local_adjust_raw[root_index] = root_fix
        else:
            print(
                f"Warning: Root joint '{root_name}' not found in joint list. Root rotation fix not applied."
            )

    def _get_backend_resources(self, sample_input):
        """
        Retrieves or creates the necessary tensors for the specific backend (Numpy/Torch),
        device (CPU/GPU), and dtype of the input.
        """
        is_torch = torch is not None and isinstance(sample_input, torch.Tensor)

        # Create a cache key
        if is_torch:
            key = ("torch", sample_input.device, sample_input.dtype)
        else:
            key = ("numpy", None, sample_input.dtype)

        if key in self._cache:
            return self._cache[key]

        # --- Cache Miss: Generate Resources ---
        if is_torch:
            xp = torch
            device = sample_input.device
            dtype = sample_input.dtype

            perm = torch.tensor(self._perm_indices_raw, device=device, dtype=torch.long)
            g_ref = torch.tensor(self._global_ref_raw, device=device, dtype=dtype)
            l_adj = torch.tensor(self._local_adjust_raw, device=device, dtype=dtype)
        else:
            xp = np
            dtype = sample_input.dtype

            perm = np.array(self._perm_indices_raw, dtype=int)
            g_ref = self._global_ref_raw.astype(dtype)
            l_adj = self._local_adjust_raw.astype(dtype)

        resources = (xp, perm, g_ref, l_adj)
        self._cache[key] = resources
        return resources

    def __call__(self, pose_world):
        """
        Mirrors the pose.

        Args:
            pose_world: (..., N, 4, 4) NumPy array or PyTorch Tensor.
                       Supports batch dimensions (B, N, 4, 4).
        Returns:
            Mirrored pose of same type/device/shape.
        """
        xp, perm, global_ref, local_adjust = self._get_backend_resources(pose_world)

        # 1. Permute (Swap Left/Right Memory)
        # [..., perm, :, :] handles arbitrary batch dimensions automatically
        pose_mirror = pose_world[..., perm, :, :]

        # 2. Global Reflection (Left-side Multiplication)
        # Broadcasts: (4,4) @ (..., 4, 4)
        pose_mirror = global_ref @ pose_mirror

        # 3. Local Adjustment (Right-side Multiplication)
        # Broadcasts: (..., N, 4, 4) @ (N, 4, 4)
        pose_mirror = pose_mirror @ local_adjust

        return pose_mirror


_DEFAULT_MHR_NEGATE_PARAMS = frozenset(
    [
        "root_tx",
        "root_ry",
        "root_rz",
        "spine0_rx_flexible",
        "spine0_ry_flexible",
        "spine_twist0",
        "spine_lean0",
        "spine1_rx_flexible",
        "spine1_ry_flexible",
        "spine_twist1",
        "spine_lean1",
        "spine2_rx_flexible",
        "spine2_ry_flexible",
        "spine3_rx_flexible",
        "spine3_ry_flexible",
        "neck_twist",
        "neck_lean",
        "head_twist",
        "head_lean",
    ]
)


class PoseMirror_MHR:
    """
    Mirrors MHR parameters across the sagittal plane (YZ plane).

    Handles both pose parameters (l_/r_ prefixed) and scale parameters
    (scale_l_/scale_r_ prefixed), swapping left/right counterparts and
    negating parameters that reverse sign under reflection.

    Features:
        - Supports both NumPy arrays and PyTorch tensors (CPU/GPU).
        - Pre-computes swap indices and sign vectors.
        - Vectorized execution.
    """

    def __init__(self, param_names, negate_params=_DEFAULT_MHR_NEGATE_PARAMS):
        """
        Args:
            param_names (list[str]): List of MHR parameter names (pose and/or scale).
            negate_params (set[str]): Set of parameter names to negate (multiply by -1).
        """
        self.num_params = len(param_names)
        self._cache = {}  # Cache for device-specific tensors

        # --- 1. Pre-compute Permutation and Signs (Pure Python) ---
        perm_indices = list(range(self.num_params))
        signs = [1.0] * self.num_params

        # Optimization: Create a lookup for names to indices
        name_to_idx = {name: i for i, name in enumerate(param_names)}

        for i, name in enumerate(param_names):
            # A. Handle Swapping (l_ <-> r_ and scale_l_ <-> scale_r_)
            mirror_name = None
            if name.startswith("scale_l_"):
                mirror_name = "scale_r_" + name[8:]
            elif name.startswith("scale_r_"):
                mirror_name = "scale_l_" + name[8:]
            elif name.startswith("l_"):
                mirror_name = "r_" + name[2:]
            elif name.startswith("r_"):
                mirror_name = "l_" + name[2:]
            if mirror_name is not None and mirror_name in name_to_idx:
                perm_indices[i] = name_to_idx[mirror_name]

            # B. Handle Negation
            # Note: Signs align with the parameter list order.
            # If we swap data into index 'i', we apply the sign for parameter 'i'.
            if name in negate_params:
                signs[i] = -1.0

        self._perm_indices_raw = perm_indices
        self._signs_raw = np.array(signs, dtype=np.float32)

    def _get_backend_resources(self, sample_input):
        """
        Retrieves cached tensors for the specific backend/device/dtype.
        """
        is_torch = torch is not None and isinstance(sample_input, torch.Tensor)

        if is_torch:
            key = ("torch", sample_input.device, sample_input.dtype)
        else:
            key = ("numpy", None, sample_input.dtype)

        if key in self._cache:
            return self._cache[key]

        # --- Cache Miss: Generate Resources ---
        if is_torch:
            xp = torch
            device = sample_input.device
            dtype = sample_input.dtype

            perm = torch.tensor(self._perm_indices_raw, device=device, dtype=torch.long)
            # Ensure signs match input dtype (e.g., float16 vs float32)
            signs = torch.tensor(self._signs_raw, device=device, dtype=dtype)
        else:
            xp = np
            dtype = sample_input.dtype

            perm = np.array(self._perm_indices_raw, dtype=int)
            signs = self._signs_raw.astype(dtype)

        resources = (xp, perm, signs)
        self._cache[key] = resources
        return resources

    def __call__(self, params):
        """
        Mirrors the MHR parameters.

        Args:
            params: (..., N) NumPy array or PyTorch Tensor.
        Returns:
            Mirrored parameters of same type/device/shape.
        """
        xp, perm, signs = self._get_backend_resources(params)

        # 1. Permute (Swap Left/Right Data)
        # [..., perm] handles arbitrary batch dimensions automatically
        mirrored_params = params[..., perm]

        # 2. Negate (Element-wise multiplication)
        # Broadcasts signs across batch dimensions
        mirrored_params = mirrored_params * signs

        return mirrored_params
