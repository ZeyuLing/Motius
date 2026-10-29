# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """Calculates the rotation matrices for a batch of rotation vectors
    Parameters
    ----------
    rot_vecs: torch.tensor Nx3
        array of N axis-angle vectors
    Returns
    -------
    R: torch.tensor Nx3x3
        The rotation matrices for the given axis-angle parameters
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    theta = torch.norm(rot_vecs, dim=1, keepdim=True)

    # K from unnormalized rot_vecs
    rx, ry, rz = torch.split(rot_vecs, 1, dim=1)
    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1).view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)

    # Coefficients with Taylor expansion for numerical stability
    c1 = torch.ones_like(theta)
    c2 = torch.ones_like(theta) * 0.5

    mask = theta > 1e-4
    if mask.any():
        t = theta[mask]
        c1[mask] = torch.sin(t) / t
        c2[mask] = (1 - torch.cos(t)) / (t * t)

    rot_mat = ident + c1.unsqueeze(-1) * K + c2.unsqueeze(-1) * torch.bmm(K, K)
    return rot_mat


def lbs(bind_shape, skinning_weights, target_pose_world):
    """
    Linear Blend Skinning (PyTorch).

    Supported:
      A) target batched only:
         bind_shape:        (V, 3)
         skinning_weights:  (V, J)       # STRICT
         target_pose_world: (..., J, 4, 4)

      B) all batched with same leading dims:
         bind_shape:        (..., V, 3)
         skinning_weights:  (V, J)       # STRICT
         target_pose_world: (..., J, 4, 4)

      C) fully unbatched:
         bind_shape:        (V, 3)
         skinning_weights:  (V, J)
         target_pose_world: (J, 4, 4)

    Returns:
      blended: (..., V, 3)  # mirrors the active batch dims (from A or B)
    """
    # ---- Shape checks ----
    if bind_shape.ndim < 2 or bind_shape.shape[-1] != 3:
        raise ValueError(f"`bind_shape` must end with (V,3); got {bind_shape.shape}.")
    V = bind_shape.shape[-2]
    bind_batched = bind_shape.ndim > 2
    bind_batch = bind_shape.shape[:-2] if bind_batched else ()

    if target_pose_world.ndim < 3 or target_pose_world.shape[-2:] != (4, 4):
        raise ValueError(
            f"`target_pose_world` must end with (J,4,4); got {target_pose_world.shape}."
        )
    J = target_pose_world.shape[-3]
    tgt_batched = target_pose_world.ndim > 3
    tgt_batch = target_pose_world.shape[:-3] if tgt_batched else ()

    if skinning_weights.shape != (V, J):
        raise ValueError(
            f"`skinning_weights` must be (V,J) with V={V}, J={J}; got {skinning_weights.shape}."
        )

    # ---- Mode resolution ----
    if bind_batched:
        if not tgt_batched:
            raise ValueError(
                "When bind_shape/inverse_bind_pose are batched, target_pose_world must also be batched."
            )
        if tgt_batch != bind_batch:
            raise ValueError(
                f"Batch dims must match across bind/inverse/target; got {bind_batch} vs {tgt_batch}."
            )

    # ---- LBS math ----
    T = target_pose_world
    R = T[..., :3, :3]  # (..., J, 3, 3)
    t = T[..., :3, 3]  # (..., J, 3)

    tv = torch.einsum("...jmk,...vk->...jvm", R, bind_shape) + t[..., None, :]

    blended = torch.einsum("vj,...jvm->...vm", skinning_weights, tv)

    return blended
