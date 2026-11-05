# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Selected SONIC G1 conversions components; see licenses/sonic."""
import enum
import os
import warnings
import torch

USE_JIT_TORCH_TRANSFORM = os.getenv("USE_JIT_TORCH_TRANSFORM", "1").lower() in (
    "1",
    "true",
    "yes",
)


def conditional_jit_script(func):
    """Conditionally apply torch.jit.script based on USE_JIT_TORCH_TRANSFORM env var"""
    if USE_JIT_TORCH_TRANSFORM:
        return torch.jit.script(func)
    return func


class QuaternionCoeffOrder(enum.Enum):
    XYZW = "xyzw"
    WXYZ = "wxyz"


@conditional_jit_script
def torch_safe_atan2(y, x, eps: float = 1e-06):
    y = y.clone()
    if len(y.shape) == 0:
        if y.abs() < eps and x.abs() < eps:
            y += eps
    else:
        y[(y.abs() < eps) & (x.abs() < eps)] += eps
    return torch.atan2(y, x)


def safe_zero_division(
    numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-06
) -> torch.Tensor:
    denominator = denominator.clone()
    if len(denominator.shape) == 0:
        if denominator.abs() < eps:
            denominator += eps
    else:
        denominator = torch.where(
            denominator.abs() < eps, denominator + eps, denominator
        )
    return numerator / denominator


@conditional_jit_script
def quaternion_to_angle_axis(
    quaternion: torch.Tensor,
    eps: float = 1e-06,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    """Convert quaternion vector to angle axis of rotation.

    The quaternion should be in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        quaternion: tensor with quaternions.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with angle axis of rotation.

    Shape:
        - Input: :math:`(*, 4)` where `*` means, any number of dimensions
        - Output: :math:`(*, 3)`

    Example:
        >>> quaternion = torch.rand(2, 4)  # Nx4
        >>> angle_axis = quaternion_to_angle_axis(quaternion)  # Nx3
    """
    if not quaternion.shape[-1] == 4:
        raise ValueError(
            f"Input must be a tensor of shape Nx4 or 4. Got {quaternion.shape}"
        )
    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )
    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and will be removed after > 0.6. Please use `QuaternionCoeffOrder.WXYZ` instead."
        )
    q1: torch.Tensor = torch.tensor([])
    q2: torch.Tensor = torch.tensor([])
    q3: torch.Tensor = torch.tensor([])
    cos_theta: torch.Tensor = torch.tensor([])
    if order == QuaternionCoeffOrder.XYZW:
        q1 = quaternion[..., 0]
        q2 = quaternion[..., 1]
        q3 = quaternion[..., 2]
        cos_theta = quaternion[..., 3]
    else:
        cos_theta = quaternion[..., 0]
        q1 = quaternion[..., 1]
        q2 = quaternion[..., 2]
        q3 = quaternion[..., 3]
    sin_squared_theta: torch.Tensor = q1 * q1 + q2 * q2 + q3 * q3
    sin_theta: torch.Tensor = torch.sqrt(sin_squared_theta.clamp_min(eps))
    two_theta: torch.Tensor = 2.0 * torch.where(
        cos_theta < 0.0,
        torch_safe_atan2(-sin_theta, -cos_theta),
        torch_safe_atan2(sin_theta, cos_theta),
    )
    k_pos: torch.Tensor = safe_zero_division(two_theta, sin_theta, eps)
    k_neg: torch.Tensor = 2.0 * torch.ones_like(sin_theta)
    k: torch.Tensor = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)
    angle_axis: torch.Tensor = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


@conditional_jit_script
def angle_axis_to_quaternion(
    angle_axis: torch.Tensor,
    eps: float = 1e-06,
    order: QuaternionCoeffOrder = QuaternionCoeffOrder.WXYZ,
) -> torch.Tensor:
    """Convert an angle axis to a quaternion.

    The quaternion vector has components in (x, y, z, w) or (w, x, y, z) format.

    Adapted from ceres C++ library: ceres-solver/include/ceres/rotation.h

    Args:
        angle_axis: tensor with angle axis.
        order: quaternion coefficient order. Note: 'xyzw' will be deprecated in favor of 'wxyz'.

    Return:
        tensor with quaternion.

    Shape:
        - Input: :math:`(*, 3)` where `*` means, any number of dimensions
        - Output: :math:`(*, 4)`

    Example:
        >>> angle_axis = torch.rand(2, 3)  # Nx3
        >>> quaternion = angle_axis_to_quaternion(angle_axis, order=QuaternionCoeffOrder.WXYZ)  # Nx4
    """
    if not angle_axis.shape[-1] == 3:
        raise ValueError(
            f"Input must be a tensor of shape Nx3 or 3. Got {angle_axis.shape}"
        )
    if not torch.jit.is_scripting():
        if order.name not in QuaternionCoeffOrder.__members__.keys():
            raise ValueError(
                f"order must be one of {QuaternionCoeffOrder.__members__.keys()}"
            )
    if order == QuaternionCoeffOrder.XYZW:
        warnings.warn(
            "`XYZW` quaternion coefficient order is deprecated and will be removed after > 0.6. Please use `QuaternionCoeffOrder.WXYZ` instead."
        )
    a0: torch.Tensor = angle_axis[..., 0:1]
    a1: torch.Tensor = angle_axis[..., 1:2]
    a2: torch.Tensor = angle_axis[..., 2:3]
    theta_squared: torch.Tensor = a0 * a0 + a1 * a1 + a2 * a2
    theta: torch.Tensor = torch.sqrt(theta_squared.clamp_min(eps))
    half_theta: torch.Tensor = theta * 0.5
    mask: torch.Tensor = theta_squared > 0.0
    ones: torch.Tensor = torch.ones_like(half_theta)
    k_neg: torch.Tensor = 0.5 * ones
    k_pos: torch.Tensor = safe_zero_division(torch.sin(half_theta), theta, eps)
    k: torch.Tensor = torch.where(mask, k_pos, k_neg)
    w: torch.Tensor = torch.where(mask, torch.cos(half_theta), ones)
    quaternion: torch.Tensor = torch.zeros(
        size=angle_axis.shape[:-1] + (4,),
        dtype=angle_axis.dtype,
        device=angle_axis.device,
    )
    if order == QuaternionCoeffOrder.XYZW:
        quaternion[..., 0:1] = a0 * k
        quaternion[..., 1:2] = a1 * k
        quaternion[..., 2:3] = a2 * k
        quaternion[..., 3:4] = w
    else:
        quaternion[..., 1:2] = a0 * k
        quaternion[..., 2:3] = a1 * k
        quaternion[..., 3:4] = a2 * k
        quaternion[..., 0:1] = w
    return quaternion
