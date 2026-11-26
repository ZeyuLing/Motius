"""Regularized position fitting for noisy HumanML body landmarks.

Independent short-bone alignment can fold adjacent vertebrae in opposite
directions. Fit positions over the entire chain instead, with a stronger
neutral-pose prior on the spine and neck. This is an approximate pose estimate,
not recovery of unobserved twist or a guarantee of anatomical validity.
"""

import numpy as np
import torch

from motius.motion.representation.rotation import axis_angle_to_matrix


def robust_landmark_loss(residual: torch.Tensor) -> torch.Tensor:
    """Use a quadratic loss near a good fit and reduce the pull of outliers."""

    squared_distance = residual.square().sum(-1)
    radius = 0.10  # metres
    return (2 * squared_distance / ((1 + squared_distance / radius**2).sqrt() + 1)).mean()


def rotation_temporal_loss(rotations: torch.Tensor) -> torch.Tensor:
    """Penalize rotation changes and acceleration without axis-angle wrapping."""

    if len(rotations) < 2:
        return rotations.new_zeros(())
    velocity = rotations[1:] @ rotations[:-1].transpose(-1, -2)
    # Half the squared Frobenius distance approximates squared radians at zero.
    identity = torch.eye(3, dtype=rotations.dtype, device=rotations.device)
    loss = 0.5 * (velocity - identity).square().sum((-1, -2)).mean()
    if len(velocity) > 1:
        loss = loss + 0.5 * velocity.diff(dim=0).square().sum((-1, -2)).mean()
    return loss


@torch.inference_mode(False)
@torch.enable_grad()
def fit_anatomical_pose(
    joints, rest, parents, initial, *, iterations=300, prior_weight=1e-3, device="cpu"
):
    """Return local rotations; preserve the input root trajectory exactly.

    The torso/neck prior is 30 times the other body joints' prior. Relative
    rotation corrections around the IK initialization keep the optimization
    independent of the axis-angle representation's branch cut.
    """
    if iterations < 0 or prior_weight < 0 or not np.isfinite(prior_weight):
        raise ValueError("iterations and finite prior_weight must be nonnegative")
    if iterations == 0 or len(joints) == 0:
        return np.asarray(initial).copy()
    parents = np.asarray(parents, dtype=np.int64)[:22]
    rest = torch.as_tensor(np.asarray(rest)[:22], dtype=torch.float32, device=device)
    offsets = rest - rest[torch.as_tensor(parents.clip(0), device=device)]
    target = torch.as_tensor(
        np.asarray(joints) - np.asarray(joints)[:, :1],
        dtype=torch.float32,
        device=device,
    )
    start = torch.as_tensor(np.asarray(initial), dtype=torch.float32, device=device)
    correction = torch.zeros_like(target, requires_grad=True)
    identity = torch.eye(3, device=device)
    weights = torch.ones(22, device=device)
    weights[[3, 6, 9, 12]] = 30
    limbs = [1, 2, 4, 5, 16, 17, 18, 19]
    optimizer = torch.optim.Adam([correction], lr=0.025)
    for _ in range(iterations):
        delta = axis_angle_to_matrix(correction)
        rotations = delta @ start
        world = [rotations[:, 0]]
        positions = [torch.zeros_like(correction[:, 0])]
        for joint in range(1, 22):
            parent = int(parents[joint])
            positions.append(
                positions[parent] + (world[parent] @ offsets[joint, :, None])[..., 0]
            )
            world.append(world[parent] @ rotations[:, joint])
        fitted = torch.stack(positions, dim=1)
        loss = robust_landmark_loss(fitted - target)
        neutral = 0.5 * (rotations[:, 1:] - identity).square().sum((-2, -1))
        loss = loss + prior_weight / 3 * (neutral * weights[None, 1:]).mean()
        loss = loss + 1e-4 * correction[:, limbs].square().mean()
        loss = loss + 0.005 * rotation_temporal_loss(rotations)
        loss = loss + 0.005 * rotation_temporal_loss(torch.stack(world, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return (axis_angle_to_matrix(correction.detach()) @ start).cpu().numpy()
