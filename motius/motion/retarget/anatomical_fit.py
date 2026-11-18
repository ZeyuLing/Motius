"""Regularized position fitting for noisy HumanML body landmarks.

Independent short-bone alignment can fold adjacent vertebrae in opposite
directions. Fit positions over the entire chain instead, with a stronger
neutral-pose prior on the spine and neck. This is an approximate pose estimate,
not recovery of unobserved twist or a guarantee of anatomical validity.
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from motius.motion.representation.rotation import axis_angle_to_matrix


def fit_anatomical_pose(
    joints, rest, parents, initial, *, iterations=300, prior_weight=1e-3, device="cpu"
):
    """Return local rotations; preserve the input root trajectory exactly.

    The loss uses metres and radians. The torso/neck prior is 30 times the
    other body joints' prior. Limb initialization and temporal acceleration
    constrain the remaining ambiguity. No body-model mesh is needed.
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
    angles = Rotation.from_matrix(np.asarray(initial).reshape(-1, 3, 3)).as_rotvec()
    with torch.enable_grad():
        pose = torch.tensor(
            angles.reshape(-1, 22, 3),
            dtype=torch.float32,
            device=device,
            requires_grad=True,
        )
        start = pose.detach().clone()
        weights = torch.ones(22, device=device)
        weights[[3, 6, 9, 12]] = 30
        limbs = [1, 2, 4, 5, 16, 17, 18, 19]
        optimizer = torch.optim.Adam([pose], lr=0.025)
        for _ in range(iterations):
            rotations = axis_angle_to_matrix(pose)
            world = [rotations[:, 0]]
            positions = [torch.zeros_like(pose[:, 0])]
            for joint in range(1, 22):
                parent = int(parents[joint])
                positions.append(
                    positions[parent]
                    + (world[parent] @ offsets[joint, :, None])[..., 0]
                )
                world.append(world[parent] @ rotations[:, joint])
            fitted = torch.stack(positions, dim=1)
            loss = ((fitted - target) ** 2).sum(-1).mean()
            loss = (
                loss
                + prior_weight * (pose[:, 1:] ** 2 * weights[None, 1:, None]).mean()
            )
            loss = loss + 1e-4 * ((pose[:, limbs] - start[:, limbs]) ** 2).mean()
            if len(pose) >= 3:
                loss = (
                    loss + 2e-3 * ((pose[2:] - 2 * pose[1:-1] + pose[:-2]) ** 2).mean()
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return axis_angle_to_matrix(pose.detach()).cpu().numpy()
