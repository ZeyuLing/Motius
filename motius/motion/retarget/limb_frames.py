"""Resolve SMPL limb twist from three-point bend planes and hinge axes.

These are anatomical priors, not recovered ground-truth rotations. For straight
limbs the plane is unobservable, so interpolate twist between observed bends in
an independently transported frame instead of following a noisy cross product.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

# Upper joint: (hinge joint, distal joint, positive flexion axis in SMPL rest).
_CHAINS = {
    1: (4, 7, (1.0, 0.0, 0.0)),
    2: (5, 8, (1.0, 0.0, 0.0)),
    16: (18, 20, (0.0, -1.0, 0.0)),
    17: (19, 21, (0.0, 1.0, 0.0)),
}


def _unit(vector):
    return vector / np.maximum(np.linalg.norm(vector, axis=-1, keepdims=True), 1e-12)


def _flexion(upper, lower, axis, cosine):
    """Positive hinge angle matching the observed inter-segment angle."""
    constant = np.dot(upper, axis) * np.dot(lower, axis)
    a = np.dot(upper, lower) - constant
    b = np.dot(upper, np.cross(axis, lower))
    radius = np.hypot(a, b)
    if radius < 1e-8:
        return 0.0
    phase = np.arctan2(b, a)
    delta = np.arccos(np.clip((cosine - constant) / radius, -1, 1))
    candidates = [(phase + delta) % (2 * np.pi), (phase - delta) % (2 * np.pi)]
    candidates = [angle for angle in candidates if angle <= np.pi] + [0.0, np.pi]
    return min(
        candidates,
        key=lambda angle: abs(
            a * np.cos(angle) + b * np.sin(angle) + constant - cosine
        ),
    )


def limb_orientation_guides(joints, rest, parents):
    """Return per-frame orthogonal guide vectors for the four SMPL limbs.

    No guide is generated if the entire clip has an unobservable bend plane.
    Each guide is perpendicular to the outgoing bone so it fixes twist without
    changing the observed child direction or canonical bone lengths.
    """
    offsets = rest.copy()
    offsets[1:] -= rest[parents[1:]]
    guides = {}
    for upper_joint, (hinge, tip, axis_tuple) in _CHAINS.items():
        axis = np.asarray(axis_tuple)
        upper_rest = _unit(offsets[hinge])
        lower_rest = _unit(offsets[tip])
        upper = joints[:, hinge] - joints[:, upper_joint]
        lower = joints[:, tip] - joints[:, hinge]
        valid = (np.linalg.norm(upper, axis=-1) > 1e-8) & (
            np.linalg.norm(lower, axis=-1) > 1e-8
        )
        if not valid.all():
            continue
        upper, lower = _unit(upper), _unit(lower)
        cross = np.cross(upper, lower)
        observed = np.linalg.norm(cross, axis=-1) > np.sin(np.deg2rad(10.0))
        if not observed.any():
            continue
        normal = _unit(cross)
        # A smoothly transported coordinate chart avoids +/-pi discontinuities
        # and permits interpolation across fully extended or folded frames.
        basis = np.zeros_like(normal)
        seed = np.eye(3)[np.argmin(np.abs(upper[0]))]
        basis[0] = _unit(np.cross(upper[0], seed))
        for t in range(1, len(joints)):
            swing = Rotation.align_vectors(upper[t : t + 1], upper[t - 1 : t])[0]
            basis[t] = swing.apply(basis[t - 1])
        angles = np.arctan2(
            np.einsum("ti,ti->t", upper, np.cross(basis, normal)),
            np.einsum("ti,ti->t", basis, normal),
        )
        times = np.arange(len(joints))
        angles = np.interp(times, times[observed], np.unwrap(angles[observed]))
        normals = Rotation.from_rotvec(upper * angles[:, None]).apply(basis)
        for t in range(len(joints)):
            theta = _flexion(upper_rest, lower_rest, axis, np.dot(upper[t], lower[t]))
            # The rest-plane normal itself degenerates at zero flexion.
            if not observed[t]:
                theta = max(theta, np.deg2rad(10.0))
            expected_lower = Rotation.from_rotvec(axis * theta).apply(lower_rest)
            rest_normal = np.cross(upper_rest, expected_lower)
            if np.linalg.norm(rest_normal) < 1e-8:
                continue
            rest_normal = _unit(rest_normal)
            world = Rotation.align_vectors(
                np.stack([upper[t], normals[t]]),
                np.stack([upper_rest, rest_normal]),
            )[0].as_matrix()
            guides[t, upper_joint] = (rest_normal, normals[t])
            hinge_world_axis = world @ axis
            source = axis - lower_rest * np.dot(axis, lower_rest)
            target = hinge_world_axis - lower[t] * np.dot(hinge_world_axis, lower[t])
            if min(np.linalg.norm(source), np.linalg.norm(target)) > 1e-8:
                guides[t, hinge] = (_unit(source), _unit(target))
    # A single spine child leaves axial twist unobserved locally. The lateral
    # hip/shoulder axes observe the total torso twist. Share it over the three
    # spine joints instead of concentrating it at the branching chest joint.
    rest_hip = _unit(rest[1] - rest[2])
    rest_shoulder = _unit(rest[16] - rest[17])
    rest_turn = Rotation.align_vectors(rest_shoulder[None], rest_hip[None])[
        0
    ].as_rotvec()
    for t, frame in enumerate(joints):
        hip = frame[1] - frame[2]
        shoulder = frame[16] - frame[17]
        if min(np.linalg.norm(hip), np.linalg.norm(shoulder)) < 1e-8:
            continue
        hip, shoulder = _unit(hip), _unit(shoulder)
        turn = Rotation.align_vectors(shoulder[None], hip[None])[0].as_rotvec()
        for joint, child, fraction in [(3, 6, 1 / 3), (6, 9, 2 / 3)]:
            source_bone = _unit(offsets[child])
            target_bone = _unit(frame[child] - frame[joint])
            source = Rotation.from_rotvec(rest_turn * fraction).apply(rest_hip)
            target = Rotation.from_rotvec(turn * fraction).apply(hip)
            source -= source_bone * np.dot(source, source_bone)
            target -= target_bone * np.dot(target, target_bone)
            if min(np.linalg.norm(source), np.linalg.norm(target)) > 1e-8:
                guides[t, joint] = (_unit(source), _unit(target))
    return guides
