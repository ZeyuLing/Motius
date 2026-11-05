# SPDX-License-Identifier: GPL-3.0-or-later
"""Contact-aware grounding and analytic two-bone IK, in meters / Z-up.

This is a post-retarget solver. It does not change the saved neural prediction.
Airborne frames keep their original vertical trajectory plus an interpolated
calibration offset; feet are pinned only during detected support intervals.
"""
import numpy as np


def rotate_between(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    a /= max(np.linalg.norm(a), 1e-12)
    b /= max(np.linalg.norm(b), 1e-12)
    v = np.cross(a, b)
    c = np.clip(a @ b, -1., 1.)
    s = np.linalg.norm(v)
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, np.eye(3)[np.argmin(np.abs(a))])
        axis /= np.linalg.norm(axis)
        return 2*np.outer(axis, axis)-np.eye(3)
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3)+k+k@k*((1-c)/(s*s))


def segments(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def forward(rest, parents, rotations, root):
    result = np.empty_like(rest, dtype=float)
    result[0] = root
    for i in range(1, len(rest)):
        p = parents[i]
        result[i] = result[p]+rotations[p] @ (rest[i]-rest[p])
    return result


def solve(joints, rotations, rest, parents, soles, sole_heights, body_min,
          chains, fps, floor=0., tolerance=.10):
    joints, rotations = np.asarray(joints), np.asarray(rotations)
    count = len(joints)
    if count < 3 or soles.shape != (count, 2, 3):
        raise ValueError('Grounding needs at least three frames and two feet')
    baseline = float(np.percentile(sole_heights.min(1), 12))
    velocity = np.gradient(soles, axis=0)*fps
    support = ((sole_heights-baseline < tolerance) &
               (np.abs(velocity[:, :, 2]) < 1.2) &
               (np.linalg.norm(velocity[:, :, :2], axis=-1) < 1.8))
    weights = np.zeros((count, 2))
    anchors = soles.copy()
    for foot in range(2):
        for start, end in segments(support[:, foot]):
            if end-start < 3:
                support[start:end, foot] = False
                continue
            anchors[start:end, foot, :2] = np.median(soles[start:min(start+3, end), foot, :2], axis=0)
            weights[start:end, foot] = 1
            ramp = min(2, (end-start-1)//2)
            for edge in range(ramp):
                value = (edge+1)/(ramp+1)
                weights[start+edge, foot] = value
                weights[end-edge-1, foot] = value
    active = support.any(1)
    if not active.any():
        raise ValueError('No reliable support contacts; choose a clip with grounded feet')
    delta = np.zeros(count)
    for t in np.flatnonzero(active):
        delta[t] = floor-np.mean(sole_heights[t, support[t]])
    delta = np.interp(np.arange(count), np.flatnonzero(active), delta[active])
    delta = np.convolve(np.pad(delta, (2, 2), mode='edge'), [1, 2, 3, 2, 1], 'valid')/9
    # This protects knees/hands during floor poses as well as the feet.
    delta = np.maximum(delta, floor-body_min)
    result_joints, result_rotations = joints.copy(), rotations.copy()
    residuals = []
    for t in range(count):
        root = joints[t, 0].copy()
        root[2] += delta[t]
        targets = []
        for f, (hip, knee, ankle) in enumerate(chains):
            correction = anchors[t, f]-soles[t, f]
            correction[2] = floor-sole_heights[t, f]
            target = joints[t, ankle]+[0, 0, delta[t]]
            target += weights[t, f]*(correction-[0, 0, delta[t]])
            # A fast swing / landing foot may not be a support contact, but it
            # must still collide with the floor without being pinned in XY.
            target[2] = max(target[2], joints[t, ankle, 2]+floor-sole_heights[t, f])
            targets.append(target)
        # Bring the pelvis within reach rather than scaling a leg or snapping
        # the foot away from its planted target.
        for _ in range(3):
            current = forward(rest, parents, rotations[t], root)
            shifts = []
            for f, (hip, knee, ankle) in enumerate(chains):
                if weights[t, f] < .99:
                    continue
                length = np.linalg.norm(rest[knee]-rest[hip])+np.linalg.norm(rest[ankle]-rest[knee])
                d = targets[f]-current[hip]
                distance = np.linalg.norm(d)
                if distance > length-.001:
                    shifts.append(d/max(distance, 1e-9)*(distance-length+.001))
            if not shifts:
                break
            root += np.mean(shifts, axis=0)
        rs = rotations[t].copy()
        for f, (hip, knee, ankle) in enumerate(chains):
            current = forward(rest, parents, rs, root)
            h, k, a = current[[hip, knee, ankle]]
            if np.linalg.norm(targets[f]-a) < 1e-7:
                continue
            l1, l2 = np.linalg.norm(k-h), np.linalg.norm(a-k)
            direction = targets[f]-h
            distance = np.clip(np.linalg.norm(direction), abs(l1-l2)+1e-5, l1+l2-1e-5)
            u = direction/max(np.linalg.norm(direction), 1e-9)
            bend = k-h-u*np.dot(k-h, u)
            if np.linalg.norm(bend) < 1e-7:
                bend = np.cross(u, [1., 0., 0.])
                if np.linalg.norm(bend) < 1e-7:
                    bend = np.cross(u, [0., 1., 0.])
            bend /= np.linalg.norm(bend)
            along = (l1*l1+distance*distance-l2*l2)/(2*distance)
            new_knee = h+u*along+bend*np.sqrt(max(0, l1*l1-along*along))
            rs[hip] = rotate_between(k-h, new_knee-h) @ rs[hip]
            rs[knee] = rotate_between(a-k, targets[f]-new_knee) @ rs[knee]
        result_rotations[t] = rs
        result_joints[t] = forward(rest, parents, rs, root)
        for f, chain in enumerate(chains):
            if weights[t, f] >= .99:
                residuals.append(float(np.linalg.norm(result_joints[t, chain[2]]-targets[f])))
    return result_joints, result_rotations, dict(
        contacts=support, weights=weights, sole_anchors=anchors,
        estimated_original_floor=baseline, root_vertical_correction=delta,
        ik_max_residual_m=max(residuals, default=0.),
        airborne_frames=int((~active).sum()))
