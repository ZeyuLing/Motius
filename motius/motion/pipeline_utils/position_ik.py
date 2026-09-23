"""Inference-only SMPL-22 IK. Positions and offsets use meters.

Analytic limb seeds followed, when necessary, by simultaneous position DLS.
Original targets are immutable. Always check the returned FK, including after
encoding/decoding, rather than inferring success from the iteration count.
"""

import torch
import torch.nn.functional as F

from motius.motion.skeleton.names import SMPL22_PARENTS
from motius.motion.skeleton.fk import differentiable_fk

NUM_JOINTS = 22
TWO_BONE_CHAINS = {7: (1, 4, 7), 8: (2, 5, 8), 10: (4, 7, 10),
                   11: (5, 8, 11), 20: (16, 18, 20), 21: (17, 19, 21)}
ONE_BONE_TARGETS = {1: 0, 2: 0, 3: 0, 4: 1, 5: 2}
GRADIENT_IK_ANCESTORS = {6: [0, 3], 9: [3, 6], 12: [6, 9], 15: [9, 12],
                       13: [6, 9], 14: [6, 9], 16: [9, 13], 17: [9, 14],
                       18: [13, 16], 19: [14, 17]}


def get_ik_strategy(joint_idx):
    if not 0 <= joint_idx < 22:
        raise ValueError(f'Invalid SMPL-22 joint: {joint_idx}')
    if joint_idx == 0:
        return 'root'
    if joint_idx in TWO_BONE_CHAINS:
        return 'two_bone'
    return 'one_bone' if joint_idx in ONE_BONE_TARGETS else 'gradient'


def ancestors(joint):
    result = []
    joint = SMPL22_PARENTS[joint]
    while joint >= 0:
        result.append(joint)
        joint = SMPL22_PARENTS[joint]
    return result


def solve_root_ik(translation, bone_offsets, target_pos):
    return target_pos - bone_offsets[0]


def _skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


def _rotation_exp(v):
    angle, skew = v.norm(dim=-1), _skew(v)
    eye = torch.eye(3, device=v.device, dtype=v.dtype)
    a = torch.sinc(angle / torch.pi)[..., None, None]
    b = (.5 * torch.sinc(angle / (2 * torch.pi)).square())[..., None, None]
    return eye + a * skew + b * (skew @ skew)


def _rotation_between_vectors(v_from, v_to):
    """Batched minimum swing, with a deterministic antiparallel axis."""
    source, target = F.normalize(v_from, dim=-1), F.normalize(v_to, dim=-1)
    cross = torch.cross(source, target, dim=-1)
    dot = (source * target).sum(-1).clamp(-1, 1)
    eye = torch.eye(3, device=source.device, dtype=source.dtype)
    sine = cross.norm(dim=-1)
    basis = F.one_hot(source.abs().argmin(-1), 3).to(source.dtype)
    fallback = F.normalize(torch.cross(source, basis, dim=-1), dim=-1)
    axis = torch.where((sine > 1e-12)[..., None], F.normalize(cross, dim=-1), fallback)
    result = _rotation_exp(axis * torch.atan2(sine, dot)[..., None])
    valid = (v_from.norm(dim=-1) > 1e-12) & (v_to.norm(dim=-1) > 1e-12)
    return torch.where(valid[..., None, None], result, eye)


@torch.no_grad()
def solve_two_bone_ik(local_rotmat, translation, bone_offsets, target_joint,
                      target_pos, num_iters=20):
    """Analytic two-bone solve; num_iters retained for compatibility.

    Reachable cues are never shrunk. Unreachable cues use the closest reach
    shell point internally, but must still be scored against the original cue.
    """
    base, mid, end = TWO_BONE_CHAINS[target_joint]
    result = local_rotmat.clone()
    pos, world = differentiable_fk(result, translation, bone_offsets)
    base_pos = pos[..., base, :]
    upper, lower = bone_offsets[mid].norm(), bone_offsets[end].norm()
    if min(float(upper), float(lower)) < 1e-12:
        return result
    to_target = target_pos - base_pos
    distance = to_target.norm(dim=-1)
    old_upper = pos[..., mid, :] - base_pos
    direction = torch.where((distance > 1e-12)[..., None], F.normalize(to_target, dim=-1),
                            F.normalize(old_upper, dim=-1))
    reach = distance.clamp(min=(upper - lower).abs(), max=upper + lower)
    along = (reach.square() + upper.square() - lower.square()) / (2 * reach.clamp_min(1e-12))
    height = (upper.square() - along.square()).clamp_min(0).sqrt()
    pole = old_upper - (old_upper * direction).sum(-1, keepdim=True) * direction
    basis = F.one_hot(direction.abs().argmin(-1), 3).to(direction.dtype)
    fallback = torch.cross(direction, basis, dim=-1)
    pole = F.normalize(torch.where((pole.norm(dim=-1) > 1e-10)[..., None], pole, fallback), dim=-1)
    desired_upper = along[..., None] * direction + height[..., None] * pole
    correction = _rotation_between_vectors(old_upper, desired_upper)
    parent_rot = world[..., SMPL22_PARENTS[base], :, :]
    result[..., base, :, :] = parent_rot.transpose(-1, -2) @ correction @ world[..., base, :, :]
    pos, world = differentiable_fk(result, translation, bone_offsets)
    correction = _rotation_between_vectors(pos[..., end, :] - pos[..., mid, :],
                                           base_pos + reach[..., None] * direction - pos[..., mid, :])
    result[..., mid, :, :] = world[..., base, :, :].transpose(-1, -2) @ correction @ world[..., mid, :, :]
    return result


def solve_one_bone_ik(local_rotmat, translation, bone_offsets, target_joint, target_pos):
    joint = ONE_BONE_TARGETS[target_joint]
    pos, world = differentiable_fk(local_rotmat, translation, bone_offsets)
    correction = _rotation_between_vectors(pos[..., target_joint, :] - pos[..., joint, :],
                                           target_pos - pos[..., joint, :])
    parent = SMPL22_PARENTS[joint]
    parent_rot = world[..., parent, :, :] if parent >= 0 else torch.eye(3, device=world.device, dtype=world.dtype)
    result = local_rotmat.clone()
    result[..., joint, :, :] = parent_rot.transpose(-1, -2) @ correction @ world[..., joint, :, :]
    return result


@torch.no_grad()
def solve_joint_positions(q, root, offsets, joints, targets, axes, fixed_rotations,
                          fixed_translation, num_steps=100, tol=1e-5):
    """Batched simultaneous damped least squares, with hard variable locks.

    q: (N,22,3,3), targets/axes: (N,K,3). Right-local tangent Jacobians
    are analytic. Returns best maximum-residual iterate, not last iterate.
    No soft pose penalty trades off positional accuracy. Not an SQP solver
    and no temporal regularizer or collision/joint-limit constraints are used.
    """
    q, root = q.clone(), root.clone()
    n, k = targets.shape[:2]
    active = sorted(set(a for j in joints for a in ancestors(j)))
    dependency = torch.tensor([[a in ancestors(j) for a in active] for j in joints], device=q.device, dtype=q.dtype)
    mutable, free_root = ~fixed_rotations[:, active], ~fixed_translation
    best_q, best_root = q.clone(), root.clone()
    residual = (targets - differentiable_fk(q, root, offsets)[0][:, joints]) * axes
    best_error = residual.norm(dim=-1).amax(-1)
    damping = torch.full((n,), 1e-5, device=q.device, dtype=q.dtype)
    stale = torch.zeros(n, device=q.device, dtype=torch.long)
    for _ in range(num_steps):
        pos, world = differentiable_fk(q, root, offsets)
        residual = (targets - pos[:, joints]) * axes
        running = (residual.norm(dim=-1).amax(-1) > tol) & (stale < 8)
        if not bool(running.any()):
            break
        lever = pos[:, joints, None, :] - pos[:, None, active, :]
        world_axes = world[:, active].transpose(-1, -2)
        jac = torch.cross(world_axes[:, None].expand(n, k, len(active), 3, 3),
                          lever[..., None, :].expand(n, k, len(active), 3, 3), dim=-1)
        jac *= dependency[None, :, :, None, None] * mutable[:, None, :, None, None]
        jac = jac.permute(0, 1, 4, 2, 3).reshape(n, k * 3, len(active) * 3)
        root_jac = torch.eye(3, device=q.device, dtype=q.dtype)[None, None].expand(n, k, 3, 3)
        root_jac = root_jac * free_root[:, None, None, :]
        jac = torch.cat((jac, root_jac.reshape(n, k * 3, 3)), -1) * axes.reshape(n, k * 3, 1)
        jt = jac.transpose(-1, -2)
        if jac.shape[1] <= jac.shape[2]:
            system = jac @ jt
            eye = torch.eye(system.shape[-1], device=q.device, dtype=q.dtype)
            solution, info = torch.linalg.solve_ex(system + damping[:, None, None] * eye,
                                                  residual.reshape(n, -1, 1))
            delta = (jt @ solution).squeeze(-1)
        else:
            system = jt @ jac
            eye = torch.eye(system.shape[-1], device=q.device, dtype=q.dtype)
            solution, info = torch.linalg.solve_ex(system + damping[:, None, None] * eye,
                                                  jt @ residual.reshape(n, -1, 1))
            delta = solution.squeeze(-1)
        valid_step = (info == 0) & torch.isfinite(delta).all(-1)
        delta = torch.where(valid_step[:, None], delta, torch.zeros_like(delta))
        step = delta[:, :-3].reshape(n, len(active), 3)
        step *= .25 / step.norm(dim=-1, keepdim=True).clamp_min(.25)
        root_step = delta[:, -3:] * free_root
        root_step *= .05 / root_step.norm(dim=-1, keepdim=True).clamp_min(.05)
        score = residual.square().sum((1, 2))
        accepted = torch.zeros(n, device=q.device, dtype=torch.bool)
        next_q, next_root = q.clone(), root.clone()
        for scale in (1., .5, .25, .125, .0625):
            trial_q = q.clone()
            trial_q[:, active] = q[:, active] @ _rotation_exp(step * scale)
            trial_root = root + root_step * scale
            trial_residual = (targets - differentiable_fk(trial_q, trial_root, offsets)[0][:, joints]) * axes
            trial_score = trial_residual.square().sum((1, 2))
            take = running & ~accepted & torch.isfinite(trial_score) & (trial_score < score)
            next_q = torch.where(take[:, None, None, None], trial_q, next_q)
            next_root = torch.where(take[:, None], trial_root, next_root)
            accepted |= take
            if bool((accepted | ~running).all()):
                break
        q, root = next_q, next_root
        error = ((targets - differentiable_fk(q, root, offsets)[0][:, joints]) * axes).norm(dim=-1).amax(-1)
        improved = error < best_error
        best_q = torch.where(improved[:, None, None, None], q, best_q)
        best_root = torch.where(improved[:, None], root, best_root)
        best_error = torch.minimum(best_error, error)
        stale = torch.where(accepted, torch.zeros_like(stale), stale + 1)
        damping_floor = 1e-10 if q.dtype == torch.float64 else 1e-6
        damping = torch.where(accepted, damping * .5, damping * 10).clamp(damping_floor, 1.)
    return best_q, best_root


def solve_gradient_ik(local_rotmat, translation, bone_offsets, target_joint, target_pos,
                      lr=.05, num_steps=100, reg_weight=.001, tol=1e-4):
    """Legacy entry point. DLS replaces Adam; lr/reg_weight are compatibility-only."""
    q, _ = solve_joint_positions(local_rotmat[None], translation[None], bone_offsets,
                                 [target_joint], target_pos[None, None],
                                 torch.ones((1, 1, 3), device=local_rotmat.device, dtype=local_rotmat.dtype),
                                 torch.zeros((1, 22), device=local_rotmat.device, dtype=torch.bool),
                                 torch.ones((1, 3), device=local_rotmat.device, dtype=torch.bool), num_steps, tol)
    return q[0]


def solve_single_constraint(local_rotmat, translation, bone_offsets, target_joint, target_pos):
    strategy = get_ik_strategy(target_joint)
    if strategy == 'root':
        return local_rotmat.clone(), solve_root_ik(translation, bone_offsets, target_pos)
    solver = {'two_bone': solve_two_bone_ik, 'one_bone': solve_one_bone_ik, 'gradient': solve_gradient_ik}[strategy]
    return solver(local_rotmat, translation, bone_offsets, target_joint, target_pos), translation.clone()
