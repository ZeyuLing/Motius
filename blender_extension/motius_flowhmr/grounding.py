# SPDX-License-Identifier: GPL-3.0-or-later
"""Bake contact grounding onto a separate, reversible Mixamo action."""
import json
import numpy as np
import bpy
from mathutils import Matrix, Vector
from .animation import _write_baked_channels
from .motion_data import match_mixamo
from .grounding_math import solve


def _world_points(obj, depsgraph, indices=None):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    points = np.empty(len(mesh.vertices)*3, np.float32)
    mesh.vertices.foreach_get('co', points)
    points = points.reshape(-1, 3)
    if indices is not None:
        points = points[indices]
    matrix = np.asarray(evaluated.matrix_world)
    result = points @ matrix[:3, :3].T+matrix[:3, 3]
    evaluated.to_mesh_clear()
    return result


def _ground_steps(context, arm, floor=0., progress=None):
    if context.mode != 'OBJECT' or arm is None or arm.type != 'ARMATURE':
        raise ValueError('Choose the animated Mixamo armature in Object Mode')
    original = arm.animation_data.action if arm.animation_data else None
    if original is None:
        raise ValueError('Retarget motion before grounding')
    if original.get('flowhmr_grounded'):
        raise ValueError('This action is already grounded; restore Raw before applying again')
    mapping = match_mixamo([b.name for b in arm.data.bones])
    bones = sorted(arm.data.bones, key=lambda b: len(b.parent_recursive))
    lookup = {b.name:i for i, b in enumerate(bones)}
    parents = np.array([-1 if b.parent is None else lookup[b.parent.name] for b in bones])
    if sum(parents == -1) != 1:
        raise ValueError('Grounding requires one connected armature root')
    scene = context.scene
    start, end = map(round, original.frame_range)
    frames = list(range(start, end+1))
    yield (0, start)
    rest_world = np.asarray(arm.matrix_world)
    rest = np.array([rest_world[:3, :3] @ np.asarray(b.head_local)+rest_world[:3, 3] for b in bones])
    rest_rotations = np.array([np.asarray((arm.matrix_world @ b.matrix_local).to_quaternion().to_matrix()) for b in bones])
    meshes = [o for o in scene.objects if o.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers)]
    if not meshes:
        raise ValueError('Grounding needs a skinned character to measure actual soles')
    foot_vertices = []
    for mesh in meshes:
        groups = []
        for foot, toe in ((7, 10), (8, 11)):
            ids = {g.index for g in mesh.vertex_groups if g.name in (mapping[foot], mapping.get(toe))}
            groups.append(np.array([v.index for v in mesh.data.vertices if sum(g.weight for g in v.groups if g.group in ids) > .5], dtype=int))
        foot_vertices.append(groups)
    joints, rotations, objects, soles, heights, minimum = [], [], [], [], [], []
    for frame in frames:
        if progress and (frame-start) % 60 == 0:
            progress(40*(frame-start)/len(frames), f'Measuring character soles · {frame-start}/{len(frames)}')
        yield (1, frame)
        dg = context.evaluated_depsgraph_get()
        objects.append(arm.matrix_world.copy())
        joints.append([list(arm.matrix_world @ arm.pose.bones[b.name].matrix.translation) for b in bones])
        posed = np.array([np.asarray((arm.matrix_world @ arm.pose.bones[b.name].matrix).to_quaternion().to_matrix()) for b in bones])
        rotations.append(posed @ rest_rotations.swapaxes(-1, -2))
        foot_points = [[], []]
        all_points = []
        for mesh, indices in zip(meshes, foot_vertices):
            points = _world_points(mesh, dg)
            all_points.append(points)
            for f in range(2):
                if len(indices[f]):
                    foot_points[f].append(points[indices[f]])
        points = np.concatenate(all_points)
        minimum.append(points[:, 2].min())
        frame_soles, frame_heights = [], []
        for f in range(2):
            if not foot_points[f]:
                raise ValueError('Foot vertex groups are missing on this character')
            points = np.concatenate(foot_points[f])
            height = points[:, 2].min()
            point = np.mean(points[points[:, 2] <= height+.006], axis=0)
            frame_soles.append(point)
            frame_heights.append(height)
        soles.append(frame_soles)
        heights.append(frame_heights)
    yield (2, start)  # All raw actors are sampled before any action is replaced.
    joints, rotations, soles, heights = map(np.asarray, (joints, rotations, soles, heights))
    chains = [(lookup[mapping[i]] for i in indices) for indices in ((1, 4, 7), (2, 5, 8))]
    chains = [tuple(c) for c in chains]
    corrected, new_rotations, result = solve(joints, rotations, rest, parents, soles, heights,
                                            np.asarray(minimum), chains,
                                            scene.render.fps/scene.render.fps_base, floor)
    original.use_fake_user = True
    action = original.copy()
    action.name = original.name+' · Grounded'
    arm.animation_data.action = action
    quaternions = {}
    channels = {}
    try:
        for t, frame in enumerate(frames):
            if progress and t % 60 == 0:
                progress(45+40*t/len(frames), f'Baking grounded action · {t}/{len(frames)}')
            inverse = objects[t].inverted()
            object_rotation = objects[t].to_quaternion().to_matrix()
            desired = {}
            for i, bone in enumerate(bones):
                rotation = Matrix((new_rotations[t, i] @ rest_rotations[i]).tolist())
                pose = (object_rotation.transposed() @ rotation).to_4x4()
                pose.translation = inverse @ Vector(corrected[t, i])
                desired[bone.name] = pose
                kwargs = dict(parent_matrix=desired[bone.parent.name], parent_matrix_local=bone.parent.matrix_local) if bone.parent else {}
                basis = bone.convert_local_to_pose(pose, bone.matrix_local, invert=True, **kwargs)
                pb = arm.pose.bones[bone.name]
                location, quaternion, scale = basis.decompose()
                if bone.name in quaternions and quaternion.dot(quaternions[bone.name]) < 0:
                    quaternion.negate()
                quaternions[bone.name] = quaternion.copy()
                pb.rotation_mode = 'QUATERNION'
                for channel, value in (('location', location), ('rotation_quaternion', quaternion), ('scale', scale)):
                    path = pb.path_from_id(channel)
                    if t == 0:
                        setattr(pb, channel, value)
                        pb.keyframe_insert(data_path=channel, frame=start, group=bone.name)
                        channels[path] = np.empty((len(frames), len(value)), np.float32)
                    channels[path][t] = value
        _write_baked_channels(action, channels, start)
        action['flowhmr_grounded'] = True
        action['flowhmr_raw_action'] = original.name
        arm['flowhmr_raw_action'] = original.name
        arm['flowhmr_grounded_action'] = action.name
        result['frames'] = len(frames)
        result['floor_height'] = floor
        result['raw_action'] = original.name
        result['grounded_action'] = action.name
        # Measure the deformed surfaces, not just the ankle joints.
        after, after_soles, after_min = [], [], []
        for frame in frames:
            if progress and (frame-start) % 60 == 0:
                progress(85+15*(frame-start)/len(frames), 'Checking planted soles and jumps')
            yield (3, frame)
            dg = context.evaluated_depsgraph_get()
            all_points = [_world_points(m, dg) for m in meshes]
            after_min.append(min(p[:, 2].min() for p in all_points))
            frame_heights, frame_soles = [], []
            for f in range(2):
                points = np.concatenate([p[ids[f]] for p, ids in zip(all_points, foot_vertices) if len(ids[f])])
                height = points[:, 2].min()
                frame_heights.append(height)
                frame_soles.append(np.mean(points[points[:, 2] <= height+.006], axis=0))
            after.append(frame_heights)
            after_soles.append(frame_soles)
        after = np.asarray(after)
        after_soles = np.asarray(after_soles)
        planted = result['weights'] >= .99
        result['support_sole_before_abs_mean_m'] = float(np.abs(heights[planted]-floor).mean())
        result['support_sole_after_abs_mean_m'] = float(np.abs(after[planted]-floor).mean())
        result['support_sole_after_abs_max_m'] = float(np.abs(after[planted]-floor).max())
        result['fully_planted_foot_frames'] = int(planted.sum())
        result['support_xy_anchor_after_mean_m'] = float(np.linalg.norm((after_soles-result['sole_anchors'])[planted, :2], axis=-1).mean())
        result['support_xy_anchor_before_mean_m'] = float(np.linalg.norm((soles-result['sole_anchors'])[planted, :2], axis=-1).mean())
        result['whole_mesh_min_z_m'] = float(min(after_min))
        flight = (heights-result['estimated_original_floor'] > .15).all(1) & (np.asarray(minimum)-result['estimated_original_floor'] > .08)
        result['verified_flight_frames'] = int(flight.sum())
        if flight.any():
            result['flight_peak_before_m'] = float((heights[flight].min(1)-result['estimated_original_floor']).max())
            result['flight_peak_after_m'] = float((after[flight].min(1)-floor).max())
        serial = {k:v.tolist() if isinstance(v, np.ndarray) else v for k,v in result.items()}
        action['flowhmr_grounding_report'] = json.dumps(serial)
        return serial
    except Exception:
        arm.animation_data.action = original
        bpy.data.actions.remove(action)
        raise


def ground(context, arm, floor=0., progress=None):
    """Ground one actor with the same real-mesh sampling and verification."""
    previous_frame = context.scene.frame_current
    steps = _ground_steps(context, arm, floor, progress)
    try:
        while True:
            try:
                phase, frame = next(steps)
            except StopIteration as done:
                return done.value
            if phase != 2:
                context.scene.frame_set(frame)
    finally:
        steps.close()
        context.scene.frame_set(previous_frame)


def _independent_cast(context, arms):
    # Dependencies on another animated object could change the solve when all
    # actions are baked together. Preserve the serial path for such scenes.
    from . import ui, viewport_preview
    if ui._PLAYBACK is not None:
        return False
    safe_handlers = {ui._follow_view, viewport_preview.invalidate}
    for name in ('frame_change_pre', 'frame_change_post',
                 'depsgraph_update_pre', 'depsgraph_update_post'):
        if any(handler not in safe_handlers for handler in getattr(bpy.app.handlers, name)):
            return False
    if context.scene.rigidbody_world:
        return False
    for obj in context.scene.objects:
        if obj.constraints or (obj.animation_data and obj.animation_data.drivers):
            return False
        if obj.type == 'ARMATURE' and any(b.constraints for b in obj.pose.bones):
            return False
        data_animation = getattr(getattr(obj, 'data', None), 'animation_data', None)
        if data_animation and data_animation.drivers:
            return False
        if obj.type == 'MESH':
            keys = obj.data.shape_keys
            if keys and keys.animation_data and keys.animation_data.drivers:
                return False
            if any(m.type not in {'ARMATURE', 'SUBSURF', 'MIRROR', 'SOLIDIFY',
                                   'TRIANGULATE', 'WEIGHTED_NORMAL'} for m in obj.modifiers):
                return False
            targets = {m.object for m in obj.modifiers if m.type == 'ARMATURE'}
            if len(targets) > 1:
                return False
            if any(m.type == 'MIRROR' and m.mirror_object is not None for m in obj.modifiers):
                return False
            parent = obj.parent
            while parent is not None:
                if parent in arms and targets and parent not in targets:
                    return False
                parent = parent.parent
        if obj in arms and obj.parent is not None:
            return False
    return True


def ground_many(context, arms, floor=0., progress=None):
    """Share scene updates across independent actors without changing the solve.

    Each generator samples its original deformed meshes at every frame, solves
    its own contacts, and verifies all resulting meshes. Only scene.frame_set
    is shared; no geometry or verification is approximated or omitted.
    """
    arms = list(arms)
    if len({a.as_pointer() for a in arms}) != len(arms):
        raise ValueError('Choose each character only once')
    if not arms:
        return []
    if not _independent_cast(context, arms):
        reports = []
        for index, arm in enumerate(arms):
            def update(value, text, i=index):
                if progress:
                    progress((i+value/100)/len(arms)*100, text)
            reports.append(ground(context, arm, floor, update))
        return reports
    if context.mode != 'OBJECT' or any(a.type != 'ARMATURE' or not a.animation_data
            or not a.animation_data.action for a in arms):
        raise ValueError('Choose animated Mixamo armatures in Object Mode')
    previous_frame = context.scene.frame_current
    originals = [(a.animation_data.action, getattr(a.animation_data, "action_slot", None),
                  {k:a[k] for k in ('flowhmr_raw_action', 'flowhmr_grounded_action') if k in a})
                 for a in arms]
    steps, pending, reports = [], {}, [None]*len(arms)
    amounts = [0.]*len(arms)
    def advance(index):
        try:
            pending[index] = next(steps[index])
        except StopIteration as done:
            reports[index] = done.value
            pending.pop(index, None)
    try:
        for index, arm in enumerate(arms):
            def update(value, text, i=index, name=arm.name):
                amounts[i] = value
                if progress:
                    progress(sum(amounts)/len(arms), f'{name} · {text}')
            steps.append(_ground_steps(context, arm, floor, update))
            advance(index)
        while pending:
            phase, frame = min(pending.values())
            if phase != 2:
                context.scene.frame_set(frame)
            for index in list(pending):
                if pending[index] == (phase, frame):
                    advance(index)
        return reports
    except BaseException:
        # Restore every actor if any solver or verification fails, including
        # actors that already finished, and keep the original action slots.
        for arm, (action, slot, properties) in zip(arms, originals):
            current = arm.animation_data.action
            arm.animation_data.action = action
            if slot:
                arm.animation_data.action_slot = slot
            for key in ('flowhmr_raw_action', 'flowhmr_grounded_action'):
                if key in properties:
                    arm[key] = properties[key]
                elif key in arm:
                    del arm[key]
            if current and current != action and current.users <= int(current.use_fake_user):
                bpy.data.actions.remove(current)
        raise
    finally:
        for generator in steps:
            generator.close()
        context.scene.frame_set(previous_frame)
