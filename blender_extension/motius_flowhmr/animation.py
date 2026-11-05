# SPDX-License-Identifier: GPL-3.0-or-later
"""Import full SMPL-H poses or transfer rotations to a Mixamo rest skeleton."""
from pathlib import Path
import bpy
import numpy as np
from mathutils import Matrix, Vector

from .motion_data import JOINT_NAMES, match_mixamo, read_motion, to_blender


def import_surface(context, motion_path, armature, start_frame=1):
    """Display the actual predicted SMPL-H surface alongside its native rig."""
    path = Path(motion_path).with_name('surface.blender.npz')
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        vertices, faces = data['vertices'], data['faces']
    if vertices.ndim != 3 or vertices.shape[2] != 3 or not np.isfinite(vertices).all():
        raise ValueError('Invalid predicted surface cache')
    mesh = bpy.data.meshes.new('FlowHMR predicted SMPL-H')
    mesh.from_pydata(vertices[0], [], faces.tolist())
    mesh.update()
    obj = bpy.data.objects.new('FlowHMR · Raw reconstruction', mesh)
    obj['flowhmr_native_surface'] = True
    context.scene.collection.objects.link(obj)
    obj.parent = armature
    obj.matrix_parent_inverse = armature.matrix_world.inverted()
    mat = bpy.data.materials.new('FlowHMR · porcelain')
    mat.diffuse_color = (.68, .76, .84, 1)
    obj.data.materials.append(mat)
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    obj.shape_key_add(name='Frame 0001', from_mix=False)
    keys = obj.data.shape_keys
    keys.use_relative = False
    for i in range(1, len(vertices)):
        key = obj.shape_key_add(name=f'Frame {i+1:04d}', from_mix=False)
        key.data.foreach_set('co', vertices[i].ravel())
    for i, key in enumerate(keys.key_blocks):
        key.interpolation = 'KEY_LINEAR'
        keys.eval_time = key.frame
        keys.keyframe_insert('eval_time', frame=start_frame+i)
    for curve in _curves(keys.animation_data.action):
        for point in curve.keyframe_points:
            point.interpolation = 'LINEAR'
    armature.hide_set(True)
    armature.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    context.scene.frame_set(start_frame)
    return obj


def _native_rig(context, data):
    arm = bpy.data.armatures.new('FlowHMR_SMPLH')
    obj = bpy.data.objects.new('FlowHMR_SMPLH', arm)
    context.scene.collection.objects.link(obj)
    for other in context.selected_objects:
        other.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')
    points, parents = data['rest_joints'], data['parents']
    for index, name in enumerate(JOINT_NAMES):
        bone = arm.edit_bones.new(name)
        head = points[index]
        children = np.flatnonzero(parents == index)
        if len(children) == 1:
            direction = points[children[0]] - head
        elif parents[index] >= 0:
            direction = head - points[parents[index]]
        else:
            direction = np.array([0., 0., .1])
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction, norm = np.array([0., 0., .03]), .03
        bone.head = Vector(head)
        bone.tail = Vector(head + direction / norm * max(.015, norm))
        if parents[index] >= 0:
            bone.parent = arm.edit_bones[JOINT_NAMES[parents[index]]]
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.show_in_front = True
    arm.display_type = 'STICK'
    return obj, {i: name for i, name in enumerate(JOINT_NAMES)}


def _curves(action):
    # Layered actions were introduced in Blender 4.4. Do not depend on the
    # removed legacy Action.fcurves API when running newer Blender versions.
    if hasattr(action, 'layers') and len(action.layers):
        for layer in action.layers:
            for strip in layer.strips:
                for bag in getattr(strip, 'channelbags', ()):
                    yield from bag.fcurves
    elif hasattr(action, 'fcurves'):
        yield from action.fcurves


def _write_baked_channels(action, channels, start_frame):
    """Fill animation curves in bulk, avoiding per-frame dependency graph updates."""
    for curve in _curves(action):
        values = channels.get(curve.data_path)
        if values is None:
            continue
        points = curve.keyframe_points
        count = len(values)
        if len(points) != count:
            points.clear()
            points.add(count)
        coordinates = np.empty((count, 2), np.float32)
        coordinates[:, 0] = np.arange(start_frame, start_frame+count)
        coordinates[:, 1] = values[:, curve.array_index]
        points.foreach_set('co', coordinates.ravel())
        for point in points:
            point.interpolation = 'LINEAR'
        curve.update()


def import_motion(context, filepath, *, target=None, start_frame=1,
                  root_motion=True, auto_scale=True):
    if context.mode != 'OBJECT':
        raise ValueError('Switch to Object Mode before importing motion')
    data = to_blender(read_motion(filepath))
    native = target is None
    if target is not None:
        if target.type != 'ARMATURE' or target.library is not None:
            raise ValueError('Choose an editable local Mixamo armature')
        if any(c.influence != 0 and not c.mute for c in target.constraints) or any(
                c.influence != 0 and not c.mute for b in target.pose.bones for c in b.constraints):
            raise ValueError('Bake or disable active rig constraints before retargeting')
        scale = np.abs(np.array(target.matrix_world.to_scale()))
        if np.min(scale) < 1e-8 or np.ptp(scale) > 1e-5:
            raise ValueError('Apply nonuniform armature scale before retargeting')
        if target.matrix_world.determinant() <= 0:
            raise ValueError('Apply mirrored armature transforms before retargeting')
        if target.animation_data and any(t.strips for t in target.animation_data.nla_tracks if not t.mute):
            raise ValueError('Mute existing NLA tracks before importing a new action')
        mapping = match_mixamo([bone.name for bone in target.data.bones])
    else:
        target, mapping = _native_rig(context, data)
    rest = {bone.name: bone.matrix_local.copy() for bone in target.data.bones}
    # Standard Mixamo FBX imports carry a 90-degree object rotation and 0.01
    # unit scale. Source motion is in Blender world axes, whereas pose matrices
    # are in armature space. Conjugate rotations and rotate travel accordingly.
    armature_to_world = target.matrix_world.to_quaternion().to_matrix()
    world_to_armature = armature_to_world.transposed()
    order = sorted(target.data.bones, key=lambda b: len(b.parent_recursive))
    by_name = {name: index for index, name in mapping.items()}
    root_name = mapping[0]
    root_scale = 1.
    if not native and auto_scale:
        source_length = sum(np.linalg.norm(data['rest_joints'][a]-data['rest_joints'][b]) for a, b in ((1, 4), (4, 7)))
        target_length = sum((rest[mapping[a]].translation-rest[mapping[b]].translation).length for a, b in ((1, 4), (4, 7)))
        if min(source_length, target_length) < 1e-6:
            raise ValueError('Cannot determine root-motion scale from zero-length legs')
        root_scale = target_length/source_length
    target.animation_data_create()
    previous_action = target.animation_data.action
    if previous_action:
        previous_action.use_fake_user = True
    action = bpy.data.actions.new(f'FlowHMR_{Path(filepath).parent.name}')
    target.animation_data.action = action
    previous_quaternions = {}
    channels = {}
    first_root = data['joints_world'][0, 0]
    previous_frame = context.scene.frame_current
    try:
        for frame in range(len(data['joints_world'])):
            desired = {}
            for bone in order:
                name = bone.name
                if native:
                    index = by_name[name]
                    position = data['joints_world'][frame, index].copy()
                    if not root_motion:
                        position[:2] -= (data['joints_world'][frame, 0]-first_root)[:2]
                    orientation = Matrix(data['global_rotations'][frame, index].tolist()) @ rest[name].to_3x3()
                    pose = orientation.to_4x4()
                    pose.translation = Vector(position)
                else:
                    pose = (desired[bone.parent.name] @ rest[bone.parent.name].inverted() @ rest[name]
                            if bone.parent else rest[name].copy())
                    if name in by_name:
                        index = by_name[name]
                        orientation = (world_to_armature @ Matrix(data['global_rotations'][frame, index].tolist())
                                       @ armature_to_world @ rest[name].to_3x3())
                        position = pose.translation.copy()
                        pose = orientation.to_4x4()
                        pose.translation = position
                        if name == root_name:
                            delta = (data['joints_world'][frame, 0]-first_root)*root_scale
                            if not root_motion:
                                delta[:2] = 0
                            pose.translation = rest[name].translation+world_to_armature @ Vector(delta)
                desired[name] = pose
                parent = bone.parent
                kwargs = dict(parent_matrix=desired[parent.name], parent_matrix_local=rest[parent.name]) if parent else {}
                basis = bone.convert_local_to_pose(pose, rest[name], invert=True, **kwargs)
                pb = target.pose.bones[name]
                location, quaternion, scale = basis.decompose()
                previous = previous_quaternions.get(name)
                if previous is not None and quaternion.dot(previous) < 0:
                    quaternion.negate()
                previous_quaternions[name] = quaternion.copy()
                pb.rotation_mode = 'QUATERNION'
                for channel, value in (('location', location), ('rotation_quaternion', quaternion), ('scale', scale)):
                    path = pb.path_from_id(channel)
                    if frame == 0:
                        setattr(pb, channel, value)
                        pb.keyframe_insert(data_path=channel, frame=start_frame, group=name)
                        channels[path] = np.empty((len(data['joints_world']), len(value)), np.float32)
                    channels[path][frame] = value
        _write_baked_channels(action, channels, start_frame)
        action['flowhmr_checkpoint_sha256'] = str(data.get('checkpoint_sha256', ''))
        action['flowhmr_inference_settings'] = str(data.get('inference_settings', '{}'))
        action['flowhmr_source'] = str(Path(filepath).resolve())
        action['flowhmr_mapping'] = 'SMPLH52' if native else 'Mixamo rest-basis rotation transfer'
        action['flowhmr_root_scale'] = root_scale
        action['flowhmr_fps'] = float(data['fps'])
        scene = context.scene
        scene.render.fps = round(float(data['fps']))
        scene.render.fps_base = scene.render.fps / float(data['fps'])
        scene.frame_start, scene.frame_end = start_frame, start_frame+len(data['joints_world'])-1
        scene.frame_set(start_frame)
        context.view_layer.objects.active = target
        for other in context.selected_objects:
            other.select_set(False)
        target.select_set(True)
        return target
    except Exception:
        target.animation_data.action = previous_action
        bpy.data.actions.remove(action)
        if native:
            armature = target.data
            bpy.data.objects.remove(target, do_unlink=True)
            bpy.data.armatures.remove(armature)
        context.scene.frame_set(previous_frame)
        raise
