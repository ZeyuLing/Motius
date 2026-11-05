"""Run inside actual Blender: register, FK fidelity, retarget, FBX, async doctor.

blender -b --factory-startup --python tests/blender_flowhmr_smoke.py --
  --motion /path/to/motion.blender.npz --output-dir /tmp/flowhmr-smoke
  --external-python /path/to/motius/venv/bin/python
"""
import argparse
import json
from pathlib import Path
import sys
import time

import addon_utils
import bpy
import numpy as np
from mathutils import Matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--external-python', type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'blender_extension'))
    addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
    from motius_flowhmr import animation, ui
    from motius_flowhmr.motion_data import MIXAMO_NAMES, read_motion, to_blender
    assert hasattr(bpy.types.Scene, 'flowhmr')
    data = to_blender(read_motion(args.motion))
    native = animation.import_motion(bpy.context, args.motion)
    errors = []
    orientation_errors = []
    for frame in (0, len(data['joints_world'])//2, len(data['joints_world'])-1):
        bpy.context.scene.frame_set(frame+1)
        for index, name in enumerate(animation.JOINT_NAMES):
            pb = native.pose.bones[name]
            errors.append(np.linalg.norm(np.array(pb.matrix.translation)-data['joints_world'][frame, index]))
            expected = data['global_rotations'][frame, index] @ np.array(native.data.bones[name].matrix_local.to_3x3())
            orientation_errors.append(np.max(np.abs(np.array(pb.matrix.to_3x3())-expected)))
    assert max(errors) < 1e-5, max(errors)
    # Blender evaluates long finger chains in float32; matrix composition and
    # decomposition add rounding beyond the serialized source matrices.
    assert max(orientation_errors) < 3e-5, max(orientation_errors)
    assert bpy.context.scene.render.fps == 30
    assert bpy.context.scene.frame_end == len(data['joints_world'])
    # Use a different armature basis, full namespaced Mixamo naming, and twice
    # the source bone lengths. Identity motion must preserve this rest pose;
    # actual motion must preserve the character's original limb lengths.
    bpy.context.scene.frame_set(1)
    rig_data = data.copy()
    rig_data['rest_joints'] = data['rest_joints']*2
    mixamo, mapping = animation._native_rig(bpy.context, rig_data)
    bpy.ops.object.mode_set(mode='EDIT')
    for index, old_name in mapping.items():
        bone = mixamo.data.edit_bones[old_name]
        bone.name = 'mixamorig:'+MIXAMO_NAMES[index]
        bone.roll += .23
    bpy.ops.object.mode_set(mode='OBJECT')
    lengths = {b.name: b.length for b in mixamo.data.bones}
    identity = dict(data, schema=np.array('motius.flowhmr.blender.v1'),
                    global_rotations=np.broadcast_to(np.eye(3), data['global_rotations'].shape).copy())
    # The interchange remains Y-up; identity rotations and constant joint
    # positions make this a rest-basis test independent of source poses.
    original = read_motion(args.motion)
    identity = dict(original, global_rotations=identity['global_rotations'],
                    joints_world=np.repeat(original['joints_world'][:1], len(original['joints_world']), axis=0))
    identity_file = args.output_dir/'identity.npz'
    np.savez_compressed(identity_file, **identity)
    animation.import_motion(bpy.context, identity_file, target=mixamo)
    bpy.context.scene.frame_set(1)
    rest_errors = [np.max(np.abs(np.array(b.matrix)-np.array(mixamo.data.bones[b.name].matrix_local))) for b in mixamo.pose.bones]
    assert max(rest_errors) < 1e-5, max(rest_errors)
    old = mixamo.animation_data.action
    animation.import_motion(bpy.context, args.motion, target=mixamo, start_frame=11)
    assert old in bpy.data.actions.values() and old.use_fake_user
    length_errors = []
    for frame in (11, 11+len(data['joints_world'])-1):
        bpy.context.scene.frame_set(frame)
        for b in mixamo.pose.bones:
            length_errors.append(abs((b.tail-b.head).length-lengths[b.name]))
    assert max(length_errors) < 2e-5, max(length_errors)
    assert abs(mixamo.animation_data.action['flowhmr_root_scale']-2) < 1e-5
    # Real Mixamo FBX uses an object rotation of 90 degrees around X and a
    # centimetre unit scale. Verify orientations/travel in WORLD coordinates,
    # rather than hiding that transform in an identity synthetic armature.
    mixamo.rotation_euler = (np.pi/2, 0., .3)
    mixamo.scale = (.01, .01, .01)
    bpy.context.view_layer.update()
    animation.import_motion(bpy.context, args.motion, target=mixamo, start_frame=11)
    object_rotation = np.array(mixamo.matrix_world.to_quaternion().to_matrix())
    world_basis_errors = []
    roots = []
    for frame in (11, 11+len(data['joints_world'])-1):
        bpy.context.scene.frame_set(frame)
        roots.append(np.array(mixamo.matrix_world @ mixamo.pose.bones['mixamorig:Hips'].head))
        for index, name in enumerate(MIXAMO_NAMES):
            bone_name = 'mixamorig:'+name
            actual = np.array((mixamo.matrix_world @ mixamo.pose.bones[bone_name].matrix).to_quaternion().to_matrix())
            expected = data['global_rotations'][frame-11, index] @ object_rotation @ np.array(mixamo.data.bones[bone_name].matrix_local.to_3x3())
            world_basis_errors.append(np.abs(actual-expected).max())
    assert max(world_basis_errors) < 3e-5, max(world_basis_errors)
    expected_travel = (data['joints_world'][-1, 0]-data['joints_world'][0, 0])*.02
    np.testing.assert_allclose(roots[-1]-roots[0], expected_travel, atol=1e-6)
    # Horizontal in-place trajectory, while retaining vertical root motion.
    inplace = animation.import_motion(bpy.context, args.motion, root_motion=False)
    horizontal = []
    for frame in (1, len(data['joints_world'])):
        bpy.context.scene.frame_set(frame)
        horizontal.append(np.array(inplace.pose.bones['Pelvis'].matrix.translation)[:2])
    np.testing.assert_allclose(*horizontal, atol=1e-6)
    bpy.context.scene.flowhmr.result_armature = mixamo
    fbx = args.output_dir/'mixamo_animation.fbx'
    assert bpy.ops.flowhmr.export_fbx(filepath=str(fbx)) == {'FINISHED'}
    assert fbx.stat().st_size > 1000
    # Round-trip the FBX to exercise Blender's actual importer and the plugin UI.
    bpy.context.scene.flowhmr.target = None
    assert bpy.ops.flowhmr.import_character(filepath=str(fbx)) == {'FINISHED'}
    assert bpy.context.scene.flowhmr.target is not None
    before = set(bpy.data.objects)
    bpy.ops.import_scene.fbx(filepath=str(fbx), use_anim=True)
    animated = next(obj for obj in set(bpy.data.objects)-before if obj.type == 'ARMATURE')
    imported_action = animated.animation_data.action
    assert imported_action is not None
    roundtrip_range = list(imported_action.frame_range)
    assert abs(roundtrip_range[1]-roundtrip_range[0]-(len(data['joints_world'])-1)) < 1e-4
    if args.external_python:
        prefs = bpy.context.preferences.addons['motius_flowhmr'].preferences
        prefs.external_python, prefs.project_root = str(args.external_python), str(root)
        assert ui._path(prefs.external_python) == str(args.external_python.absolute())
        prefs.body_model = str(root/'outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz')
        props = bpy.context.scene.flowhmr
        props.device, props.output_root = 'cpu', str(args.output_dir/'jobs')
        assert bpy.ops.flowhmr.run(kind='doctor') == {'FINISHED'}
        deadline = time.monotonic()+60
        while ui._ACTIVE is not None and time.monotonic() < deadline:
            ui._tick()
            time.sleep(.1)
        assert ui._ACTIVE is None and props.state == 'completed', props.message
        if bpy.app.timers.is_registered(ui._tick):
            bpy.app.timers.unregister(ui._tick)
        assert bpy.ops.flowhmr.run(kind='doctor') == {'FINISHED'}
        child = ui._ACTIVE['job'].process
        assert bpy.ops.flowhmr.cancel() == {'FINISHED'}
        deadline = time.monotonic()+10
        while ui._ACTIVE is not None and time.monotonic() < deadline:
            ui._tick()
            time.sleep(.1)
        assert ui._ACTIVE is None and child.poll() is not None and props.state == 'cancelled'
        if bpy.app.timers.is_registered(ui._tick):
            bpy.app.timers.unregister(ui._tick)
    bpy.ops.wm.save_as_mainfile(filepath=str(args.output_dir/'flowhmr-smoke.blend'))
    report = dict(blender=bpy.app.version_string, frames=len(data['joints_world']), native_joints=52,
                  native_max_position_error_m=float(max(errors)), native_max_rotation_matrix_error=float(max(orientation_errors)),
                  mixamo_rest_basis_max_error=float(max(rest_errors)), mixamo_max_bone_length_error=float(max(length_errors)),
                  mixamo_object_transform_world_rotation_error=float(max(world_basis_errors)),
                  fbx_export_import=True, external_doctor=bool(args.external_python),
                  fbx_frame_range=roundtrip_range,
                  job_cancellation=bool(args.external_python))
    (args.output_dir/'blender-validation.json').write_text(json.dumps(report, indent=2)+'\n')
    if args.external_python:
        assert bpy.ops.flowhmr.run(kind='doctor') == {'FINISHED'}
        child = ui._ACTIVE['job'].process
    addon_utils.disable('motius_flowhmr', default_set=True)
    if args.external_python:
        assert child.poll() is not None
    assert not hasattr(bpy.types.Scene, 'flowhmr')
    addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
    addon_utils.disable('motius_flowhmr', default_set=True)
    print('FLOWHMR_BLENDER_SMOKE_PASSED', json.dumps(report))


if __name__ == '__main__':
    main()
