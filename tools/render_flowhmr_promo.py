"""Blender: render paired, frame-synchronous SMPL-H / Mixamo studio plates.

The mesh cache is reconstructed from the very same capture as motion.blender.npz.
Only horizontal root centering and a constant floor offset affect presentation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import addon_utils
import bpy
import numpy as np
from mathutils import Vector


def material(name, color, metallic=0., roughness=.45):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    node = mat.node_tree.nodes['Principled BSDF']
    node.inputs['Base Color'].default_value = (*color, 1)
    node.inputs['Metallic'].default_value = metallic
    node.inputs['Roughness'].default_value = roughness
    return mat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion', type=Path, required=True)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--character', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--preview', action='store_true')
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int)
    p.add_argument('--samples', type=int, default=8)
    p.add_argument('--checkpoint-repo', default='ZeyuLing/Motius-FlowHMR-Latest')
    p.add_argument('--skip-project', action='store_true', help='Render a second frame range without rewriting the scene/FBX')
    args = p.parse_args(sys.argv[sys.argv.index('--')+1:])
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'blender_extension'))
    addon_utils.enable('motius_flowhmr', default_set=True)
    from motius_flowhmr.animation import import_motion, _curves
    from motius_flowhmr.motion_data import read_motion, YUP_TO_BLENDER, match_mixamo
    data = read_motion(args.motion)
    roots = data['joints_world'][:, 0] @ YUP_TO_BLENDER.T
    with np.load(args.mesh, allow_pickle=False) as f:
        vertices = f['vertices'] @ YUP_TO_BLENDER.T
        faces = f['faces'].copy()
    frames = len(roots)
    assert len(vertices) == frames
    vertices[:, :, :2] -= roots[:, None, :2]
    smpl_floor_offset = float(-vertices[:, :, 2].min()+.015)
    vertices[:, :, 2] += smpl_floor_offset
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.fbx(filepath=str(args.character.resolve()), use_anim=False)
    arm, = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    assert meshes and all(any(m.type == 'ARMATURE' for m in o.modifiers) for o in meshes)
    assert not arm.animation_data or arm.animation_data.action is None
    import_motion(bpy.context, args.motion, target=arm, root_motion=True)
    scene = bpy.context.scene
    arm.name = 'FlowHMR_Mixamo_Character'
    scene.flowhmr.target = arm
    scene.flowhmr.import_mode = 'MIXAMO'
    scene.flowhmr.motion_file = str(args.motion.resolve())
    scene.flowhmr.checkpoint = args.checkpoint_repo
    scene.flowhmr.state = 'completed'
    scene.flowhmr.message = f'{len(data["joints_world"])}-frame video motion retargeted, including hands.'
    scene.flowhmr.show_setup = False
    scene.render.fps = round(float(data['fps']))
    initial_location = arm.location.copy()
    travel_scale = arm.animation_data.action['flowhmr_root_scale'] * arm.matrix_world.to_scale()[0]
    root_deltas = (roots-roots[0]) * travel_scale

    def center_character(frame):
        arm.location = initial_location-Vector((*root_deltas[frame-1, :2], 0))

    bounds = []
    for frame in range(1, frames+1):
        scene.frame_set(frame)
        center_character(frame)
        dg = bpy.context.evaluated_depsgraph_get()
        bounds.extend([list(o.evaluated_get(dg).matrix_world @ Vector(c))
                       for o in meshes for c in o.evaluated_get(dg).bound_box])
    char_bounds = np.asarray(bounds)
    mixamo_floor_offset = float(-char_bounds[:, 2].min()+.015)
    initial_location.z += mixamo_floor_offset
    char_bounds[:, 2] += mixamo_floor_offset
    for mat in bpy.data.materials:
        if mat.use_nodes:
            node = mat.node_tree.nodes.get('Principled BSDF')
            if node:
                node.inputs['Metallic'].default_value = .22
                node.inputs['Roughness'].default_value = .33
    mesh = bpy.data.meshes.new('FlowHMR predicted SMPL-H surface')
    mesh.from_pydata(vertices[0], [], faces.tolist())
    mesh.update()
    body = bpy.data.objects.new('FlowHMR · SMPL-H', mesh)
    scene.collection.objects.link(body)
    body.data.materials.append(material('Porcelain', (.66, .74, .81), .12, .35))
    for poly in mesh.polygons:
        poly.use_smooth = True
    floor = material('Slate cyclorama', (.075, .095, .125), .05, .68)
    bpy.ops.mesh.primitive_plane_add(size=200)
    bpy.context.object.data.materials.append(floor)
    facing = Vector(YUP_TO_BLENDER @ data['global_rotations'][0, 0] @ YUP_TO_BLENDER.T @ np.array([0., -1., 0.]))
    facing.z = 0
    facing.normalize()
    side = Vector((-facing.y, facing.x, 0))
    # Fit projected bounds across the complete clip, never auto-zoom mid-motion.
    fits = []
    for points in (vertices.reshape(-1, 3), char_bounds):
        x = points @ np.asarray(side)
        low, high = points[:, 2].min(), points[:, 2].max()
        fits.append(dict(center_z=float((low+high)/2),
                         center_x=float((x.max()+x.min())/2),
                         scale=float(max((high-low)*1.20, (x.max()-x.min())*700/568*1.12))))
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    camera.data.type = 'ORTHO'
    scene.camera = camera
    center = Vector((0, 0, 1))
    for name, position, power, size, color in (
        ('Softbox / key', facing*4+side*3+Vector((0, 0, 5)), 750, 4, (1., .93, .84)),
        ('Softbox / fill', facing*2-side*4+Vector((0, 0, 3)), 500, 5, (.69, .85, 1.)),
        ('Edge / rim', -facing*3+side*2+Vector((0, 0, 4)), 1100, 3, (.67, .88, 1.))):
        bpy.ops.object.light_add(type='AREA', location=position)
        light = bpy.context.object
        light.name = name
        light.data.energy, light.data.size, light.data.color = power, size, color
        light.rotation_euler = (center-light.location).to_track_quat('-Z', 'Y').to_euler()
    world = bpy.data.worlds.new('Navy studio')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.075, .10, .15, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .45
    scene.world = world
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 4
    scene.cycles.diffuse_bounces = 2
    scene.cycles.glossy_bounces = 2
    scene.cycles.adaptive_threshold = .05
    scene.cycles.adaptive_min_samples = min(4, args.samples)
    scene.render.use_persistent_data = True
    scene.render.threads_mode, scene.render.threads = 'FIXED', 8
    scene.render.resolution_x = 568
    scene.render.resolution_y = 700
    scene.render.resolution_percentage = 60 if args.preview else 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'AgX'

    def view(stage, fit):
        body.hide_render = stage != 'flowhmr'
        body.hide_viewport = stage != 'flowhmr'
        for obj in meshes:
            obj.hide_render = stage != 'mixamo'
        target = side*fit['center_x']+Vector((0, 0, fit['center_z']))
        camera.location = target+facing*6+Vector((0, 0, .3))
        camera.rotation_euler = (target-camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.ortho_scale = fit['scale']

    report = dict(frames=frames, fps=float(data['fps']), camera_fit=fits,
                  source_fbx_animation_imported=False, horizontal_root_centering=True,
                  mapped_joints=len(match_mixamo([b.name for b in arm.data.bones])),
                  skinned_meshes={o.name:len(o.data.vertices) for o in meshes},
                  render_engine='Cycles CPU', samples=args.samples, plate_resolution=[568, 700],
                  smpl_constant_floor_offset_m=smpl_floor_offset,
                  mixamo_constant_floor_offset_m=mixamo_floor_offset,
                  root_travel_scale=travel_scale,
                  temporal_pose_filter=False, foot_ik=False,
                  character_sha256=hashlib.sha256(args.character.read_bytes()).hexdigest(),
                  motion_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
                  mesh_sha256=hashlib.sha256(args.mesh.read_bytes()).hexdigest(),
                  checkpoint_sha256=str(data['checkpoint_sha256']),
                  inference_settings=json.loads(str(data['inference_settings'])))
    (out/'render-provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    if not args.preview and not args.skip_project:
        arm.location = initial_location
        scene.flowhmr.result_armature = arm
        assert bpy.ops.flowhmr.export_fbx(filepath=str(out/'flowhmr-mixamo-animation.fbx')) == {'FINISHED'}
        # Save the same centered presentation as an independently playable scene.
        # The FBX above retains the full predicted horizontal root trajectory.
        for frame in range(1, frames+1):
            scene.frame_set(frame)
            center_character(frame)
            arm.keyframe_insert('location', frame=frame)
        # Absolute, linear shape keys make the predicted surface playable in
        # Blender without Python handlers, Torch, or an external vertex cache.
        body.shape_key_add(name='Frame 0001')
        keys = body.data.shape_keys
        keys.use_relative = False
        keys.key_blocks[0].interpolation = 'KEY_LINEAR'
        for index in range(1, frames):
            key = body.shape_key_add(name=f'Frame {index+1:04d}')
            key.data.foreach_set('co', vertices[index].astype(np.float32).ravel())
            key.interpolation = 'KEY_LINEAR'
        for index, key in enumerate(keys.key_blocks):
            keys.eval_time = key.frame
            keys.keyframe_insert('eval_time', frame=index+1)
        for curve in _curves(keys.animation_data.action):
            for point in curve.keyframe_points:
                point.interpolation = 'LINEAR'
        parent = bpy.data.objects.new('Mixamo display placement', None)
        scene.collection.objects.link(parent)
        arm.parent = parent
        parent.location = side*1.25
        body.location = -side*1.25
        scene.frame_set(min(151, frames))
        view('mixamo', fits[1])
        body.hide_render = body.hide_viewport = False
        camera.location = center+facing*6+Vector((0, 0, .3))
        camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
        camera.data.ortho_scale = max(5., max(f['scale'] for f in fits)*2.1)
        scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
        scene.render.filepath = '//replay/flowhmr-mixamo-'
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from render_flowhmr_mixamo_demo import prepare_workspace
        prepare_workspace(arm, meshes, camera)
        bpy.ops.wm.save_as_mainfile(filepath=str(out/'flowhmr-mixamo-studio.blend'), compress=True)
        parent.location = (0, 0, 0)
        body.location = (0, 0, 0)
        scene.render.resolution_x, scene.render.resolution_y = 568, 700
    selection = sorted({min(f, frames) for f in (1, 151, 376, 601, 750)}) if args.preview else range(args.start, min(args.end or frames, frames)+1)
    for frame in selection:
        scene.frame_set(frame)
        center_character(frame)
        if body.data.shape_keys is None:
            body.data.vertices.foreach_set('co', vertices[frame-1].astype(np.float32).ravel())
            body.data.update()
        for stage, fit in zip(('flowhmr', 'mixamo'), fits):
            view(stage, fit)
            folder = out/('preview' if args.preview else 'plates')/stage
            folder.mkdir(parents=True, exist_ok=True)
            scene.render.filepath = str(folder/f'{frame:04d}.png')
            bpy.ops.render.render(write_still=True)
        print('PAIRED_FRAME', frame, flush=True)


if __name__ == '__main__':
    main()
