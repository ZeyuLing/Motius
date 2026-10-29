"""Render an actual skinned Mixamo FBX driven by the FlowHMR Blender add-on.

Runs in Blender. Imports no source FBX animation. --preview renders three
low-resolution poses; the final pass renders the whole motion at 30 fps.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import addon_utils
import bpy
import numpy as np
from mathutils import Vector


def prepare_workspace(arm, meshes, camera):
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    meshes[0].select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    arm.show_in_front = False
    center = camera.location-camera.rotation_euler.to_quaternion() @ Vector((0, 0, 6.))
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.show_region_ui = True
                space.shading.type = 'SOLID'
                space.shading.color_type = 'MATERIAL'
                space.overlay.show_extras = False
                space.overlay.show_relationship_lines = False
                space.region_3d.view_rotation = camera.rotation_euler.to_quaternion()
                space.region_3d.view_location = center
                space.region_3d.view_distance = camera.data.ortho_scale*1.3
                space.region_3d.view_perspective = 'ORTHO'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--character', type=Path, required=True)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', action='store_true')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root/'blender_extension'))
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
    from motius_flowhmr.animation import import_motion
    from motius_flowhmr.motion_data import read_motion, match_mixamo, YUP_TO_BLENDER

    bpy.ops.import_scene.fbx(filepath=str(args.character.resolve()), use_anim=False)
    armatures = [o for o in bpy.context.scene.objects if o.type == 'ARMATURE']
    if len(armatures) != 1:
        raise ValueError('Expected one Mixamo character armature')
    arm = armatures[0]
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not meshes or not all(any(m.type == 'ARMATURE' and m.object == arm for m in o.modifiers) for o in meshes):
        raise ValueError('The character must include meshes skinned to the Mixamo armature')
    if arm.animation_data and arm.animation_data.action is not None:
        raise ValueError('Source FBX animation was unexpectedly imported')
    mapping = match_mixamo([b.name for b in arm.data.bones])
    data = read_motion(args.motion)
    started = time.perf_counter()
    import_motion(bpy.context, args.motion, target=arm, root_motion=True)
    print('RETARGET_COMPLETE', time.perf_counter()-started, flush=True)
    arm.name = 'FlowHMR_Mixamo_Character'
    scene = bpy.context.scene
    scene.flowhmr.result_armature = arm
    scene.flowhmr.import_mode = 'MIXAMO'
    scene.flowhmr.target = arm
    scene.flowhmr.checkpoint = 'ZeyuLing/Motius-FlowHMR-Base'
    scene.flowhmr.state = 'completed'
    scene.flowhmr.message = 'Real Mixamo character: FlowHMR video motion imported, including hands.'
    scene.flowhmr.show_setup = False
    scene.flowhmr.motion_file = str(args.motion.resolve())
    if (output/'inference.json').is_file():
        scene.flowhmr.video = json.loads((output/'inference.json').read_text())['source_video']
    frames = len(data['joints_world'])

    def mesh_bounds():
        depsgraph = bpy.context.evaluated_depsgraph_get()
        return np.array([list(obj.evaluated_get(depsgraph).matrix_world @ Vector(corner))
                         for obj in meshes for corner in obj.evaluated_get(depsgraph).bound_box])

    bounds = []
    for frame in range(1, frames+1):
        scene.frame_set(frame)
        bounds.append(mesh_bounds())
    bounds = np.concatenate(bounds)
    bottom, top = bounds.min(0), bounds.max(0)
    # A single constant display placement avoids floor intersections. There is
    # no temporal foot-lock, smoothing, or per-frame mesh manipulation here.
    placement = Vector((-(top[0]+bottom[0])/2, -(top[1]+bottom[1])/2, -bottom[2]+.005))
    arm.location += placement
    bottom += np.array(placement)
    top += np.array(placement)
    center = Vector((0., 0., (top[2]+bottom[2])/2))
    for material in bpy.data.materials:
        if material.use_nodes:
            for node in material.node_tree.nodes:
                if node.type == 'BSDF_PRINCIPLED':
                    node.inputs['Metallic'].default_value = .15
                    node.inputs['Roughness'].default_value = .42

    ground_material = bpy.data.materials.new('Studio floor')
    ground_material.diffuse_color = (.28, .30, .33, 1)
    ground_material.use_nodes = True
    ground_material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (.28, .30, .33, 1)
    ground_material.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = .85
    bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, 0))
    bpy.context.object.data.materials.append(ground_material)
    bpy.context.object.name = 'Studio floor'

    first_root = YUP_TO_BLENDER @ data['global_rotations'][0, 0] @ YUP_TO_BLENDER.T
    facing = Vector(first_root @ np.array([0., -1., 0.]))
    facing.z = 0
    facing.normalize()
    side = Vector((-facing.y, facing.x, 0))
    camera_location = center+facing*6+Vector((0, 0, .45))
    bpy.ops.object.camera_add(location=camera_location)
    camera = bpy.context.object
    camera.rotation_euler = (center-camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.type = 'ORTHO'
    camera.data.ortho_scale = max((top[2]-bottom[2])*1.18, (top[0]-bottom[0])*1.5*1.18)
    scene.camera = camera
    for name, location, energy, size in (
        ('Key', facing*4+side*3+Vector((0, 0, 6)), 600, 4),
        ('Fill', facing*2-side*3+Vector((0, 0, 3)), 350, 4),
        ('Rim', -facing*3+side*2+Vector((0, 0, 5)), 700, 3)):
        bpy.ops.object.light_add(type='AREA', location=location)
        light = bpy.context.object
        light.name = name
        light.data.energy, light.data.shape, light.data.size = energy, 'DISK', size
        light.rotation_euler = (center-light.location).to_track_quat('-Z', 'Y').to_euler()
    world = bpy.data.worlds.new('Studio world')
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (.18, .20, .24, 1)
    world.node_tree.nodes['Background'].inputs['Strength'].default_value = .5
    scene.world = world
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'CPU'
    scene.cycles.samples = 4 if args.preview else 8
    scene.cycles.use_denoising = True
    scene.render.threads_mode, scene.render.threads = 'FIXED', 4 if args.preview else 8
    scene.render.resolution_x = 320 if args.preview else 480
    scene.render.resolution_y = 480 if args.preview else 720
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = 'PNG'
    scene.view_settings.view_transform = 'AgX'

    report = dict(blender=bpy.app.version_string, frames=frames, fps=float(data['fps']),
                  character_source='https://github.com/mrdoob/three.js/blob/dev/examples/models/fbx/Samba%20Dancing.fbx',
                  character_sha256=hashlib.sha256(args.character.read_bytes()).hexdigest(),
                  motion_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
                  checkpoint_sha256=str(data['checkpoint_sha256']),
                  inference_settings=json.loads(str(data['inference_settings'])),
                  source_fbx_animation_imported=False, mapped_joints=len(mapping),
                  skinned_meshes={o.name:len(o.data.vertices) for o in meshes},
                  retarget_seconds=time.perf_counter()-started, uniform_display_translation=list(placement),
                  foot_ik=False, render_engine='Cycles CPU', root_motion=True,
                  fixed_camera_facing=list(facing))
    (output/'character-provenance.json').write_text(json.dumps(report, indent=2)+'\n')
    scene.frame_set(min(frames, 76))
    if not args.preview:
        assert bpy.ops.flowhmr.export_fbx(filepath=str(output/'flowhmr-mixamo-animation.fbx')) == {'FINISHED'}
        prepare_workspace(arm, meshes, camera)
        bpy.ops.wm.save_as_mainfile(filepath=str(output/'flowhmr-mixamo-character.blend'))
    selection = (1, min(frames, 76), frames) if args.preview else range(1, frames+1)
    folder = output/('preview' if args.preview else 'frames')
    folder.mkdir(exist_ok=True)
    for frame in selection:
        scene.frame_set(frame)
        scene.render.filepath = str(folder/f'character-{frame:04d}.png')
        bpy.ops.render.render(write_still=True)
        print('RENDERED_FRAME', frame, flush=True)


if __name__ == '__main__':
    main()
