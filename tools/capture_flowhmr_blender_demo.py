"""Capture actual Blender UI with plugin-imported native and test Mixamo rigs.

Run in foreground Blender (Xvfb is sufficient on a headless Linux host):
  blender --window-geometry 0 0 1600 1000 --python this.py --
    --source-blend /path/to/flowhmr-smoke.blend --output-dir /path/to/demo
The Mixamo fixture is synthetic, without a character mesh. No poses are invented.
"""
import argparse
import json
from pathlib import Path
import sys
import subprocess

import addon_utils
import bpy
from mathutils import Vector


def configure(args):
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(root/'blender_extension'))
    addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
    from motius_flowhmr.animation import _curves

    native = bpy.data.objects['FlowHMR_SMPLH']
    mixamo = bpy.data.objects['FlowHMR_SMPLH.001']
    for obj in list(bpy.data.objects):
        if obj not in (native, mixamo):
            bpy.data.objects.remove(obj, do_unlink=True)
    for curve in _curves(mixamo.animation_data.action):
        for point in curve.keyframe_points:
            for value in (point.co, point.handle_left, point.handle_right):
                value.x -= 10
    native.name = 'Native SMPL-H · 52 joints'
    mixamo.name = 'Mixamo retarget · synthetic test rig'
    mixamo.scale = (.5, .5, .5)
    scene = bpy.context.scene
    window = bpy.context.window
    scene.frame_start, scene.frame_end = 1, 30
    scene.render.fps, scene.render.fps_base = 30, 1
    scene.frame_set(1)
    bpy.context.view_layer.update()
    for arm, x in ((native, -1.05), (mixamo, 1.05)):
        pelvis = arm.matrix_world @ arm.pose.bones[0].head
        lowest = min((arm.matrix_world @ b.head).z for b in arm.pose.bones)
        arm.location += Vector((x-pelvis.x, -pelvis.y, .025-lowest))
        arm.data.display_type = 'OCTAHEDRAL'
        arm.show_in_front = True
        arm.hide_set(False)
    for arm, color in ((native, (.18, .58, .96)), (mixamo, (.96, .52, .18))):
        for bone in arm.data.bones:
            bone.color.palette = 'CUSTOM'
            bone.color.custom.normal = color
            bone.color.custom.select = color
            bone.color.custom.active = color
            bone.select = False
        arm.data.bones.active = None
    bpy.context.view_layer.update()

    view_location = Vector((0., 0., 1.05))
    view_rotation = (view_location-Vector((.8, -8., 3.3))).to_track_quat('-Z', 'Y')
    for x, title in ((-1.05, 'SMPL-H / 52 joints'), (1.05, 'Mixamo / test rig')):
        text = bpy.data.curves.new(title, 'FONT')
        text.body, text.size, text.align_x = title, .13, 'CENTER'
        obj = bpy.data.objects.new(title, text)
        scene.collection.objects.link(obj)
        obj.location = (x, 0, 2.13)
        obj.rotation_euler = view_rotation.to_euler()
    for obj in bpy.context.selected_objects:
        obj.select_set(False)
    native.select_set(True)
    mixamo.select_set(True)
    bpy.context.view_layer.objects.active = native
    bpy.ops.object.mode_set(mode='POSE')

    props = scene.flowhmr
    props.import_mode, props.target, props.result_armature = 'MIXAMO', mixamo, mixamo
    props.checkpoint = 'ZeyuLing/Motius-FlowHMR-Base'
    props.mode = 'STANDARD'
    props.show_setup, props.show_advanced = False, False
    props.state = 'completed'
    props.message = 'Imported measured Base capture: 30 frames, body and full hands.'
    props.job_directory = ''
    props.video = json.loads((root/'outputs/validation/flowhmr/real/input.json').read_text())['video']

    bpy.context.preferences.view.show_splash = False
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D':
                space = area.spaces.active
                space.show_region_ui = True
                space.show_region_toolbar = False
                space.overlay.show_relationship_lines = False
                space.overlay.show_cursor = False
                space.shading.background_type = 'VIEWPORT'
                space.shading.background_color = (.075, .085, .105)
                space.region_3d.view_rotation = view_rotation
                space.region_3d.view_location = view_location
                space.region_3d.view_distance = 4.7
                space.region_3d.view_perspective = 'ORTHO'
                for region in area.regions:
                    if region.type == 'UI' and hasattr(region, 'active_panel_category'):
                        # Blender exposes this as read-only; activate the tab
                        # through ordinary UI input when requested below.
                        try:
                            region.active_panel_category = 'FlowHMR'
                        except AttributeError:
                            pass
                area.tag_redraw()
    scene.frame_set(15)
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'flowhmr-blender-display.blend'))
    if args.activate_tab:
        def dismiss_setup():
            # First-run Quick Setup is a real modal UI, not part of the scene.
            subprocess.run(['xdotool', 'mousemove', '800', '706', 'click', '1'], check=True)
            return None

        def activate_tab():
            subprocess.run(['xdotool', 'search', '--onlyvisible', '--class', 'Blender', 'windowfocus'], check=True)
            subprocess.run(['xdotool', 'key', 'Escape', 'mousemove', '1308', '253'], check=True)
            return None

        def click_tab():
            subprocess.run(['xdotool', 'click', '1'], check=True)
            return None

        def clear_pointer():
            subprocess.run(['xdotool', 'mousemove', '1590', '985'], check=True)
            return None

        bpy.app.timers.register(dismiss_setup, first_interval=1.5)
        bpy.app.timers.register(activate_tab, first_interval=3.)
        bpy.app.timers.register(click_tab, first_interval=4.)
        bpy.app.timers.register(clear_pointer, first_interval=5.)
    frames = [1, 15, 30] if args.preview else list(range(1, 31))
    pending = []
    counter = 0

    def capture():
        nonlocal counter
        try:
            if pending:
                frame = pending.pop()
                path = output/f'blender-{frame:03d}.png'
                if args.screenshot_python:
                    # Some software GL drivers return a black buffer from the
                    # Blender operator. Read the actual X11 desktop instead.
                    subprocess.run([args.screenshot_python, '-c',
                        'from PIL import ImageGrab; import os,sys; '
                        'ImageGrab.grab(xdisplay=os.environ["DISPLAY"]).save(sys.argv[1])',
                        str(path)], check=True)
                else:
                    with bpy.context.temp_override(window=window):
                        bpy.ops.screen.screenshot(filepath=str(path))
                counter += 1
                print(f'CAPTURED_FRAME {frame}', flush=True)
            if counter == len(frames):
                scene.frame_set(15)
                bpy.ops.wm.save_as_mainfile(filepath=str(output/'flowhmr-blender-display.blend'))
                (output/'display.json').write_text(json.dumps(dict(
                    source_blend=str(args.source_blend.resolve()), frames=frames,
                    blender=bpy.app.version_string, native_joints=52,
                    mixamo_fixture='synthetic rest-basis test armature, no character mesh',
                    inference='previously measured Base capture; poses imported by the plugin'), indent=2)+'\n')
                if args.hold_seconds:
                    bpy.app.timers.register(lambda: bpy.ops.wm.quit_blender() and None,
                                            first_interval=args.hold_seconds)
                else:
                    bpy.ops.wm.quit_blender()
                return None
            frame = frames[counter]
            scene.frame_set(frame)
            for area in window.screen.areas:
                area.tag_redraw()
            pending.append(frame)
            return 1.
        except Exception as error:
            (output/'error.txt').write_text(repr(error))
            import traceback
            traceback.print_exc()
            bpy.ops.wm.quit_blender()
            return None

    bpy.app.timers.register(capture, first_interval=7.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-blend', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--preview', action='store_true', help='Capture three poses instead of the full 30-frame clip')
    parser.add_argument('--screenshot-python', default='', help='External Python with Pillow X11 ImageGrab support')
    parser.add_argument('--hold-seconds', type=float, default=0., help='Keep the window open after capturing')
    parser.add_argument('--activate-tab', action='store_true', help='Use xdotool to dismiss first-run setup and select the FlowHMR tab at 1600x1000')
    args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    bpy.ops.wm.open_mainfile(filepath=str(args.source_blend))

    def launch():
        try:
            window = bpy.context.window_manager.windows[0]
            area = next(a for a in window.screen.areas if a.type == 'VIEW_3D')
            region = next(r for r in area.regions if r.type == 'WINDOW')
            with bpy.context.temp_override(window=window, area=area, region=region):
                configure(args)
        except Exception:
            import traceback
            traceback.print_exc()
            bpy.ops.wm.quit_blender()
        return None

    # Foreground file loading replaces the window's screen. Wait for that
    # replacement to finish before using context-dependent pose operators.
    bpy.app.timers.register(launch, first_interval=1.)


if __name__ == '__main__':
    main()
