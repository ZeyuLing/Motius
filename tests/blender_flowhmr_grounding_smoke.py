"""Run in a real three-character Blender scene; exercise grounding rollback.

blender --threads 4 -b three-characters.blend --python this_file.py -- --motion motion.blender.npz
"""
import argparse
from pathlib import Path
import sys
import addon_utils
import bpy

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'blender_extension'))
addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
from motius_flowhmr import grounding
from motius_flowhmr.animation import import_motion
from motius_flowhmr.motion_data import match_mixamo
parser = argparse.ArgumentParser()
parser.add_argument('--motion', required=True)
a = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
arms = []
for obj in bpy.context.scene.objects:
    if obj.type != 'ARMATURE':
        continue
    try:
        match_mixamo([b.name for b in obj.data.bones])
    except ValueError:
        continue
    arms.append(obj)
assert len(arms) == 3
assert grounding._independent_cast(bpy.context, arms)
def unknown_handler(*args):
    pass
bpy.app.handlers.frame_change_post.append(unknown_handler)
try:
    assert not grounding._independent_cast(bpy.context, arms)
finally:
    bpy.app.handlers.frame_change_post.remove(unknown_handler)
mesh = next(o for o in bpy.context.scene.objects if o.type == 'MESH'
            and any(m.type == 'ARMATURE' and m.object == arms[0] for m in o.modifiers))
parent = mesh.parent
try:
    mesh.parent = arms[1]
    assert not grounding._independent_cast(bpy.context, arms)
finally:
    mesh.parent = parent
for arm in arms:
    import_motion(bpy.context, a.motion, target=arm)
    action = arm.animation_data.action
    action.use_frame_range = True
    action.frame_start = 1
    action.frame_end = 12
    assert tuple(action.frame_range) == (1., 12.)
originals = [arm.animation_data.action for arm in arms]
properties = [{key:arm[key] for key in ('flowhmr_raw_action', 'flowhmr_grounded_action') if key in arm}
              for arm in arms]
bpy.context.scene.frame_set(7)
actions_before = set(bpy.data.actions)
original_solve = grounding.solve
calls = [0]
def fail_second(*args, **kwargs):
    calls[0] += 1
    if calls[0] == 2:
        raise RuntimeError('Deliberate second-character failure')
    return original_solve(*args, **kwargs)
grounding.solve = fail_second
try:
    try:
        grounding.ground_many(bpy.context, arms)
    except RuntimeError as error:
        assert 'Deliberate second-character failure' in str(error)
    else:
        raise AssertionError('The injected solver failure did not propagate')
finally:
    grounding.solve = original_solve
assert bpy.context.scene.frame_current == 7
assert [arm.animation_data.action for arm in arms] == originals
assert set(bpy.data.actions) == actions_before
assert [{key:arm[key] for key in ('flowhmr_raw_action', 'flowhmr_grounded_action') if key in arm}
        for arm in arms] == properties
print('BLENDER_GROUNDING_ROLLBACK_COMPLETE', flush=True)
