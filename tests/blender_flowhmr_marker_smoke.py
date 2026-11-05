"""Run with Blender --background --python ... -- --subjects manifest --video input."""
import argparse
import sys
from pathlib import Path
import bpy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'blender_extension'))
import motius_flowhmr
from motius_flowhmr.subjects import load_subjects
from motius_flowhmr.subject_overlay import install_markers

parser=argparse.ArgumentParser()
parser.add_argument('--subjects',required=True)
parser.add_argument('--video',required=True)
a=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
motius_flowhmr.register()
bpy.context.scene.flowhmr.video=a.video
data=load_subjects(a.subjects)
selected=data['tracks'][0]['id']
install_markers(bpy.context,data,selected)
clip=next(c for c in bpy.data.movieclips if c.tracking.tracks.get(f'FlowHMR person {selected}'))
count=len(clip.tracking.tracks)
install_markers(bpy.context,data,selected)
assert len(clip.tracking.tracks)==count
track=clip.tracking.tracks[f'FlowHMR person {selected}']
assert len(track.markers)==data['frames']
observed=set(data['tracks'][0]['frames'])
assert all(m.mute == (m.frame-1 not in observed) for m in track.markers)
from motius_flowhmr import viewport_preview
area=next(a for a in bpy.context.screen.areas if a.type=='VIEW_3D')
with bpy.context.temp_override(area=area):
    original=viewport_preview.key(bpy.context)
    obj=next(o for o in bpy.context.scene.objects if o.type=='MESH')
    obj.hide_viewport=True
    assert viewport_preview.key(bpy.context)!=original
    obj.hide_viewport=False
    bpy.context.scene.flowhmr.show_surface=not bpy.context.scene.flowhmr.show_surface
    assert viewport_preview.key(bpy.context)!=original
motius_flowhmr.unregister()
print('REPEATED_SUBJECT_REVIEW_PASS',flush=True)
