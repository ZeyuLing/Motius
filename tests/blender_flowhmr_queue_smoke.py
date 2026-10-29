"""Actual Blender: independent queue lifetime and explicit result import."""
import argparse
import importlib
import json
import os
from pathlib import Path
import sys
import time
import bpy
import addon_utils

p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--external-python',required=True);p.add_argument('--body-model',required=True);p.add_argument('--motion',required=True);p.add_argument('--output',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);root=Path(a.project_root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
os.environ['MOTIUS_FLOWHMR_QUEUE']=str(out/'queue.sqlite3')
sys.path.insert(0,str(root/'blender_extension'));addon_utils.enable('motius_flowhmr',default_set=True,persistent=False)
from motius_flowhmr import ui
from motius_flowhmr.task_queue import Queue
props=bpy.context.scene.flowhmr;prefs=ui._preferences(bpy.context)
prefs.external_python=a.external_python;prefs.project_root=str(root);prefs.body_model=a.body_model
props.device='cpu';props.output_root=str(out/'jobs');props.surface=False
assert not props.auto_import
before=len(bpy.context.scene.objects)
assert bpy.ops.flowhmr.run(kind='doctor')=={'FINISHED'}
handle=ui._ACTIVE['job'];identity=handle.identity
assert handle.poll() is None
ui._cleanup()  # Same hook used when loading another .blend / disabling the plugin.
assert handle.queue.get(identity)['cancel']==0
ui._restore_queue()
deadline=time.monotonic()+90
while handle.poll() is None and time.monotonic()<deadline:time.sleep(.1)
assert handle.poll()==0,handle.read_status()
assert len(bpy.context.scene.objects)==before
queue=Queue();directory=out/'ready-result'
record=queue.enqueue(a.external_python,dict(output_dir=str(directory),project_root=str(root),video='source.mp4',mode='LIGHTWEIGHT'))
attempt=directory/'attempt-001';attempt.mkdir();(attempt/'status.json').write_text(json.dumps(dict(state='completed',motion_file=a.motion)))
queue.update(record,state='completed',attempt=1,message='Result ready')
ui._tick();assert len(bpy.context.scene.objects)==before
assert bpy.ops.flowhmr.queue_action(action='IMPORT',identity=record)=={'FINISHED'}
assert props.result_armature is not None and len(props.result_armature.data.bones)==52
assert 'torch' not in sys.modules
bpy.ops.wm.save_as_mainfile(filepath=str(out/'queue-smoke.blend'))
report=dict(blender=bpy.app.version_string,queue_survives_scene_cleanup=True,manual_import_only=True,imported_joints=52,torch_outside_blender=True,doctor_state=queue.get(identity)['state'])
(out/'validation.json').write_text(json.dumps(report,indent=2));addon_utils.disable('motius_flowhmr',default_set=True)
print('FLOWHMR_QUEUE_SMOKE_PASSED',json.dumps(report))
