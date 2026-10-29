"""Enqueue two real environment checks and exit Blender without cancelling."""
import argparse
import json
import os
from pathlib import Path
import sys
import addon_utils
import bpy

p=argparse.ArgumentParser();p.add_argument('--project-root',required=True);p.add_argument('--external-python',required=True);p.add_argument('--body-model',required=True);p.add_argument('--output',required=True)
a=p.parse_args(sys.argv[sys.argv.index('--')+1:]);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
os.environ['MOTIUS_FLOWHMR_QUEUE']=str(out/'queue.sqlite3')
sys.path.insert(0,str(Path(a.project_root)/'blender_extension'));addon_utils.enable('motius_flowhmr',default_set=True,persistent=False)
from motius_flowhmr import ui
prefs=ui._preferences(bpy.context);prefs.external_python=a.external_python;prefs.project_root=a.project_root;prefs.body_model=a.body_model
props=bpy.context.scene.flowhmr;props.device='cpu';props.output_root=str(out/'jobs')
identities=[]
for _ in range(2):
 assert bpy.ops.flowhmr.run(kind='doctor')=={'FINISHED'}
 identities.append(ui._ACTIVE['job'].identity)
(out/'submitted.json').write_text(json.dumps(identities))
print('FLOWHMR_EXIT_WITH_PENDING_QUEUE',json.dumps(identities))
# Background Blender exits here; its Python/process lifetime must not own tasks.
