"""Validate an installed extension, including its actual namespace and default.

Run after `blender --command extension install-file --enable ...` with the
same isolated BLENDER_USER_RESOURCES directory. Reuses a measured capture;
this checks the Blender/external-Python bridge without rerunning GPU inference.
"""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time

import bpy

parser = argparse.ArgumentParser()
parser.add_argument('--expected-mode', required=True)
parser.add_argument('--external-python', required=True)
parser.add_argument('--project-root', required=True)
parser.add_argument('--body-model', required=True)
parser.add_argument('--capture', required=True)
parser.add_argument('--output-dir', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
package = next(a.module for a in bpy.context.preferences.addons if a.module.endswith('.motius_flowhmr'))
ui = importlib.import_module(package+'.ui')
prefs = ui._preferences(bpy.context)
props = bpy.context.scene.flowhmr
assert props.mode == args.expected_mode, props.mode
prefs.external_python, prefs.project_root, prefs.body_model = args.external_python, args.project_root, args.body_model
props.device, props.output_root = 'cpu', args.output_dir
props.capture_file = args.capture
assert bpy.ops.flowhmr.run(kind='capture') == {'FINISHED'}
deadline = time.monotonic()+90
while ui._ACTIVE is not None and time.monotonic() < deadline:
    ui._tick()
    time.sleep(.1)
assert ui._ACTIVE is None and props.state == 'completed', props.message
assert props.result_armature is None, 'Completion must not import automatically'
from importlib import import_module
queue = import_module(package+'.task_queue').Queue()
record = next(r for r in queue.list(100000) if r['state']=='completed' and json.loads(r['request']).get('capture') == args.capture)
assert bpy.ops.flowhmr.queue_action(action='IMPORT', identity=record['id']) == {'FINISHED'}
assert props.result_armature is not None, props.message
assert 'torch' not in sys.modules, 'Torch must stay in the external interpreter'
assert len(props.result_armature.data.bones) == 52
settings = json.loads(props.result_armature.animation_data.action['flowhmr_inference_settings'])
assert settings['lightweight'] == (args.expected_mode == 'LIGHTWEIGHT')
output = Path(args.output_dir)
bpy.ops.wm.save_as_mainfile(filepath=str(output/'installed-extension.blend'))
report = dict(blender=bpy.app.version_string, package=package, mode=props.mode,
              import_joints=52, external_capture_conversion=True,
              checkpoint_sha256=props.result_armature.animation_data.action['flowhmr_checkpoint_sha256'],
              inference_settings=settings, message=props.message)
(output/'installed-extension-validation.json').write_text(json.dumps(report, indent=2)+'\n')
ui._cleanup()
print('FLOWHMR_INSTALLED_EXTENSION_PASSED', json.dumps(report))
