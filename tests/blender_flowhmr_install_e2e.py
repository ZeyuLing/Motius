"""Run inside Blender on a CUDA host: install ZIP, download assets, run and import a real full video."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile

import bpy
import addon_utils

parser = argparse.ArgumentParser()
parser.add_argument('--zip', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--body-model', required=True)
parser.add_argument('--source-request', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
os.environ['MOTIUS_FLOWHMR_RUNTIME_DIR'] = str(out/'runtime')
os.environ['MOTIUS_FLOWHMR_QUEUE'] = str(out/'queue.sqlite3')
addon = out/'addons/motius_flowhmr'; addon.mkdir(parents=True)
with zipfile.ZipFile(args.zip) as archive: archive.extractall(addon)
sys.path.insert(0, str(addon.parent))
addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
from motius_flowhmr import ui
props = bpy.context.scene.flowhmr
prefs = ui._preferences(bpy.context)
prefs.runtime_home = str(out/'runtime'); prefs.body_model = args.body_model
props.output_root = str(out/'jobs'); props.device = 'cuda'; props.surface = False
assert bpy.app.online_access, 'Blender must be launched with --online-mode'
assert bpy.ops.flowhmr.install_runtime() == {'FINISHED'}
# Reconnect using only persisted process identity, as after re-opening Blender.
deadline = time.monotonic()+30
while time.monotonic() < deadline:
    initial = ui._runtime_status(str(out/'runtime'))
    if initial.get('installer_start') and initial.get('state') == 'running': break
    time.sleep(.1)
else: raise AssertionError('Installer did not publish process identity')
ui._RUNTIME['log'].close()
ui._RUNTIME = None
ui._restore_installed_runtime()
assert isinstance(ui._RUNTIME['process'], ui._ReattachedInstall)
phases = set(); maximum_bytes = 0; weight_total = 0
end = time.monotonic()+1200
while ui._RUNTIME is not None and time.monotonic() < end:
    state = ui._runtime_status(str(out/'runtime'))
    if state.get('phase'): phases.add(state['phase'])
    if state.get('phase') == 'weights':
        maximum_bytes = max(maximum_bytes, state.get('current', 0)); weight_total = max(weight_total, state.get('total', 0))
    ui._runtime_tick()
    time.sleep(.2)
assert ui._RUNTIME is None, 'Runtime installation timed out'
state = ui._runtime_status(str(out/'runtime'))
assert state.get('state') == 'completed', state
assert Path(prefs.external_python).is_file() and Path(prefs.project_root, 'motius').is_dir()
assert Path(props.checkpoint, 'model.safetensors').is_file()
assert weight_total > 1_000_000_000 and maximum_bytes > 0, (weight_total, maximum_bytes, phases)
assert state['info']['body_ready'] and state['info']['gpu']
assert 'torch' not in sys.modules

def wait_job(handle, timeout):
    deadline = time.monotonic()+timeout
    while handle.poll() is None and time.monotonic() < deadline:
        ui._tick(); time.sleep(.3)
    assert handle.poll() == 0, handle.read_status()
    ui._tick()

assert bpy.ops.flowhmr.run(kind='doctor') == {'FINISHED'}
doctor = ui._ACTIVE['job']; wait_job(doctor, 120)
source = json.loads(Path(args.source_request).read_text())
props.video = source['video']; props.subjects_file = source['subjects_file']; props.subject_id = source['subject_id']
props.mode = 'STANDARD'; props.frontend_precision = 'float16'; props.max_frames = 0; props.camera_interval = 1
assert bpy.ops.flowhmr.run(kind='inference') == {'FINISHED'}
job = ui._ACTIVE['job']; identity = job.identity; wait_job(job, 600)
status = job.read_status(); assert status.get('frames') == 750, status
assert status['inference_settings']['frontend_precision'] == 'float16'
assert bpy.ops.flowhmr.queue_action(action='IMPORT', identity=identity) == {'FINISHED'}
assert props.result_armature and len(props.result_armature.data.bones) == 52
assert 'torch' not in sys.modules
report = dict(success=True, blender=bpy.app.version_string, phases=sorted(phases),
              weight_total_bytes=weight_total, weight_observed_bytes=maximum_bytes,
              gpu=state['info']['gpu'], python=state['info']['python'], torch=state['info']['torch'],
              frames=status['frames'], precision=status['inference_settings']['frontend_precision'],
              bones=len(props.result_armature.data.bones), reattached_installer=True,
              job_directory=str(job.directory))
(out/'e2e-result.json').write_text(json.dumps(report, indent=2)+'\n')
print('BLENDER_RUNTIME_INSTALL_E2E_PASSED', json.dumps(report), flush=True)
addon_utils.disable('motius_flowhmr', default_set=True)
