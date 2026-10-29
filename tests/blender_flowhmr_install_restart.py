"""Two real Blender processes: start installation, exit, and reconnect to it."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import zipfile
import bpy
import addon_utils

p = argparse.ArgumentParser()
p.add_argument('--zip', required=True); p.add_argument('--output', required=True)
p.add_argument('--body-model', required=True); p.add_argument('--phase', choices=['start', 'resume'], required=True)
a = p.parse_args(sys.argv[sys.argv.index('--')+1:]); out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
os.environ['MOTIUS_FLOWHMR_RUNTIME_DIR'] = str(out/'runtime')
os.environ['MOTIUS_FLOWHMR_QUEUE'] = str(out/'queue.sqlite3')
addon = out/'addons/motius_flowhmr'
if not addon.exists():
    addon.mkdir(parents=True)
    with zipfile.ZipFile(a.zip) as z: z.extractall(addon)
sys.path.insert(0, str(addon.parent))
addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
from motius_flowhmr import ui
prefs = ui._preferences(bpy.context); prefs.runtime_home = str(out/'runtime'); prefs.body_model = a.body_model
if a.phase == 'start':
    assert bpy.app.online_access
    assert bpy.ops.flowhmr.install_runtime() == {'FINISHED'}
    deadline = time.monotonic()+300
    while time.monotonic() < deadline:
        state = ui._runtime_status(str(out/'runtime'))
        if state.get('state') == 'failed': raise AssertionError(state)
        if state.get('phase') == 'weights' and state.get('current', 0) > 0:
            (out/'start.json').write_text(json.dumps(state, indent=2))
            print('INSTALL_STARTED_BEFORE_BLENDER_EXIT', json.dumps(state), flush=True)
            break
        time.sleep(.2)
    else: raise AssertionError('Installer never reached weight download')
    # Blender exits without cancelling the separately-owned installation group.
else:
    ui._restore_installed_runtime()
    assert ui._RUNTIME is not None and isinstance(ui._RUNTIME['process'], ui._ReattachedInstall)
    deadline = time.monotonic()+900
    while ui._RUNTIME is not None and time.monotonic() < deadline:
        ui._runtime_tick(); time.sleep(.2)
    assert ui._RUNTIME is None
    state = ui._runtime_status(str(out/'runtime'))
    assert state.get('state') == 'completed', state
    assert Path(prefs.external_python).is_file() and Path(prefs.project_root, 'motius').is_dir()
    assert Path(bpy.context.scene.flowhmr.checkpoint, 'model.safetensors').is_file()
    (out/'restart-result.json').write_text(json.dumps(dict(success=True, body_ready=state['info']['body_ready'],
        python=state['info']['python'], gpu=state['info']['gpu'], reattached=True), indent=2))
    print('INSTALL_RECONNECTED_AFTER_BLENDER_EXIT', flush=True)
