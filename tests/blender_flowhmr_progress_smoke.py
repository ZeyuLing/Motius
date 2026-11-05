"""Run inside Blender to verify atomic worker progress reaches real RNA properties."""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import time
import addon_utils
import bpy

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root/'blender_extension'))
addon_utils.enable('motius_flowhmr', default_set=True, persistent=False)
from motius_flowhmr import ui, worker
from motius_flowhmr.jobs import Job
from motius_flowhmr.progress import InferenceProgress
try:
    with tempfile.TemporaryDirectory() as directory:
        job = Job.__new__(Job)
        job.directory = Path(directory)
        job.process = SimpleNamespace(poll=lambda: None)
        job.cancel_started = None
        ui._ACTIVE = {'job':job, 'scene':bpy.context.scene, 'started_at':time.monotonic()}
        now = [0.]
        progress = InferenceProgress(lambda state,message,**extra: worker.write_json(
            job.directory/'status.json', dict(state=state, message=message, **extra)), clock=lambda: now[0])
        progress('Extract body and hand features', 0, 750)
        now[0] = 10.
        progress('Extract body and hand features', 80, 750)
        assert ui._tick() == .5
        props = bpy.context.scene.flowhmr
        assert props.progress_current == 80 and props.progress_total == 750
        assert '8.0 frames/s' in props.progress_detail
        progress('Estimate camera · global layers', 0, 24)
        ui._tick()
        assert props.progress_current == 0 and props.progress_total == 24
        assert props.progress_detail == ''
        assert props.sam_frame_batch_size == 8
        assert props.frontend_precision == 'source'
        props.frontend_precision = 'float16'
        assert props.frontend_precision == 'float16'
        print('BLENDER_PROGRESS_SMOKE_COMPLETE', json.dumps({'phase':props.phase,'current':props.progress_current,'total':props.progress_total}))
finally:
    ui._ACTIVE = None
    addon_utils.disable('motius_flowhmr', default_set=True)
