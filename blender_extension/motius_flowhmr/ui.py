# SPDX-License-Identifier: GPL-3.0-or-later
"""Responsive Blender panel and job lifecycle; no ML dependencies in Blender."""
import os
from pathlib import Path
import time
import math
import json
import signal
import subprocess
import sys

import bpy
from bpy.app.handlers import persistent
from bpy.props import (BoolProperty, EnumProperty, IntProperty, PointerProperty, StringProperty)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup
from bpy_extras.io_utils import ExportHelper, ImportHelper

from . import animation, jobs, viewport_preview, queue_runner
from .viewport_preview import FLOWHMR_OT_prepare_playback
from .defaults import DEFAULT_MODE
from .subject_picker import FLOWHMR_OT_choose_subject

PACKAGE = __package__
# Blender 4.5 requires this flag for file-browser validation of // paths.
_RELATIVE_PATH_OPTIONS = {'PATH_SUPPORTS_BLEND_RELATIVE'} if bpy.app.version >= (4, 5, 0) else set()
_ACTIVE = None
_PLAYBACK = None
_QUEUE_STAMP = None
_RUNTIME = None


def _surface_visibility(self, context):
    surfaces = [o for o in context.scene.objects if o.get('flowhmr_native_surface')]
    for obj in context.scene.objects:
        if obj.get('flowhmr_native_surface'):
            obj.hide_set(not self.show_surface)
            obj.hide_viewport = not self.show_surface
    # Keep the raw result readable instead of drawing it through a cast member.
    # Preserve each character's prior visibility when returning from this view.
    arms = {o for o in context.scene.objects if o.get('flowhmr_character')}
    for obj in context.scene.objects:
        if obj not in arms and not (obj.type == 'MESH' and any(m.type == 'ARMATURE' and m.object in arms for m in obj.modifiers)):
            continue
        if surfaces and self.show_surface:
            if 'flowhmr_before_raw_hidden' not in obj:
                obj['flowhmr_before_raw_hidden'] = obj.hide_get()
            obj.hide_set(True)
        elif 'flowhmr_before_raw_hidden' in obj:
            obj.hide_set(bool(obj['flowhmr_before_raw_hidden']))
            del obj['flowhmr_before_raw_hidden']


@persistent
def _follow_view(scene, _depsgraph=None):
    global _PLAYBACK
    if _PLAYBACK is not None and _PLAYBACK['scene'] == scene:
        seen = _PLAYBACK['frames']
        if not seen or seen[-1] != scene.frame_current:
            seen.append(scene.frame_current)
    if not hasattr(scene, 'flowhmr'):
        return
    props = scene.flowhmr
    arm = props.target if props.import_mode == 'MIXAMO' else props.result_armature
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'CLIP_EDITOR':
                area.spaces.active.clip_user.frame_current = scene.frame_current
                area.tag_redraw()
            if area.type == 'VIEW_3D' and props.follow_view and arm is not None and arm.pose.bones:
                root = arm.matrix_world @ arm.pose.bones[0].matrix.translation
                root.z = 1.
                area.spaces.active.region_3d.view_location = root


def _playback_drawn(kind):
    if _PLAYBACK is None:
        return
    area=bpy.context.area
    if kind=='stage' and area==_PLAYBACK['area']:
        frame=_PLAYBACK['frame']
        viewport_preview.draw(frame)
    elif kind=='clip' and area in _PLAYBACK['clip_areas']:
        frame=_PLAYBACK['frame']
        viewport_preview.draw_source(frame)
    else:return
    _PLAYBACK[kind+'_drawn']=frame
    seen=_PLAYBACK[kind+'_frames']
    if not seen or seen[-1]!=frame:
        seen.append(frame)
        if kind=='stage':_PLAYBACK['presentation_times'].append(time.monotonic()-_PLAYBACK['started_at'])


def _remove_playback_drawers():
    if _PLAYBACK is not None:
        for owner,handle in _PLAYBACK.get('drawers',[]):
            owner.draw_handler_remove(handle,'WINDOW')


def _finish_playback():
    global _PLAYBACK
    if _PLAYBACK is None:
        return None
    stopped_at=time.monotonic()
    _remove_playback_drawers()
    _PLAYBACK['area'].spaces.active.shading.type=_PLAYBACK['shading']
    _PLAYBACK['area'].spaces.active.show_region_ui=_PLAYBACK['sidebar']
    for space, clip in _PLAYBACK.get('source_clips',[]): space.clip=clip
    for obj, hidden in _PLAYBACK.get('visibility', []):
        obj.hide_viewport = hidden
    scene = _PLAYBACK['scene']
    scene.frame_set(_PLAYBACK['frame'])
    bpy.context.view_layer.update()
    if viewport_preview._CACHE is not None:viewport_preview._CACHE['last_frame']=scene.frame_current
    import json
    scene['flowhmr_playback_frames'] = json.dumps(_PLAYBACK['frames'])
    report = dict(frames=_PLAYBACK['frames'], presented_frames=_PLAYBACK['stage_frames'], source_presented_frames=_PLAYBACK['clip_frames'], seconds=stopped_at-_PLAYBACK['started_at'], presentation_times=_PLAYBACK['presentation_times'], mode='cached_viewport', target_fps=1/_PLAYBACK['period'], sync_mode=scene.sync_mode,
                  clips=[a.spaces.active.clip_user.frame_current for screen in bpy.data.screens for a in screen.areas if a.type=='CLIP_EDITOR'])
    expected = list(range(scene.frame_start, scene.frame_end+1))
    report['completed'] = report['frames'] == expected and report['presented_frames'] == expected and (not _PLAYBACK['clip_areas'] or report['source_presented_frames'] == expected)
    expected_seconds=len(expected)*_PLAYBACK['period']
    report['expected_seconds']=expected_seconds
    report['original_rate']=abs(report['seconds']-expected_seconds)<.05
    report['completed']=report['completed'] and report['original_rate']
    audit_error = ''
    if scene.flowhmr.job_directory:
        directory=Path(_path(scene.flowhmr.job_directory))
        try:
            if directory.is_dir():
                temporary=directory/'playback.tmp'
                temporary.write_text(json.dumps(report,indent=2))
                temporary.replace(directory/'playback.json')
        except OSError:
            audit_error = ' · could not save playback audit'
    scene.flowhmr.message = f"Playback complete · {len(_PLAYBACK['frames'])} frames shown"
    scene.flowhmr.message += audit_error
    _PLAYBACK = None
    return None


@persistent
def _restore_scene_status(_unused=None):
    import json
    for scene in bpy.data.scenes:
        props = scene.flowhmr
        arm = props.target if props.import_mode == 'MIXAMO' else props.result_armature
        action = arm.animation_data.action if arm and arm.animation_data else None
        if action is None or not action.get('flowhmr_source'):
            continue
        props.state = 'completed'
        props.message = 'Loaded FlowHMR animation'
        if action.get('flowhmr_grounded'):
            props.message = 'Loaded grounded action · Raw remains available'
            try:
                report = json.loads(action['flowhmr_grounding_report'])
                props.ground_message = f"Support sole error: {report['support_sole_after_abs_mean_m']*1000:.1f} mm · {report['verified_flight_frames']} flight frames kept"
            except (KeyError, ValueError, TypeError):
                props.ground_message = ''


def _preferences(context):
    entry = context.preferences.addons.get(PACKAGE)
    if entry is None:
        raise RuntimeError('Enable Motius FlowHMR in Preferences first')
    return entry.preferences


class FlowHMRPreferences(AddonPreferences):
    bl_idname = PACKAGE
    external_python: StringProperty(name='External Python', subtype='FILE_PATH', default=os.environ.get('MOTIUS_PYTHON', ''))
    project_root: StringProperty(name='Motius checkout', subtype='DIR_PATH', default=os.environ.get('MOTIUS_ROOT', ''))
    body_model: StringProperty(name='Neutral SMPL-H (.npz)', subtype='FILE_PATH', default=os.environ.get('MOTIUS_SMPLH_MODEL', ''))
    cache_dir: StringProperty(name='Weight cache (optional)', subtype='DIR_PATH')
    runtime_home: StringProperty(name='Runtime folder', subtype='DIR_PATH', default=os.environ.get('MOTIUS_FLOWHMR_RUNTIME_DIR', str(Path.home()/'.cache/motius/flowhmr-runtime')))
    ffmpeg_path: StringProperty(options={'HIDDEN'})

    def draw(self, context):
        self.layout.label(text='Inference uses this external environment, never Blender Python.')
        self.layout.operator('flowhmr.install_runtime', text='Install / update runtime', icon='IMPORT')
        for name in ('runtime_home', 'external_python', 'project_root', 'body_model', 'cache_dir'):
            self.layout.prop(self, name)
        self.layout.label(text="Backend setup: pip install -e '.[flowhmr,video-to-motion]'")


class FlowHMRSettings(PropertyGroup):
    subjects_file: StringProperty(subtype='FILE_PATH', options=_RELATIVE_PATH_OPTIONS)
    subject_id: IntProperty(default=-1)
    phase: StringProperty(options={'SKIP_SAVE'})
    progress_current: IntProperty(options={'SKIP_SAVE'})
    progress_total: IntProperty(options={'SKIP_SAVE'})
    progress_detail: StringProperty(options={'SKIP_SAVE'})
    video: StringProperty(name='Video', subtype='FILE_PATH', options=_RELATIVE_PATH_OPTIONS)
    checkpoint: StringProperty(name='Checkpoint / Hub repo', default='ZeyuLing/Motius-FlowHMR-Latest')
    mode: EnumProperty(name='Inference', items=[
        ('STANDARD', 'Standard / 标准版', 'Original SAM conditioning with shared camera and memory optimizations'),
        ('LIGHTWEIGHT', 'Lightweight / 低显存', 'Fast SAM GPU crops and merged full-hand features, query batching and sequential GPU residency')], default=DEFAULT_MODE)
    frontend_precision: EnumProperty(name='Frontend precision', items=[('source', 'Upstream default', 'Use upstream SAM checkpoint configuration and VGGT precision'), ('float16', 'FP16 mixed', 'Use FP16 for the SAM backbone and VGGT aggregator; numerically sensitive operations retain their existing precision'), ('float32', 'FP32 comparison', 'Compute frontends in FP32; SAM checkpoints may already contain BF16 weights, which cannot regain lost precision')], default='source')
    device: StringProperty(name='Device', default='cuda', description='External PyTorch device, e.g. cuda, cuda:1 or cpu')
    output_root: StringProperty(name='Job output', subtype='DIR_PATH', default='//flowhmr_jobs', options=_RELATIVE_PATH_OPTIONS)
    camera_interval: IntProperty(name='Camera interval', default=1, min=1, description='1 preserves per-frame cameras; larger values trade camera accuracy for memory')
    vggt_frame_batch_size: IntProperty(name='VGGT frames per batch', default=8, min=1, max=32)
    show_queue: BoolProperty(name='Task queue', default=True)
    queue_page: IntProperty(name='Queue page', default=1, min=1)
    show_history: BoolProperty(name='Completed and failed tasks', default=False)
    sam_frame_batch_size: IntProperty(name='Frames per feature batch', description='Keep every frame; use a smaller batch if GPU memory is limited', default=8, min=1, max=8)
    query_chunk_size: IntProperty(name='Attention query batch', default=256, min=1, max=4096)
    seed: IntProperty(name='Seed', default=0, min=0)
    max_frames: IntProperty(name='Frame limit (0 = all)', default=0, min=0)
    postprocess: BoolProperty(name='Motion post-processing', default=False)
    surface: BoolProperty(name='Show reconstructed surface', default=True)
    preprocessing_cache: StringProperty(name='Verified preprocessing cache', subtype='DIR_PATH', options=_RELATIVE_PATH_OPTIONS)
    use_preprocessing_cache: BoolProperty(name='Reuse video preprocessing', default=False,
        description='Validate the video and cached tracking/camera/full-hand features, then run motion sampling again')
    ground_floor: bpy.props.FloatProperty(name='Floor height', default=0., subtype='DISTANCE')
    ground_message: StringProperty(options={'SKIP_SAVE'})
    show_surface: BoolProperty(name='Show raw reconstruction', default=True, update=_surface_visibility)
    follow_view: BoolProperty(name='Follow character in viewport', default=False)
    import_mode: EnumProperty(name='Destination', items=[
        ('NATIVE', 'New SMPL-H skeleton (52 joints)', 'Create an animated body and hands armature'),
        ('MIXAMO', 'Mixamo character', 'Retarget to a local editable Mixamo armature')], default='NATIVE')
    target: PointerProperty(name='Mixamo armature', type=bpy.types.Object, poll=lambda self, obj: obj.type == 'ARMATURE')
    result_armature: PointerProperty(type=bpy.types.Object)
    auto_import: BoolProperty(name='Import when finished (legacy)', default=False)
    start_frame: IntProperty(name='Start frame', default=1, min=1)
    root_motion: BoolProperty(name='Root motion', default=True, description='Disable for in-place horizontal motion; preserve vertical motion')
    auto_scale: BoolProperty(name='Scale travel to character', default=True)
    show_setup: BoolProperty(name='Backend setup', default=False)
    show_advanced: BoolProperty(name='Advanced settings', default=False)
    motion_file: StringProperty(name='Blender motion NPZ', subtype='FILE_PATH', options=_RELATIVE_PATH_OPTIONS)
    capture_file: StringProperty(name='Existing FlowHMR result.npz', subtype='FILE_PATH', options=_RELATIVE_PATH_OPTIONS)
    job_directory: StringProperty(subtype='DIR_PATH', options=_RELATIVE_PATH_OPTIONS)
    state: StringProperty(default='idle', options={'SKIP_SAVE'})
    message: StringProperty(default='Choose a video and check the backend.', options={'SKIP_SAVE'})


def _path(value):
    # Resolving venv/bin/python's symlink selects the base interpreter and
    # silently loses its installed packages. Keep the user's executable path.
    return str(Path(bpy.path.abspath(value)).expanduser().absolute()) if value.strip() else ''


def _runtime_status(home):
    try:
        return json.loads((Path(home)/'install-status.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return {}


def _install_process_matches(home, state):
    try:
        pid = int(state['installer_pid'])
        if os.name == 'nt':
            birth = state.get('installer_start')
            return bool(birth and queue_runner.alive(pid, birth))
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        args = [arg.decode(errors='replace') for arg in args if arg]
        return (stat[19] == state.get('installer_start') and stat[0] != 'Z' and os.getpgid(pid) == pid
                and any(Path(arg).name == 'runtime_bootstrap.py' for arg in args)
                and '--home' in args and args[args.index('--home')+1] == str(home))
    except (OSError, KeyError, ValueError, IndexError):
        return False


class _ReattachedInstall:
    def __init__(self, home, state):
        self.home, self.state, self.pid = home, state, int(state['installer_pid'])
    def poll(self):
        if _install_process_matches(self.home, self.state):
            return None
        return 0 if _runtime_status(self.home).get('state') == 'completed' else 1


def _use_runtime(context, data):
    prefs = _preferences(context)
    prefs.external_python = data['python']
    prefs.project_root = data['project']
    prefs.cache_dir = data.get('cache_dir', '')
    prefs.ffmpeg_path = data.get('ffmpeg_path', '')
    context.scene.flowhmr.checkpoint = data['checkpoint']


@persistent
def _restore_installed_runtime(_unused=None):
    global _RUNTIME
    try:
        context = bpy.context
        prefs = _preferences(context)
        if _RUNTIME is not None:
            _RUNTIME['scene'] = context.scene
        home = Path(_path(prefs.runtime_home))
        state = _runtime_status(home)
        if _RUNTIME is None and state.get('state') == 'running' and _install_process_matches(home, state):
            _RUNTIME = dict(process=_ReattachedInstall(home, state), log=None, home=str(home), scene=context.scene, cancelled=False)
            if not bpy.app.timers.is_registered(_runtime_tick):
                bpy.app.timers.register(_runtime_tick, first_interval=.5, persistent=True)
        data = json.loads((home/'runtime.json').read_text(encoding='utf-8'))
        if not Path(data['python']).is_file() or not Path(data['project']).is_dir() or not Path(data['checkpoint']).is_dir():
            return
        if not prefs.external_python and not prefs.project_root:
            _use_runtime(context, data)
        elif _path(prefs.external_python) == data['python'] and _path(prefs.project_root) == data['project']:
            prefs.ffmpeg_path = data.get('ffmpeg_path', '')
            if context.scene.flowhmr.checkpoint == 'ZeyuLing/Motius-FlowHMR-Latest':
                context.scene.flowhmr.checkpoint = data['checkpoint']
    except (OSError, ValueError, KeyError, AttributeError, RuntimeError):
        pass


def _runtime_tick():
    global _RUNTIME
    if _RUNTIME is None:
        return None
    process = _RUNTIME['process']
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == 'VIEW_3D': area.tag_redraw()
    code = process.poll()
    if code is None:
        if _RUNTIME.get('cancelled_at') and time.monotonic()-_RUNTIME['cancelled_at'] > 3 and os.name != 'nt':
            try: os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError: pass
        return .5
    if _RUNTIME['log'] is not None:
        _RUNTIME['log'].close()
    state = _runtime_status(_RUNTIME['home'])
    scene = _RUNTIME['scene']
    if _RUNTIME.get('cancelled'):
        scene.flowhmr.message = 'Runtime installation cancelled; previous runtime remains available.'
    elif code == 0 and state.get('state') == 'completed':
        try:
            _use_runtime(bpy.context, state)
            scene.flowhmr.message = state.get('message', 'Runtime installed')
        except Exception as error:
            scene.flowhmr.message = f'Runtime installed; could not apply settings: {error}'
    else:
        scene.flowhmr.message = ('Runtime installation interrupted; retry or see setup.log' if state.get('state') == 'running'
                                 else state.get('message', 'Runtime installation failed; see setup.log'))
    _RUNTIME = None
    return None


class FLOWHMR_OT_install_runtime(Operator):
    bl_idname = 'flowhmr.install_runtime'
    bl_label = 'Install FlowHMR Runtime'

    def execute(self, context):
        global _RUNTIME
        try:
            if _RUNTIME is not None:
                raise RuntimeError('Runtime installation is already running')
            if not bpy.app.online_access:
                raise RuntimeError('Enable Blender online access to download Python, dependencies and weights')
            prefs = _preferences(context)
            if not prefs.runtime_home.strip():
                raise RuntimeError('Choose a runtime folder')
            home = Path(_path(prefs.runtime_home))
            existing = _runtime_status(home)
            if existing.get('state') == 'running' and _install_process_matches(home, existing):
                _RUNTIME = dict(process=_ReattachedInstall(home, existing), log=None, home=str(home), scene=context.scene, cancelled=False)
                if not bpy.app.timers.is_registered(_runtime_tick):
                    bpy.app.timers.register(_runtime_tick, first_interval=.5, persistent=True)
                return {'FINISHED'}
            bundle = Path(__file__).with_name('backend.tar.gz')
            requirements = Path(__file__).with_name('runtime_requirements.txt')
            if not bundle.is_file():
                raise RuntimeError('Install the packaged Motius FlowHMR extension ZIP to use one-click setup')
            home.mkdir(parents=True, exist_ok=True)
            log = (home/'setup.log').open('ab', buffering=0)
            command = [sys.executable, '-u', str(Path(__file__).with_name('runtime_bootstrap.py')),
                       '--home', str(home), '--bundle', str(bundle), '--requirements', str(requirements),
                       '--body', _path(prefs.body_model)]
            try:
                process = subprocess.Popen(command, env=jobs.clean_environment(), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=subprocess.STDOUT,
                    **({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
                       if os.name == 'nt' else {'start_new_session': True}))
            except Exception:
                log.close()
                raise
            _RUNTIME = dict(process=process, log=log, home=str(home), scene=context.scene, cancelled=False)
            context.scene.flowhmr.message = 'Installing runtime in the background'
            if not bpy.app.timers.is_registered(_runtime_tick):
                bpy.app.timers.register(_runtime_tick, first_interval=.5, persistent=True)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class FLOWHMR_OT_cancel_runtime(Operator):
    bl_idname = 'flowhmr.cancel_runtime'
    bl_label = 'Cancel runtime installation'

    def execute(self, context):
        if _RUNTIME and _RUNTIME['process'].poll() is None:
            _RUNTIME['cancelled'] = True
            _RUNTIME['cancelled_at'] = time.monotonic()
            previous_state = _runtime_status(_RUNTIME['home'])
            try:
                path = Path(_RUNTIME['home'])/'install-status.json'
                tmp = path.with_suffix('.tmp')
                tmp.write_text(json.dumps(dict(state='cancelled', message='Installation cancelled')), encoding='utf-8')
                os.replace(tmp, path)
            except OSError:
                pass
            try:
                if os.name == 'nt':
                    process = _RUNTIME['process']
                    birth = previous_state.get('installer_start')
                    if birth and queue_runner.alive(process.pid, birth):
                        queue_runner.cancel_recovered(process.pid, birth, force=True)
                    elif not isinstance(process, _ReattachedInstall):
                        subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    os.killpg(_RUNTIME['process'].pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        return {'FINISHED'}


class FLOWHMR_OT_use_runtime(Operator):
    bl_idname = 'flowhmr.use_runtime'
    bl_label = 'Use installed runtime'

    def execute(self, context):
        try:
            prefs = _preferences(context)
            data = json.loads((Path(_path(prefs.runtime_home))/'runtime.json').read_text(encoding='utf-8'))
            if not Path(data['python']).is_file() or not Path(data['project']).is_dir() or not Path(data['checkpoint']).is_dir():
                raise RuntimeError('The installed runtime is incomplete or has moved')
            _use_runtime(context, data)
            context.scene.flowhmr.message = data.get('message', 'Runtime selected')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


def _draw_runtime(layout, context):
    prefs = _preferences(context)
    state = _runtime_status(_path(prefs.runtime_home)) if prefs.runtime_home else {}
    box = layout.box()
    if _RUNTIME is not None:
        box.label(text=state.get('message', 'Starting runtime installation'))
        stages = {'bootstrap': 1, 'backend': 2, 'python': 3, 'pytorch': 4, 'dependencies': 5, 'weights': 6, 'check': 7}
        if state.get('phase') in stages:
            box.label(text=f"Step {stages[state['phase']]} / 7")
        total, current = state.get('total', 0), state.get('current', 0)
        if total:
            text = f'{current/1024**3:.2f} / {total/1024**3:.2f} GiB' if state.get('unit') == 'bytes' else f'{current}/{total}'
            box.progress(factor=max(0., min(1., current/total)), text=text)
            if state.get('bytes_per_second'):
                box.label(text=f"{state['bytes_per_second']/1024**2:.1f} MiB/s average")
        elif state.get('detail'):
            box.label(text=state['detail'][:75])
        box.operator('flowhmr.cancel_runtime', text='Cancel installation', icon='CANCEL')
    else:
        box.operator('flowhmr.install_runtime', text='Install / update runtime', icon='IMPORT')
        if prefs.runtime_home and (Path(_path(prefs.runtime_home))/'runtime.json').is_file():
            box.operator('flowhmr.use_runtime', text='Use installed runtime', icon='CHECKMARK')
        box.label(text='Neutral SMPL-H model')
        box.prop(prefs, 'body_model', text='')
        if state.get('state') == 'failed':
            box.label(text=state.get('message', 'Setup failed')[:75], icon='ERROR')
            box.label(text='See setup.log in the runtime folder')
        if prefs.external_python and prefs.project_root:
            box.operator('flowhmr.run', text='Check environment', icon='CHECKMARK').kind='doctor'


def _destination(props):
    if props.import_mode == 'MIXAMO':
        if props.target is None:
            raise ValueError('Choose the Mixamo armature or import its FBX first')
        from .motion_data import match_mixamo
        match_mixamo([bone.name for bone in props.target.data.bones])
    return dict(target=props.target if props.import_mode == 'MIXAMO' else None,
                start_frame=props.start_frame, root_motion=props.root_motion, auto_scale=props.auto_scale)


def _import(context, props, options=None):
    path = _path(props.motion_file)
    if not Path(path).is_file():
        raise ValueError('Choose the generated motion.blender.npz file')
    target = animation.import_motion(context, path, **(options or _destination(props)))
    if (options or _destination(props)).get('target') is None and props.surface:
        animation.import_surface(context, path, target, props.start_frame)
        _surface_visibility(props, context)
    props.result_armature = target
    return target


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type in {'VIEW_3D','CLIP_EDITOR'}:
                area.tag_redraw()


def _tick():
    global _ACTIVE, _QUEUE_STAMP
    if _ACTIVE is None:
        from .task_queue import Queue
        stamp=Queue().revision()
        if stamp != _QUEUE_STAMP:
            _QUEUE_STAMP=stamp
            _redraw()
        return 1.0
    job, scene = _ACTIVE['job'], _ACTIVE['scene']
    try:
        props = scene.flowhmr
    except ReferenceError:
        _cleanup()
        return None
    status = job.read_status()
    props.job_directory = str(job.directory)
    props.state = status.get('state', 'running')
    props.phase = status.get('phase', status.get('state','').title())
    props.progress_current = status.get('current', 0)
    props.progress_total = status.get('total', 0)
    rate, eta = status.get('rate_per_second', 0), status.get('eta_seconds', -1)
    props.progress_detail = (f'{rate:.1f} {status.get("rate_unit", "units/s")} · about {int(eta)//60}:{int(eta)%60:02d} left in this stage'
                             if rate > 0 and eta >= 0 else '')
    code = job.process.poll()
    if job.cancel_started is not None:
        props.message = 'Cancelling external inference…'
        if code is None and time.monotonic()-job.cancel_started > 3:
            job.stop(force=True)
    else:
        props.message = status.get('message', 'Running…')[:500]
    if code is None:
        _redraw()
        return .5
    job.close()
    _ACTIVE = None
    if job.cancel_started is not None or status.get('state') == 'cancelled':
        props.state, props.message = 'cancelled', 'Cancelled. Partial files remain in the job directory.'
    elif code != 0 or status.get('state') != 'completed':
        props.state = 'failed'
        props.message = status.get('message', 'External Python failed; open job.log')[:500]
        if status.get('state') == 'running':
            props.message = f'External Python exited with code {code}; open job.log'
    else:
        props.state = 'completed'
        props.message = 'Result ready · choose a task from the queue'
    _redraw()
    return 1.0


@persistent
def _cleanup(_unused=None):
    global _ACTIVE, _PLAYBACK
    if _PLAYBACK is not None:
        if _PLAYBACK.get('timer'):
            bpy.context.window_manager.event_timer_remove(_PLAYBACK['timer'])
        _finish_playback()
    if viewport_preview._BUILD is not None:
        viewport_preview._BUILD.finish(bpy.context,False)
    viewport_preview.clear()
    # Queue workers are independent: unloading/changing .blend never cancels them.
    _ACTIVE = None
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)


class FLOWHMR_OT_run(Operator):
    bl_idname = 'flowhmr.run'
    bl_label = 'Generate Motion'
    kind: EnumProperty(items=[('subjects', 'Detect People', ''), ('inference', 'Inference', ''), ('doctor', 'Check Environment', ''), ('capture', 'Prepare Capture', '')])

    def execute(self, context):
        global _ACTIVE
        try:
            props, prefs = context.scene.flowhmr, _preferences(context)
            python, project = _path(prefs.external_python), _path(prefs.project_root)
            if not python or not Path(python).is_file():
                raise ValueError('Set External Python to the Motius environment interpreter')
            if not project or not (Path(project)/'motius').is_dir():
                raise ValueError('Set Motius checkout to the repository directory')
            destination = {}
            if self.kind in {'inference', 'subjects'} and not props.device.strip().startswith('cuda'):
                raise ValueError('Queued inference requires a CUDA device; CPU/MPS execution is not validated')
            if self.kind in {'inference', 'subjects'} and not Path(_path(props.video)).is_file():
                raise ValueError('Choose an existing video file')
            if self.kind == 'inference':
                if props.subject_id < 0:
                    raise ValueError('Detect people and choose your subject first')
                from .subjects import selected_boxes
                selected_boxes(_path(props.subjects_file), _path(props.video), props.subject_id)
            if self.kind == 'capture' and not Path(_path(props.capture_file)).is_file():
                raise ValueError('Choose an existing FlowHMR result.npz')
            if props.output_root.startswith('//') and not bpy.data.is_saved:
                raise ValueError('Save the .blend file or choose an absolute Job output directory')
            if not props.output_root.strip():
                raise ValueError('Choose a Job output directory')
            directory = jobs.new_directory(_path(props.output_root))
            checkpoint = props.checkpoint.strip()
            if checkpoint.startswith('//') or Path(checkpoint).expanduser().exists():
                checkpoint = _path(checkpoint)
            request = dict(kind=self.kind if self.kind in {'doctor', 'subjects'} else 'inference',
                subjects_file=_path(props.subjects_file) if self.kind == 'inference' else '', subject_id=props.subject_id,
                project_root=project, output_dir=str(directory), body_model=_path(prefs.body_model),
                cache_dir=_path(prefs.cache_dir), video=_path(props.video), checkpoint=checkpoint,
                mode=props.mode, device=props.device.strip(), frontend_precision=props.frontend_precision, seed=props.seed,
                max_frames=props.max_frames, camera_interval=props.camera_interval,
                query_chunk_size=props.query_chunk_size, sam_frame_batch_size=props.sam_frame_batch_size,
                vggt_frame_batch_size=props.vggt_frame_batch_size, postprocess=props.postprocess,
                surface=props.surface,
                ffmpeg_path=prefs.ffmpeg_path,
                preprocessing_cache=_path(props.preprocessing_cache) if props.use_preprocessing_cache else '',
                offline=not bpy.app.online_access)
            if self.kind == 'capture':
                request['capture'] = _path(props.capture_file)
            job = jobs.QueuedJob(python, request)
            _ACTIVE = dict(job=job, scene=context.scene, auto_import=False, started_at=time.monotonic(),
                           destination=destination)
            props.job_directory, props.state, props.message = str(directory), 'queued', 'Added to task queue'
            if not bpy.app.timers.is_registered(_tick):
                bpy.app.timers.register(_tick, first_interval=.5, persistent=True)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class FLOWHMR_OT_cancel(Operator):
    bl_idname = 'flowhmr.cancel'
    bl_label = 'Cancel Inference'

    def execute(self, context):
        if _ACTIVE is not None:
            _ACTIVE['job'].stop()
        return {'FINISHED'}


class FLOWHMR_OT_preview(Operator):
    """Open the source video in a Movie Clip Editor and synchronize the timeline."""
    bl_idname = 'flowhmr.preview_video'
    bl_label = 'Preview Input Video'

    def execute(self, context):
        try:
            props=context.scene.flowhmr
            path = _path(props.video)
            if props.subjects_file:
                import json
                from .subjects import digest
                try:
                    from .subjects import load_subjects
                    data=load_subjects(_path(props.subjects_file))
                    if data['video_sha256']==digest(path) and Path(data.get('preview_sequence','')).is_file():
                        path=data['preview_sequence']
                except (OSError,ValueError,KeyError):
                    pass
            clip = next((c for c in bpy.data.movieclips if _path(c.filepath) == path), None)
            clip = clip or bpy.data.movieclips.load(path)
            clip.frame_start = 1
            areas = [a for a in context.screen.areas if a.type == 'CLIP_EDITOR']
            if not areas:
                self.report({'INFO'}, 'Open a Movie Clip Editor to see the input video')
            for area in areas:
                area.spaces.active.clip = clip
                area.spaces.active.mode = 'TRACKING'
                area.spaces.active.view = 'CLIP'
                with context.temp_override(area=area, region=next(r for r in area.regions if r.type == 'WINDOW')):
                    bpy.ops.clip.view_all(fit_view=True)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class FLOWHMR_OT_apply(Operator):
    """Retarget the generated motion onto the selected destination armature."""
    bl_idname = 'flowhmr.apply_motion'
    bl_label = 'Retarget Loaded Motion'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        try:
            _import(context, context.scene.flowhmr)
            context.scene.flowhmr.message = 'Retarget complete · body and hands'
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class FLOWHMR_OT_ground(Operator):
    """Detect foot support and bake reversible contact grounding with planted-foot IK."""
    bl_idname = 'flowhmr.ground_motion'
    bl_label = 'Ground & Plant Feet'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .grounding import ground
        props = context.scene.flowhmr
        arm = props.target if props.import_mode == 'MIXAMO' else props.result_armature
        try:
            context.window_manager.progress_begin(0, 100)
            props.message = 'Measuring soles, detecting support and solving leg IK…'
            def progress(value, text):
                props.message = text
                context.window_manager.progress_update(value)
                if not bpy.app.background:
                    bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1)
            report = ground(context, arm, props.ground_floor, progress=progress)
            props.ground_message = f"Support sole error: {report['support_sole_after_abs_mean_m']*1000:.1f} mm · {report['verified_flight_frames']} flight frames kept"
            props.message = 'Grounded action ready · Raw remains available'
            props.result_armature = arm
            if props.job_directory:
                import json
                (Path(_path(props.job_directory))/'grounding.json').write_text(json.dumps(report, indent=2)+'\n')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            props.message = str(error)
            return {'CANCELLED'}
        finally:
            context.window_manager.progress_end()


class FLOWHMR_OT_ground_switch(Operator):
    bl_idname = 'flowhmr.ground_switch'
    bl_label = 'Compare Grounding'
    kind: EnumProperty(items=[('RAW', 'Raw', ''), ('GROUNDED', 'Grounded', '')])

    def execute(self, context):
        props = context.scene.flowhmr
        arm = props.target if props.import_mode == 'MIXAMO' else props.result_armature
        key = 'flowhmr_raw_action' if self.kind == 'RAW' else 'flowhmr_grounded_action'
        name = arm.get(key) if arm else None
        action = bpy.data.actions.get(name or '')
        if action is None:
            self.report({'ERROR'}, 'Apply grounding before comparing actions')
            return {'CANCELLED'}
        arm.animation_data.action = action
        context.scene.frame_set(context.scene.frame_current)
        props.message = 'Playing '+self.kind.title()+' action'
        return {'FINISHED'}


class FLOWHMR_OT_play_all(Operator):
    bl_idname = 'flowhmr.play_all'
    bl_label = 'Play at Original Speed'
    bl_description = 'Prepare a viewport cache, then play all frames at the original rate'

    def execute(self, context):
        global _PLAYBACK
        if _PLAYBACK is not None:
            _PLAYBACK['cancel']=True
            return {'FINISHED'}
        if not viewport_preview.ready(context):
            return bpy.ops.flowhmr.prepare_playback()
        if context.screen.is_animation_playing:
            bpy.ops.screen.animation_cancel(restore_frame=False)
        sidebar=context.space_data.show_region_ui
        context.space_data.show_region_ui=False
        scene=context.scene
        scene.sync_mode='NONE'
        context.screen.use_play_clip_editors=True
        context.screen.use_play_3d_editors=True
        self.timer=context.window_manager.event_timer_add(.002, window=context.window)
        self.scene=scene
        _PLAYBACK=dict(scene=scene, frame=scene.frame_start, frames=[scene.frame_start], started_at=time.monotonic(), timer=self.timer,
                       area=context.area, sidebar=sidebar, shading=context.space_data.shading.type, clip_areas=[a for a in context.screen.areas if a.type=='CLIP_EDITOR' and a.spaces.active.clip],
                       stage_drawn=-1, clip_drawn=-1, stage_frames=[], clip_frames=[],
                       period=scene.render.fps_base/scene.render.fps, presentation_times=[],
                       visibility=[(o,o.hide_viewport) for o in scene.objects if o.type=='MESH'])
        scene.frame_set(scene.frame_start)
        _PLAYBACK['source_clips']=[(a.spaces.active,a.spaces.active.clip) for a in _PLAYBACK['clip_areas']]
        if viewport_preview._CACHE['source_frames']:
            for space,clip in _PLAYBACK['source_clips']:space.clip=None
        for obj, _ in _PLAYBACK['visibility']: obj.hide_viewport=True
        context.space_data.shading.type='WIREFRAME'
        _PLAYBACK['drawers']=[(owner,owner.draw_handler_add(_playback_drawn,(kind,),'WINDOW','POST_PIXEL'))
                              for owner,kind in [(bpy.types.SpaceView3D,'stage'),(bpy.types.SpaceClipEditor,'clip')]]
        _redraw()
        scene.flowhmr.message='Cached viewport · original speed · Space or Esc to stop'
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def modal(self, context, event):
        global _PLAYBACK
        if _PLAYBACK is None:
            return {'FINISHED'}
        cancelled=_PLAYBACK.get('cancel') or (event.type in {'ESC','SPACE'} and event.value=='PRESS')
        displayed=(_PLAYBACK['stage_drawn']==_PLAYBACK['frame'] and (not _PLAYBACK['clip_areas'] or _PLAYBACK['clip_drawn']==_PLAYBACK['frame']))
        elapsed=time.monotonic()-_PLAYBACK['started_at']
        due=(_PLAYBACK['frame']-self.scene.frame_start+1)*_PLAYBACK['period']
        if cancelled or (event.type=='TIMER' and displayed and elapsed>=due and _PLAYBACK['frame']>=self.scene.frame_end):
            context.window_manager.event_timer_remove(self.timer)
            _finish_playback()
            if cancelled:self.scene.flowhmr.message='Playback stopped · no frames skipped'
            return {'FINISHED'}
        if event.type=='TIMER' and elapsed>due+.1:
            context.window_manager.event_timer_remove(self.timer)
            _finish_playback()
            self.scene.flowhmr.message='Playback stopped: display could not sustain the original frame rate'
            self.report({'WARNING'},self.scene.flowhmr.message)
            return {'CANCELLED'}
        if event.type=='TIMER' and displayed and elapsed>=due:
            _PLAYBACK['frame']+=1
            _PLAYBACK['frames'].append(_PLAYBACK['frame'])
            _redraw()
            return {'RUNNING_MODAL'}
        return {'RUNNING_MODAL'}


class FLOWHMR_OT_lineup(Operator):
    bl_idname = 'flowhmr.lineup'
    bl_label = 'Arrange Characters'
    bl_options = {'REGISTER', 'UNDO'}
    bl_description = 'Arrange all FlowHMR Mixamo characters side by side without changing their animation'

    def execute(self, context):
        from mathutils import Vector
        arms = [o for o in context.scene.objects if o.type == 'ARMATURE' and o.animation_data and o.animation_data.action and o.animation_data.action.get('flowhmr_source') and not o.get('flowhmr_native')]
        arms = [o for o in arms if any('mixamorig' in b.name.lower() for b in o.data.bones)]
        if not arms:
            self.report({'ERROR'}, 'Import and retarget your characters first');return {'CANCELLED'}
        for i, arm in enumerate(sorted(arms, key=lambda o:o.name)):
            desired = Vector(((i-(len(arms)-1)/2)*2.2, 0, 0))
            delta = desired-arm.matrix_world.translation
            arm.matrix_world.translation += delta
            for mesh in context.scene.objects:
                if mesh.type == 'MESH' and mesh.parent != arm and any(m.type=='ARMATURE' and m.object==arm for m in mesh.modifiers):
                    ancestor = mesh.parent
                    while ancestor is not None and ancestor != arm:
                        ancestor = ancestor.parent
                    if ancestor is None:
                        mesh.matrix_world.translation += delta
        context.scene.flowhmr.follow_view = False
        for area in context.screen.areas:
            if area.type=='VIEW_3D':
                area.spaces.active.region_3d.view_location = Vector((0,0,1))
                area.spaces.active.region_3d.view_distance = max(5.,len(arms)*1.95)
        return {'FINISHED'}


class FLOWHMR_OT_animate_cast(Operator):
    bl_idname = 'flowhmr.animate_cast'
    bl_label = 'Animate and Ground All Characters'
    bl_description = 'Retarget the loaded motion and solve floor contacts separately for each imported character'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        from .grounding import ground, ground_many, _independent_cast
        props=context.scene.flowhmr
        arms=[o for o in context.scene.objects if o.type=='ARMATURE' and o.get('flowhmr_character')]
        if not arms or not props.motion_file:
            self.report({'ERROR'},'Generate motion and add characters first');return {'CANCELLED'}
        try:
            props.show_surface=False
            props.state='processing'
            context.window_manager.progress_begin(0,100)
            shared = _independent_cast(context, arms)
            for i,arm in enumerate(arms):
                props.target=arm;props.import_mode='MIXAMO'
                props.message=f'Applying motion · {arm.name} · {i+1}/{len(arms)}'
                if not bpy.app.background:bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1)
                _import(context,props)
                if not shared:
                    def serial_progress(value, text, index=i, name=arm.name):
                        props.message=f'{name} · {text}'
                        context.window_manager.progress_update((index+value/100)/len(arms)*100)
                        if not bpy.app.background:bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1)
                    ground(context,arm,props.ground_floor,progress=serial_progress)
            def progress(value, text):
                props.message=text
                context.window_manager.progress_update(value)
                if not bpy.app.background:bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP',iterations=1)
            if shared:
                ground_many(context,arms,props.ground_floor,progress=progress)
            bpy.ops.flowhmr.lineup()
            props.show_surface=False
            props.state='completed'
            props.message=f'{len(arms)} characters ready · grounded separately'
            return {'FINISHED'}
        except Exception as error:
            props.state='failed';props.message=str(error);self.report({'ERROR'},str(error));return {'CANCELLED'}
        finally:context.window_manager.progress_end()


class FLOWHMR_OT_import(Operator, ImportHelper):
    bl_idname = 'flowhmr.import_motion'
    bl_label = 'Import Motion'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.npz'
    filter_glob: StringProperty(default='*.npz', options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.flowhmr
        try:
            props.motion_file = self.filepath
            _import(context, props)
            props.message = 'Animation imported'
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class FLOWHMR_OT_fbx(Operator, ImportHelper):
    bl_idname = 'flowhmr.import_character'
    bl_label = 'Import Mixamo Character'
    bl_options = {'REGISTER', 'UNDO'}
    filename_ext = '.fbx'
    filter_glob: StringProperty(default='*.fbx;*.glb;*.gltf', options={'HIDDEN'})

    def execute(self, context):
        before = set(bpy.data.objects)
        try:
            if Path(self.filepath).suffix.lower() in {'.glb', '.gltf'}:
                bpy.ops.import_scene.gltf(filepath=self.filepath)
            else:
                bpy.ops.import_scene.fbx(filepath=self.filepath, use_anim=False)
            arms = [o for o in set(bpy.data.objects)-before if o.type == 'ARMATURE']
            if len(arms) == 1:
                arms[0].animation_data_clear()
                for bone in arms[0].pose.bones:
                    bone.matrix_basis.identity()
                arms[0].name = Path(self.filepath).stem
                arms[0]['flowhmr_character'] = True
                context.scene.flowhmr.target = arms[0]
                context.scene.flowhmr.import_mode = 'MIXAMO'
                for obj in context.selected_objects:
                    obj.select_set(False)
                arms[0].select_set(True)
                context.view_layer.objects.active = arms[0]
            else:
                self.report({'WARNING'}, 'Select the intended Mixamo armature in Destination')
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}


class FLOWHMR_OT_export(Operator, ExportHelper):
    """Export the current animated character and selected Action as FBX."""
    bl_idname = 'flowhmr.export_fbx'
    bl_label = 'Export Animated FBX'
    filename_ext = '.fbx'
    filter_glob: StringProperty(default='*.fbx', options={'HIDDEN'})

    def execute(self, context):
        props = context.scene.flowhmr
        target = props.result_armature
        selected, active = list(context.selected_objects), context.view_layer.objects.active
        scene = context.scene
        frames = (scene.frame_start, scene.frame_end, scene.frame_current, scene.render.fps, scene.render.fps_base)
        try:
            if context.mode != 'OBJECT' or target is None or target.animation_data is None or target.animation_data.action is None:
                raise ValueError('Import a motion in Object Mode before exporting')
            for obj in selected:
                obj.select_set(False)
            target.select_set(True)
            for obj in context.scene.objects:
                if obj.type == 'MESH' and any(m.type == 'ARMATURE' and m.object == target for m in obj.modifiers):
                    obj.select_set(True)
            context.view_layer.objects.active = target
            action = target.animation_data.action
            scene.frame_start, scene.frame_end = (math.floor(action.frame_range[0]), math.ceil(action.frame_range[1]))
            fps = float(action.get('flowhmr_fps', 30.))
            scene.render.fps = round(fps)
            scene.render.fps_base = scene.render.fps/fps
            bpy.ops.export_scene.fbx(filepath=self.filepath, use_selection=True,
                object_types={'ARMATURE', 'MESH'}, add_leaf_bones=False,
                bake_anim=True, bake_anim_use_nla_strips=False, bake_anim_use_all_actions=False,
                bake_anim_simplify_factor=0.)
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'}, str(error))
            return {'CANCELLED'}
        finally:
            scene.frame_start, scene.frame_end = frames[:2]
            scene.render.fps, scene.render.fps_base = frames[3:]
            scene.frame_set(frames[2])
            for obj in context.selected_objects:
                obj.select_set(False)
            for obj in selected:
                obj.select_set(True)
            context.view_layer.objects.active = active


class FLOWHMR_OT_open(Operator):
    bl_idname = 'flowhmr.open_output'
    bl_label = 'Open Output Folder'
    log: BoolProperty(default=False)

    def execute(self, context):
        path = Path(_path(context.scene.flowhmr.job_directory))
        if self.log:
            path = path/'job.log'
        if not path.exists():
            self.report({'ERROR'}, 'No job output yet')
            return {'CANCELLED'}
        bpy.ops.wm.path_open(filepath=str(path))
        return {'FINISHED'}



@persistent
def _restore_queue(_unused=None):
    if not bpy.app.timers.is_registered(_tick):
        bpy.app.timers.register(_tick, first_interval=1., persistent=True)
    from .task_queue import Queue
    from .queue_runner import start
    queue = Queue()
    pending = next((r for r in queue.list(100000) if r['state'] in {'queued','running','retrying'}),None)
    if pending:
        try:start(pending['python'],queue.path)
        except OSError as error:queue.update(pending['id'],message=f'Cannot start queue runner: {error}. Check External Python and resubmit if its path changed.')


class FLOWHMR_OT_queue_action(Operator):
    bl_idname = 'flowhmr.queue_action'
    bl_label = 'Task action'
    identity: StringProperty()
    action: StringProperty()

    def execute(self, context):
        from .task_queue import Queue
        from .queue_runner import start
        import json
        queue = Queue()
        try:
            row = queue.get(self.identity) if self.identity else None
            if self.action == 'PAUSE':
                queue.pause(True);return {'FINISHED'}
            if self.action == 'RESUME':
                queue.pause(False)
                rows = queue.list(100000)
                queued = next((r for r in rows if r['state'] in {'queued','running','retrying'}),None)
                if queued: start(queued['python'],queue.path)
                return {'FINISHED'}
            if row is None:
                raise ValueError('Task no longer exists')
            if self.action == 'CANCEL': queue.cancel(self.identity)
            elif self.action == 'RETRY':
                queue.retry(self.identity);start(row['python'],queue.path)
            elif self.action in {'UP','DOWN'}: queue.move(self.identity,-1 if self.action == 'UP' else 1)
            elif self.action == 'FILES':
                bpy.ops.wm.path_open(filepath=str(Path(row['directory'])))
            elif self.action == 'IMPORT':
                if row['state'] != 'completed': raise ValueError('Wait for this task to finish')
                status = queue.status(row);props = context.scene.flowhmr
                if status.get('subjects_file'):
                    request = json.loads(row['request'])
                    props.video = request['video'];props.subjects_file = status['subjects_file'];props.subject_id = -1
                    bpy.ops.flowhmr.choose_subject('INVOKE_DEFAULT')
                elif status.get('motion_file'):
                    if context.mode != 'OBJECT': raise ValueError('Return to Object Mode to import the result')
                    result=bpy.ops.flowhmr.import_motion(filepath=status['motion_file'])
                    if 'FINISHED' not in result:return {'CANCELLED'}
                    props.state,props.message = 'completed','Result imported'
                else:
                    self.report({'INFO'},status.get('message','Environment check completed'))
            _redraw()
            return {'FINISHED'}
        except Exception as error:
            self.report({'ERROR'},str(error))
            return {'CANCELLED'}


def _draw_queue(layout, props):
    from .task_queue import Queue
    import json
    layout.prop(props,'show_queue',toggle=True)
    if not props.show_queue: return
    try:
        queue = Queue();rows = queue.list(100000)
    except Exception as error:
        layout.label(text=f'Queue unavailable: {error}',icon='ERROR');return
    active = [r for r in rows if r['state'] in {'queued','running','retrying'}]
    box = layout.box();box.label(text=f'Background queue · {len(active)} active')
    controls=box.row(align=True)
    controls.operator('flowhmr.queue_action',text='Resume queue' if queue.paused() else 'Reconnect',icon='FILE_REFRESH').action='RESUME'
    controls.operator('flowhmr.queue_action',text='Pause after current').action='PAUSE'
    if queue.paused():box.label(text='Queue paused · current task continues')
    box.prop(props,'show_history',toggle=True)
    visible = rows if props.show_history else active
    page=max(1,min(props.queue_page,max(1,(len(visible)+4)//5)))
    if len(visible)>5:box.prop(props,'queue_page')
    for record in visible[(page-1)*5:page*5]:
        item=box.box();request=json.loads(record['request']);status=queue.status(record)
        name=Path(request.get('video') or request.get('capture') or 'Environment check').name
        item.label(text=f"{name} · {record['state'].title()}")
        item.label(text=f"{request.get('mode','')} · attempt {record['attempt']} · {record['id'][:8]}")
        if record['state']=='running' and status.get('total'):
            item.progress(factor=min(1.,status.get('current',0)/status['total']),text=f"{status.get('phase','Working')} · {status.get('current',0)}/{status['total']}")
        if record['state']=='running' and status.get('rate_per_second',0)>0:
            eta=status.get('eta_seconds',-1)
            item.label(text=f"{status['rate_per_second']:.1f} {status.get('rate_unit','units/s')}"+(f" · about {int(eta)//60}:{int(eta)%60:02d} left in stage" if eta>=0 else ''))
        preflight=Path(status['attempt_directory'])/'preflight.json'
        if preflight.is_file():
            try:
                hardware=json.loads(preflight.read_text())
                item.label(text=f"{hardware['gpu']} · {hardware['free_gib']:.1f} GiB free at start")
                schedule=hardware.get('schedule',{})
                if schedule:item.label(text=f"Batch: SAM {schedule['sam_frame_batch_size']} / VGGT {schedule['vggt_frame_batch_size']}")
            except (OSError,ValueError,KeyError):pass
        import textwrap
        for line in textwrap.wrap(status.get('message',''),width=max(22,int(bpy.context.region.width/7)-4))[:4]:item.label(text=line)
        row=item.row(align=True)
        actions=[('FILES','Files')]
        if record['state'] in {'queued','running','retrying'}:actions.append(('CANCEL','Cancel'))
        if record['state']=='queued':actions.extend([('UP','↑'),('DOWN','↓')])
        if record['state'] in {'failed','interrupted','cancelled'}:actions.append(('RETRY','Retry'))
        if record['state']=='completed':actions.append(('IMPORT','Choose person' if status.get('subjects_file') else 'Import result' if status.get('motion_file') else 'Details'))
        for action,label in actions:
            op=row.operator('flowhmr.queue_action',text=label);op.identity=record['id'];op.action=action
    if len(visible)>5:box.label(text=f'Page {page} / {(len(visible)+4)//5} · {len(visible)} tasks')


class FLOWHMR_PT_main(Panel):
    bl_label = 'FlowHMR · Motius'
    bl_idname = 'FLOWHMR_PT_main'
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'FlowHMR'

    def draw(self, context):
        layout, props = self.layout, context.scene.flowhmr
        layout.use_property_split = False
        busy = _ACTIVE is not None
        prefs = _preferences(context)
        if _RUNTIME is not None or not prefs.external_python or not prefs.project_root:
            _draw_runtime(layout, context)
        source = layout.column()
        source.label(text='Your performance')
        source.enabled = True
        source.prop(props, 'video', text='')
        row = source.row(align=True)
        row.operator('flowhmr.preview_video', text='Preview')
        row.operator('flowhmr.run', text='Detect people').kind = 'subjects'
        if props.subjects_file:
            source.operator('flowhmr.choose_subject', text=f'Person {props.subject_id} · Review selection' if props.subject_id >= 0 else 'Choose person', icon='RESTRICT_SELECT_OFF')
        layout.separator(factor=1.3)
        motion = layout.column()
        motion.label(text='Capture the motion')
        motion.enabled = True
        motion.prop(props, 'mode', text='')
        row = motion.row(); row.scale_y = 1.5
        row.operator('flowhmr.run', text='Add capture to queue', icon='ADD').kind = 'inference'
        if busy or props.state not in {'idle', ''}:
            status = layout.box()
            elapsed = int(time.monotonic()-_ACTIVE['started_at']) if busy else 0
            status.label(text=f'{props.phase or "Working"} · {elapsed//60:02d}:{elapsed%60:02d}' if busy else props.state.title())
            if busy and props.progress_total:
                status.progress(factor=min(1.,props.progress_current/props.progress_total), text=f'{props.progress_current} / {props.progress_total} · {100*props.progress_current/props.progress_total:.0f}%')
                if props.progress_detail:
                    status.label(text=props.progress_detail)
            import textwrap
            for line in textwrap.wrap(props.message, width=max(22,int(context.region.width/7)-4)):
                status.label(text=line)
            if busy:
                status.label(text='You can add characters while this runs.')
                status.operator('flowhmr.cancel', text='Cancel')
        _draw_queue(layout, props)
        layout.separator(factor=1.3)
        cast = layout.column()
        cast.label(text='Bring your characters')
        cast.operator('flowhmr.import_character', text='Add Mixamo character', icon='ADD')
        cast.prop(props, 'target', text='')
        actions = cast.column(); actions.enabled = bool(props.motion_file)
        actions.operator('flowhmr.apply_motion', text='Apply motion to character')
        if props.target:
            actions.operator('flowhmr.ground_motion', text='Plant feet on the floor')
            if props.target.get('flowhmr_grounded_action'):
                row = actions.row(align=True)
                row.operator('flowhmr.ground_switch',text='Original').kind='RAW'
                row.operator('flowhmr.ground_switch',text='Grounded').kind='GROUNDED'
        actions.operator('flowhmr.lineup',text='Arrange characters side by side')
        if sum(bool(o.get('flowhmr_character')) for o in context.scene.objects)>=2:
            actions.operator('flowhmr.animate_cast',text='Animate and ground all characters')
        layout.separator(factor=1.3)
        review = layout.column(); review.enabled = bool(props.motion_file)
        review.label(text='Watch the performance')
        row=review.row();row.scale_y=1.4
        row.operator('flowhmr.play_all',text='Stop playback' if _PLAYBACK else ('Play at original speed' if viewport_preview.ready(context) else 'Prepare & play at original speed'),icon='PAUSE' if _PLAYBACK else 'PLAY')
        review.label(text=f'Cached viewport · {context.scene.render.fps/context.scene.render.fps_base:g} fps')
        review.prop(props,'show_surface')
        review.operator('flowhmr.export_fbx',text='Export animated FBX',icon='EXPORT')
        layout.separator(factor=1.3)
        layout.prop(props,'show_advanced',toggle=True,text='Options')
        if props.show_advanced:
            col=layout.column();col.enabled=True
            for name in ('checkpoint','device','frontend_precision','surface','import_mode','output_root','camera_interval','seed','max_frames','postprocess','start_frame','root_motion','auto_scale','ground_floor','follow_view'):
                col.prop(props,name)
            col.prop(props,'sam_frame_batch_size')
            col.prop(props,'vggt_frame_batch_size')
            col.prop(props,'use_preprocessing_cache')
            if props.use_preprocessing_cache:
                col.prop(props,'preprocessing_cache')
                col.label(text='Exact video + subject + frontend validation')
            col.operator('flowhmr.import_motion')
            col.prop(props,'show_setup',toggle=True)
            if props.show_setup:
                prefs=_preferences(context)
                _draw_runtime(col, context)
                for name in ('runtime_home','external_python','project_root','cache_dir'):
                    col.prop(prefs,name)
            if props.job_directory:
                col.operator('flowhmr.open_output',text='Job files')
                col.operator('flowhmr.open_output',text='Detailed log').log=True


CLASSES = (FLOWHMR_OT_queue_action, FLOWHMR_OT_prepare_playback, FLOWHMR_OT_animate_cast, FLOWHMR_OT_choose_subject, FLOWHMR_OT_play_all, FLOWHMR_OT_lineup, FlowHMRPreferences, FlowHMRSettings, FLOWHMR_OT_install_runtime, FLOWHMR_OT_cancel_runtime, FLOWHMR_OT_use_runtime, FLOWHMR_OT_run, FLOWHMR_OT_cancel,
           FLOWHMR_OT_preview, FLOWHMR_OT_apply, FLOWHMR_OT_ground, FLOWHMR_OT_ground_switch,
           FLOWHMR_OT_import, FLOWHMR_OT_fbx, FLOWHMR_OT_export, FLOWHMR_OT_open, FLOWHMR_PT_main)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.flowhmr = PointerProperty(type=FlowHMRSettings)
    bpy.app.handlers.load_pre.append(_cleanup)
    bpy.app.handlers.frame_change_post.append(_follow_view)
    bpy.app.handlers.depsgraph_update_post.append(viewport_preview.invalidate)
    bpy.app.handlers.load_post.append(_restore_scene_status)
    bpy.app.handlers.load_post.append(_restore_queue)
    bpy.app.handlers.load_post.append(_restore_installed_runtime)
    _restore_queue()
    _restore_installed_runtime()


def unregister():
    _cleanup()
    if _restore_installed_runtime in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_installed_runtime)
    if _restore_queue in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_queue)
    if viewport_preview.invalidate in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(viewport_preview.invalidate)
    if _restore_scene_status in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_restore_scene_status)
    if _follow_view in bpy.app.handlers.frame_change_post:
        bpy.app.handlers.frame_change_post.remove(_follow_view)
    if _cleanup in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(_cleanup)
    del bpy.types.Scene.flowhmr
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
