# SPDX-License-Identifier: GPL-3.0-or-later
"""Owned subprocesses, bounded log reads, and Blender-free job state."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


def clean_environment():
    env = os.environ.copy()
    for key in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONEXECUTABLE', 'PYTHONUSERBASE',
                '__PYVENV_LAUNCHER__', 'BLENDER_SYSTEM_PYTHON'):
        env.pop(key, None)
    for key in ('LD_LIBRARY_PATH', 'DYLD_LIBRARY_PATH', 'DYLD_FRAMEWORK_PATH', 'PATH'):
        if key in env:
            env[key] = os.pathsep.join(p for p in env[key].split(os.pathsep) if 'blender' not in p.lower())
    env['PYTHONUNBUFFERED'] = '1'
    return env


def new_directory(root):
    path = Path(root)/f'{time.strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:8]}'
    path.mkdir(parents=True, exist_ok=False)
    return path.resolve()


def tail_log(path, limit=1800):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


class Job:
    def __init__(self, python, request):
        self.directory = Path(request['output_dir'])
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request = request
        self.log_path = self.directory/'job.log'
        self.cancel_started = None
        request_path = self.directory/'request.json'
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        self.log = self.log_path.open('wb', buffering=0)
        try:
            env = clean_environment()
            if request.get('ffmpeg_path'):
                env['PATH'] = request['ffmpeg_path'] + os.pathsep + env.get('PATH', '')
            self.process = subprocess.Popen([str(python), '-u', str(Path(__file__).with_name('worker.py')),
                                            '--request', str(request_path)],
                cwd=request['project_root'], env=env, stdin=subprocess.DEVNULL,
                stdout=self.log, stderr=subprocess.STDOUT, shell=False,
                **({'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == 'nt' else {'start_new_session': True}))
        except Exception:
            self.log.close()
            raise

    def read_status(self):
        try:
            return json.loads((self.directory/'status.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            return dict(state='running', message='Starting external Python…')

    def stop(self, force=False):
        if self.process.poll() is not None:
            return
        if self.cancel_started is None:
            self.cancel_started = time.monotonic()
        try:
            if os.name == 'nt':
                # taskkill /T is scoped to the plugin-owned child and its tree.
                subprocess.run(['taskkill', '/PID', str(self.process.pid), '/T', '/F'],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.killpg(self.process.pid, signal.SIGKILL if force else signal.SIGTERM)
        except ProcessLookupError:
            pass

    def close(self):
        self.log.close()


class QueuedJob:
    """Blender-side handle; owns no worker lifetime and never imports Torch."""
    def __init__(self, python=None, request=None, identity=None, database=None):
        from .task_queue import Queue
        self.queue = Queue(database)
        self.identity = identity or self.queue.enqueue(python,request)
        self.process = self
        self.cancel_started = None
        if identity is None:
            from .queue_runner import start
            try:
                start(python,self.queue.path)
            except OSError as error:
                self.queue.update(self.identity,state='failed',message=f'Cannot start External Python: {error}')
                raise

    @property
    def directory(self):
        row = self.queue.get(self.identity)
        return Path(row['directory'])/f"attempt-{row['attempt']:03d}"

    @property
    def log_path(self):
        return self.directory/'job.log'

    def read_status(self):
        return self.queue.status(self.queue.get(self.identity))

    def poll(self):
        state = self.queue.get(self.identity)['state']
        return None if state in {'queued','running','retrying'} else 0 if state == 'completed' else 1

    def stop(self,force=False):
        self.queue.cancel(self.identity)
        if self.cancel_started is None:
            self.cancel_started = time.monotonic()

    def close(self):
        pass
