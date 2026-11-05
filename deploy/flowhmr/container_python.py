#!/usr/bin/env python3
"""External-Python adapter: keep the queue on host, isolate inference in Docker."""
import csv
import io
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import uuid


def mount(source, target, readonly):
    stream = io.StringIO()
    csv.writer(stream, lineterminator='').writerow([
        'type=bind', f'source={source}', f'target={target}', *(['readonly'] if readonly else [])])
    return stream.getvalue()


def existing_path(value, base=None):
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (Path(base) if base else Path.cwd()) / path
    path = Path(os.path.abspath(path))
    return path if path.exists() else None


def worker_command(argv, image=None, docker=None):
    """Return argv and exact job-owned container name; no shell interpolation."""
    arguments = list(argv)
    if arguments and arguments[0] == '-u':
        arguments.pop(0)
    if not arguments or Path(arguments[0]).name != 'worker.py' or '--request' not in arguments:
        raise ValueError('Expected worker.py --request <request.json>')
    request_file = Path(arguments[arguments.index('--request') + 1]).resolve(strict=True)
    request = json.loads(request_file.read_text(encoding='utf-8'))
    output = Path(request['output_dir']).resolve(strict=True)
    if request_file.parent != output:
        raise ValueError('Worker request must be inside its job output directory')
    old_project = Path(request['project_root']).resolve(strict=True)
    # Resolve default body assets on the host before changing the project root.
    if not request.get('body_model'):
        configured = os.environ.get('MOTIUS_SMPLH_MODEL')
        candidates = [Path(configured).expanduser()] if configured else []
        candidates += [old_project/'checkpoints/body_models/smplh/neutral/model.npz',
                       old_project/'outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz']
        body = next((p for p in candidates if p.is_file()), None)
        if body:
            request['body_model'] = str(body.absolute())
    request['project_root'] = '/opt/motius'
    request['output_dir'] = str(output)
    # The container reads a separate request; the original is kept for provenance.
    container_request = output/'container-request.json'
    temporary = output/'container-request.json.tmp'
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(container_request)
    binds = {str(output):(output, False)}
    for field in ('video', 'body_model', 'capture', 'subjects_file', 'preprocessing_cache', 'checkpoint', 'cache_dir'):
        value = request.get(field)
        path = existing_path(value, old_project)
        if path is None:
            if value and field not in ('checkpoint', 'cache_dir'):
                raise FileNotFoundError(f'{field} does not exist on the Docker host: {value}')
            if value and field == 'cache_dir':
                path = Path(value).expanduser().absolute()
                path.mkdir(parents=True, exist_ok=True)
            else:
                continue
        request[field] = str(path)
        # Cache and outputs are writable; all source assets are read-only.
        binds[str(path)] = (path.resolve(), field != 'cache_dir')
        if field == 'subjects_file':
            data = json.loads(path.read_text(encoding='utf-8'))
            used_video = existing_path(data.get('video'), path.parent)
            if used_video is None:
                raise FileNotFoundError('The detection manifest video is missing')
            binds[str(used_video)] = (used_video.resolve(), True)
    temporary.write_text(json.dumps(request, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(container_request)
    name = 'motius-flowhmr-' + uuid.uuid4().hex
    executable = docker or os.environ.get('MOTIUS_FLOWHMR_DOCKER', 'docker')
    image = image or os.environ.get('MOTIUS_FLOWHMR_IMAGE', 'motius-flowhmr:0.3.16')
    command = [executable, 'create', '--name', name, '--gpus', 'all', '--workdir', '/opt/motius',
               '--env', 'PYTHONUNBUFFERED=1', '--env', 'HOME=/tmp',
               '--env', 'HF_HOME=/tmp/.cache/huggingface']
    if hasattr(os, 'getuid'):
        command += ['--user', f'{os.getuid()}:{os.getgid()}']
    if request.get('offline'):
        command += ['--network', 'none']
    for key in ('HF_TOKEN', 'CUDA_VISIBLE_DEVICES'):
        if key in os.environ:
            command += ['--env', key]
    # A writable ancestor first, then narrower file/dir bindings.
    for target, (source, readonly) in sorted(binds.items(), key=lambda item: len(Path(item[0]).parts)):
        command += ['--mount', mount(str(source), target, readonly)]
    command += [image, 'python3', '-u', '/opt/motius/blender_extension/motius_flowhmr/worker.py',
                '--request', str(container_request)]
    return command, name


def run(argv):
    arguments = list(argv)
    stripped = arguments[1:] if arguments and arguments[0] == '-u' else arguments
    if stripped and Path(stripped[0]).name == 'queue_runner.py':
        # The queue owns a host SQLite DB and only starts the worker in Docker.
        os.execv(sys.executable, [sys.executable, *arguments])
    command, name = worker_command(arguments)
    docker = command[0]
    cancelled = False
    def cancel(_signal, _frame):
        nonlocal cancelled
        cancelled = True
    old_term = signal.signal(signal.SIGTERM, cancel)
    old_int = signal.signal(signal.SIGINT, cancel)
    child = None
    try:
        for stage in ('create', 'start'):
            args = command if stage == 'create' else [docker, 'start', '--attach', name]
            child = subprocess.Popen(args)
            while child.poll() is None:
                if cancelled:
                    child.terminate()
                    try: child.wait(timeout=0.5)
                    except subprocess.TimeoutExpired: child.kill()
                    return 130
                time.sleep(.1)
            if child.returncode:
                return child.returncode
        inspected = subprocess.run([docker, 'inspect', '--format', '{{.State.ExitCode}}', name],
                                   capture_output=True, text=True, timeout=10, check=True)
        return int(inspected.stdout.strip())
    finally:
        # Name is a fresh UUID scoped to this attempt; never affect other containers.
        try: subprocess.run([docker, 'rm', '--force', name], stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, timeout=2)
        except (OSError, subprocess.SubprocessError): pass
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGINT, old_int)


def main():
    try:
        if not shutil.which(os.environ.get('MOTIUS_FLOWHMR_DOCKER', 'docker')) and not any('queue_runner.py' in x for x in sys.argv[1:]):
            raise RuntimeError('Docker CLI not found; install Docker and NVIDIA Container Toolkit on the inference host')
        return run(sys.argv[1:])
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'FlowHMR container startup failed: {error}', file=sys.stderr, flush=True)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
