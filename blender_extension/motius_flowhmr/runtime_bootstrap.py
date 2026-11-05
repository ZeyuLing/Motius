# SPDX-License-Identifier: GPL-3.0-or-later
"""Install an isolated NVIDIA runtime without modifying Blender Python."""
import argparse
import csv
try:
    import fcntl
except ImportError:
    fcntl = None
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import uuid
import zipfile

UV_URL = 'https://files.pythonhosted.org/packages/73/da/2597becbc0fcbb59608d38fda5db79969e76dedf5b072f0e8564c8f0628b/uv-0.9.11-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl'
UV_SHA256 = 'e97906ca1b90dac91c23af20e282e2e37c8eb80c3721898733928a295f2defda'
UV_BINARY_SHA256 = '534bcd0cf0a0d00a0977363abfc93cd1300914d03e16a77d8d5133a3fd99dacb'
WINDOWS_UV_URL = 'https://files.pythonhosted.org/packages/2c/2f/d51c02627de68a7ca5b82f0a5d61d753beee3fe696366d1a1c5d5e40cd58/uv-0.9.11-py3-none-win_amd64.whl'
WINDOWS_UV_SHA256 = 'c65a024ad98547e32168f3a52360fe73ff39cd609a8fb9dd2509aac91483cfc8'
WINDOWS_UV_BINARY_SHA256 = 'e3f1abf6b4ee854b7514d12bc450a1bd64d71ad9ce9518744199079f227795b3'
REVISION = 'e77d64daaf23a030d6908c843b2f4e36888097b3'


class InstallationInProgress(RuntimeError):
    pass


def discover_nvidia_gpus():
    """Detect physical GPUs before installing Torch; do not infer from model names."""
    candidates = [shutil.which('nvidia-smi')]
    if platform.system() == 'Windows':
        for variable, suffix in [('SystemRoot', 'System32/nvidia-smi.exe'),
                                 ('ProgramFiles', 'NVIDIA Corporation/NVSMI/nvidia-smi.exe')]:
            root = os.environ.get(variable)
            if root:
                candidate = Path(root)/suffix
                if candidate.is_file():
                    candidates.append(str(candidate))
    executable = next((p for p in candidates if p), None)
    if not executable:
        raise RuntimeError('NVIDIA GPU detection failed: nvidia-smi was not found. Install an NVIDIA driver and retry.')
    try:
        result = subprocess.run([executable, '--query-gpu=index,uuid,name,compute_cap,driver_version',
                                 '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=20, check=True)
        devices = []
        for row in csv.reader(result.stdout.splitlines(), skipinitialspace=True):
            if len(row) != 5:
                raise ValueError('unexpected nvidia-smi output')
            major, minor = (int(part) for part in row[3].strip().split('.'))
            devices.append(dict(index=int(row[0]), uuid=row[1].strip(), name=row[2].strip(),
                                capability=(major, minor), driver=row[4].strip()))
        if not devices:
            raise ValueError('no NVIDIA GPU detected')
        return devices
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        raise RuntimeError('Could not identify NVIDIA GPU architecture. Check the driver and nvidia-smi, then retry.') from error


def pytorch_profile(devices):
    # A mixed-GPU machine must have kernels for every visible physical GPU.
    # Modern CUDA builds may drop old architectures; reject an unverified mix.
    capabilities = [device['capability'] for device in devices]
    if any(major >= 10 for major, _ in capabilities):
        if any(capability < (7, 5) for capability in capabilities):
            raise RuntimeError('This mixed GPU system includes Blackwell and a pre-Turing GPU; use an external environment for the intended device.')
        return dict(cuda='cu128', torch='2.7.1+cu128', torchvision='0.22.1+cu128')
    return dict(cuda='cu124', torch='2.6.0+cu124', torchvision='0.21.0+cu124')


def write_status(path, **data):
    data.setdefault('installer_pid', os.getpid())
    try:
        from queue_runner import process_birth
        data.setdefault('installer_start', process_birth(os.getpid()))
    except (ImportError, OSError, IndexError):
        pass
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def download_uv(home, status):
    tools = home/'tools'; tools.mkdir(parents=True, exist_ok=True)
    windows = platform.system() == 'Windows'
    url = WINDOWS_UV_URL if windows else UV_URL
    archive_sha = WINDOWS_UV_SHA256 if windows else UV_SHA256
    binary_sha = WINDOWS_UV_BINARY_SHA256 if windows else UV_BINARY_SHA256
    executable = 'uv.exe' if windows else 'uv'
    uv = tools/('uv-0.9.11.exe' if windows else 'uv-0.9.11')
    if uv.is_file() and os.access(uv, os.X_OK):
        with uv.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() == binary_sha:
                return uv
    temp = tools/('uv-' + uuid.uuid4().hex + '.whl')
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temp.open('wb') as output:
            total = int(response.headers.get('Content-Length') or 0)
            done = 0; last = 0.
            while True:
                data = response.read(1024*1024)
                if not data: break
                output.write(data); digest.update(data); done += len(data)
                if time.monotonic()-last > .5:
                    write_status(status, state='running', phase='bootstrap', message='Downloading installer', current=done, total=total, unit='bytes')
                    last = time.monotonic()
        if digest.hexdigest() != archive_sha:
            raise RuntimeError('Installer checksum mismatch')
        with zipfile.ZipFile(temp) as wheel:
            members = [name for name in wheel.namelist() if '.data/scripts/' in name and name.endswith('/' + executable)]
            if len(members) != 1: raise RuntimeError('Unexpected installer package layout')
            binary = tools/('uv-' + uuid.uuid4().hex + '.tmp')
            payload = wheel.read(members[0])
            if hashlib.sha256(payload).hexdigest() != binary_sha:
                raise RuntimeError('Installer executable checksum mismatch')
            binary.write_bytes(payload); binary.chmod(0o755); os.replace(binary, uv)
        return uv
    finally:
        temp.unlink(missing_ok=True)


def extract_backend(archive, home):
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    apps = home/'apps'; apps.mkdir(parents=True, exist_ok=True)
    target = apps/digest
    if (target/'motius/__init__.py').is_file() and (target/'blender_extension/motius_flowhmr/worker.py').is_file():
        return target
    staging = apps/('install-' + uuid.uuid4().hex); staging.mkdir()
    try:
        with tarfile.open(archive, 'r:gz') as source:
            for item in source:
                path = Path(item.name)
                if path.is_absolute() or '..' in path.parts or not item.isfile():
                    raise RuntimeError('Invalid bundled backend archive')
                output = staging/path; output.parent.mkdir(parents=True, exist_ok=True)
                with source.extractfile(item) as inp, output.open('wb') as out:
                    shutil.copyfileobj(inp, out)
        if not (staging/'motius/__init__.py').is_file(): raise RuntimeError('Backend archive is incomplete')
        try: staging.rename(target)
        except FileExistsError: shutil.rmtree(staging)
        return target
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def run_command(command, env, status, phase, message):
    write_status(status, state='running', phase=phase, message=message, current=0, total=0, unit='indeterminate')
    child = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in child.stdout:
        line = line.strip()
        if line:
            print(line, flush=True)
            if line.startswith(('Downloading ', 'Downloaded ', 'Preparing ', 'Installing ', 'Installed ', 'Resolved ', 'Prepared ')):
                write_status(status, state='running', phase=phase, message=message, detail=line[:180], current=0, total=0, unit='indeterminate')
    if child.wait() != 0:
        raise RuntimeError(f'{message} failed (exit {child.returncode}); see setup.log')


def main(home, bundle, requirements, body):
    if platform.system() not in ('Linux', 'Windows') or platform.machine().lower() not in ('x86_64', 'amd64'):
        raise RuntimeError('Automatic installation requires 64-bit Linux or Windows with an NVIDIA GPU.')
    home.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(home).free < 8*1024**3:
        raise RuntimeError('At least 8 GiB of free disk space is required to install Python and CUDA dependencies')
    status = home/'install-status.json'
    lock = (home/'install.lock').open('a+')
    try:
        if os.name == 'nt':
            import msvcrt
            lock.seek(0); lock.write('0'); lock.flush(); lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise InstallationInProgress('Another FlowHMR runtime installation is already running')
    write_status(status, state='running', phase='bootstrap', message='Preparing installer', current=0, total=0, unit='indeterminate')
    devices = discover_nvidia_gpus()
    profile = pytorch_profile(devices)
    print('Detected NVIDIA GPUs:', ', '.join(f"{d['name']} (compute {d['capability'][0]}.{d['capability'][1]}, driver {d['driver']})" for d in devices), flush=True)
    print(f"Selected PyTorch {profile['torch']} / CUDA {profile['cuda']}", flush=True)
    uv = download_uv(home, status)
    write_status(status, state='running', phase='backend', message='Preparing bundled Motius backend', current=0, total=0, unit='indeterminate')
    project = extract_backend(bundle, home)
    env = os.environ.copy()
    env.update(UV_CACHE_DIR=str(home/'cache'), UV_PYTHON_INSTALL_DIR=str(home/'python'), PYTHONUNBUFFERED='1',
               MOTIUS_INSTALL_PID=str(os.getpid()))
    try:
        from queue_runner import process_birth
        env['MOTIUS_INSTALL_START'] = process_birth(os.getpid()) or ''
    except (ImportError, OSError, IndexError):
        pass
    for name in ('PYTHONHOME', 'PYTHONPATH', 'PYTHONUSERBASE', 'PYTHONEXECUTABLE', '__PYVENV_LAUNCHER__'):
        env.pop(name, None)
    venv = home/'envs'/('runtime-' + uuid.uuid4().hex[:12]); venv.parent.mkdir(parents=True, exist_ok=True)
    run_command([str(uv), 'venv', '--python', '3.12.12', '--python-preference', 'only-managed', str(venv)], env, status, 'python', 'Installing Python 3.12.12')
    python = venv/('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    run_command([str(uv), 'pip', 'install', '--python', str(python), '--index-url', f"https://download.pytorch.org/whl/{profile['cuda']}",
                 '--extra-index-url', 'https://pypi.org/simple', '--index-strategy', 'unsafe-best-match',
                 f"torch=={profile['torch']}", f"torchvision=={profile['torchvision']}", 'numpy==1.26.4', 'scipy==1.14.1'],
                env, status, 'pytorch', f"Installing PyTorch ({profile['cuda']})")
    run_command([str(uv), 'pip', 'install', '--python', str(python), '--no-deps', '-r', str(requirements)],
                env, status, 'dependencies', 'Installing FlowHMR dependencies')
    weights = home/'weights'/REVISION
    helper = project/'blender_extension/motius_flowhmr/runtime_helper.py'
    write_status(status, state='running', phase='check', message='Checking GPU and CUDA before downloading weights', current=0, total=0, unit='indeterminate')
    probe = subprocess.run([str(python), '-u', str(helper), 'probe', '--target', str(weights), '--status', str(status)], env=env, text=True, capture_output=True)
    if probe.returncode:
        print(probe.stdout, probe.stderr, file=sys.stderr, flush=True)
        detail = (probe.stderr.strip().splitlines() or ['Unknown CUDA failure'])[-1]
        raise RuntimeError(f'GPU compatibility check failed: {detail}; see setup.log')
    write_status(status, state='running', phase='weights', message='Preparing model download', current=0, total=0, unit='indeterminate')
    env['HF_HOME'] = str(home/'hf-cache')
    subprocess.run([str(python), '-u', str(helper), 'download', '--target', str(weights), '--status', str(status)], env=env, check=True)
    write_status(status, state='running', phase='check', message='Checking Python, GPU, operators and model files', current=0, total=0, unit='indeterminate')
    result = subprocess.run([str(python), '-u', str(helper), 'check', '--target', str(weights), '--status', str(status), '--project', str(project), '--body', body], env=env, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout, result.stderr, file=sys.stderr, flush=True)
        raise RuntimeError('Environment check failed; see setup.log')
    info = json.loads(result.stdout.strip().splitlines()[-1])
    ready = dict(state='completed', phase='done', message='Runtime installed' if info['body_ready'] else 'Runtime installed · choose your neutral SMPL-H model',
                 python=str(python), project=str(project), checkpoint=str(weights), cache_dir=str(home/'hf-cache/hub'),
                 ffmpeg_path=info['ffmpeg_path'], info=info)
    tmp = home/'runtime.json.tmp'; tmp.write_text(json.dumps(ready, indent=2)+'\n', encoding='utf-8'); os.replace(tmp, home/'runtime.json')
    write_status(status, **ready)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--home', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--requirements', required=True)
    parser.add_argument('--body', default='')
    args = parser.parse_args()
    path = Path(args.home).expanduser().absolute(); path.mkdir(parents=True, exist_ok=True)
    try:
        main(path, Path(args.bundle), Path(args.requirements), args.body)
    except InstallationInProgress as error:
        print(str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
    except Exception as error:
        print(f'Runtime installation failed: {error}', file=sys.stderr, flush=True)
        write_status(path/'install-status.json', state='failed', phase='failed', message=str(error))
        raise SystemExit(1)
