# SPDX-License-Identifier: GPL-3.0-or-later
"""Download model assets and check the isolated inference environment."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time

REPO = 'ZeyuLing/Motius-FlowHMR-Latest'
REVISION = 'e77d64daaf23a030d6908c843b2f4e36888097b3'
PATTERNS = ['model.safetensors', 'config.yml', 'model_index.json', 'motion_stats.json',
            'preprocessor/**', '*LICENSE*', 'ATTRIBUTIONS.md']
REQUIRED = ['model.safetensors', 'config.yml', 'motion_stats.json',
            'preprocessor/sam/model.ckpt', 'preprocessor/sam/model_config.yaml',
            'preprocessor/sam/assets/mhr_model.pt', 'preprocessor/vggt_omega_1b_512.pt',
            'preprocessor/yolox_l.pth']


def atomic_status(path, data):
    if os.environ.get('MOTIUS_INSTALL_PID'):
        data.setdefault('installer_pid', int(os.environ['MOTIUS_INSTALL_PID']))
        data.setdefault('installer_start', os.environ.get('MOTIUS_INSTALL_START', ''))
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def download(target, status):
    from huggingface_hub import HfApi, snapshot_download
    from fnmatch import fnmatch
    import shutil
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    info = HfApi().model_info(REPO, revision=REVISION, files_metadata=True)
    files = [item for item in info.siblings if any(fnmatch(item.rfilename, pattern) for pattern in PATTERNS)]
    sizes = {item.rfilename: item.size for item in files if item.size is not None}
    total = sum(sizes.values())
    already = sum(min(size, (target/name).stat().st_size) for name, size in sizes.items() if (target/name).is_file())
    required = max(0, total-already) + 1024**3
    if shutil.disk_usage(target).free < required:
        raise RuntimeError(f'Insufficient disk space for model weights: need about {required/1024**3:.1f} GiB free')
    stop = threading.Event()
    started = time.monotonic()
    lock = threading.Lock()
    transferred = 0
    # HF reports HTTP/Xet progress through this context for each file. Count
    # actual download callbacks rather than preallocated on-disk file sizes.
    from huggingface_hub import file_download
    original_progress = file_download._get_progress_bar_context
    @contextmanager
    def tracked_progress(*args, **kwargs):
        nonlocal transferred
        with lock:
            transferred += max(0, int(kwargs.get('initial', 0) or 0))
        with original_progress(*args, **kwargs) as bar:
            class ObservedBar:
                def update(self, amount=1):
                    nonlocal transferred
                    with lock:
                        transferred += max(0, int(amount))
                    return bar.update(amount)
                def __getattr__(self, name):
                    return getattr(bar, name)
            yield ObservedBar()
    file_download._get_progress_bar_context = tracked_progress
    def report():
        while not stop.is_set():
            with lock:
                received = transferred
            current = already + received
            elapsed = max(time.monotonic()-started, .001)
            atomic_status(status, dict(state='running', phase='weights', message='Downloading model weights',
                                       current=min(current, total) if total else current, total=total,
                                       unit='bytes', bytes_per_second=received/elapsed))
            stop.wait(1)
    thread = threading.Thread(target=report, daemon=True)
    thread.start()
    try:
        snapshot_download(REPO, revision=REVISION, local_dir=str(target), allow_patterns=PATTERNS, max_workers=4)
        missing = [name for name in REQUIRED if not (target/name).is_file()]
        if missing:
            raise RuntimeError('Downloaded model is incomplete: ' + ', '.join(missing))
    finally:
        stop.set()
        thread.join(timeout=3)
        file_download._get_progress_bar_context = original_progress


def activate_static_ffmpeg():
    import sys
    # The add-on has its own progress.py; do not shadow the third-party
    # progress package required by static_ffmpeg when run as a script.
    original_path = sys.path[:]
    helper_dir = Path(__file__).resolve().parent
    shadow = sys.modules.get('progress')
    is_shadow = bool(shadow and getattr(shadow, '__file__', None)
                     and Path(shadow.__file__).resolve().parent == helper_dir
                     and not hasattr(shadow, '__path__'))
    try:
        if is_shadow:
            del sys.modules['progress']
        sys.path[:] = [entry for entry in sys.path if Path(entry or '.').resolve() != helper_dir]
        import static_ffmpeg
        static_ffmpeg.add_paths()
    finally:
        sys.path[:] = original_path
        if is_shadow:
            sys.modules['progress'] = shadow


def probe_cuda(torch, torchvision):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable: check the NVIDIA driver and CUDA-enabled PyTorch installation.')
    if torch.cuda.device_count() < 1:
        raise RuntimeError('No CUDA devices are visible to PyTorch; check CUDA_VISIBLE_DEVICES and the NVIDIA driver.')
    for index in range(torch.cuda.device_count()):
        capability = torch.cuda.get_device_capability(index)
        supported = torch.cuda.get_arch_list()
        majors = {int(arch.split('_', 1)[1]) // 10 for arch in supported
                  if arch.startswith('sm_') and arch.split('_', 1)[1].isdigit()}
        if majors and capability[0] not in majors:
            raise RuntimeError(f"GPU {torch.cuda.get_device_name(index)} has compute capability {capability[0]}.{capability[1]}, unsupported by PyTorch {torch.__version__} (architectures: {', '.join(supported)}). Update the runtime or use a compatible external Python.")
        try:
            device = f'cuda:{index}'
            test = torch.ones((16, 16), device=device); test = test @ test
            torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]], device=device),
                                torch.tensor([1.], device=device), .5)
            torch.cuda.synchronize(index)
            del test
        except Exception as error:
            raise RuntimeError(f'GPU {index} ({torch.cuda.get_device_name(index)}) failed a CUDA and TorchVision kernel check. Check the NVIDIA driver and installed PyTorch build: {error}') from error
    return dict(torch=torch.__version__, cuda=torch.version.cuda,
                devices=[dict(name=torch.cuda.get_device_name(i), capability=list(torch.cuda.get_device_capability(i)))
                         for i in range(torch.cuda.device_count())])


def check(project, weights, body):
    import sys
    sys.path.insert(0, str(project))
    import torch
    import torchvision
    import cv2
    import shutil
    from motius.pipelines.flowhmr import FlowHMRPipeline  # noqa: F401
    from motius.models.vision.sam3d_body.bundle import SAM3DBodyBundle  # noqa: F401
    from motius.models.vision.vggt_omega.bundle import VGGTOmegaBundle  # noqa: F401
    # Check a compiled TorchVision operator without loading model weights.
    torchvision.ops.nms(torch.tensor([[0., 0., 1., 1.]]), torch.tensor([1.]), .5)
    probe_cuda(torch, torchvision)
    downloaded_tools = not shutil.which('ffmpeg') or not shutil.which('ffprobe')
    if downloaded_tools:
        activate_static_ffmpeg()
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    if not ffmpeg or not ffprobe:
        raise RuntimeError('FFmpeg and FFprobe could not be installed or found.')
    if downloaded_tools:
        expected = ({'ffmpeg': '5af82a0d4fe2b9eae211b967332ea97edfc51c6b328ca35b827e73eac560dc0d',
                     'ffprobe': '192a1d6899059765ac8c39764fc3148d4e6049955956dc2029f81f4bd6a8972d'}
                    if os.name == 'nt' else
                    {'ffmpeg': 'ff45e18f5823f3036e419b22de5582ee5c956907910465e94e6d56df0aebb470',
                     'ffprobe': '9960ffd53be2a73bf2d33849ab792ba405e3701a996917c6f6b5d28bc9354296'})
        for name, path in (('ffmpeg', ffmpeg), ('ffprobe', ffprobe)):
            with open(path, 'rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            if digest != expected[name]:
                raise RuntimeError(f'Downloaded {name} checksum mismatch')
    missing = [name for name in REQUIRED if not (Path(weights)/name).is_file()]
    if missing:
        raise RuntimeError('Model files are incomplete: ' + ', '.join(missing))
    body_ready = bool(body and Path(body).is_file())
    if body_ready:
        from motius.motion.representation import flowhmr  # noqa: F401
        from motius.motion.body_models.flowhmr import SMPLSkeleton
        model = SMPLSkeleton(model_path=body)
        if model.j_template.shape != (52, 3) or model.j_shapedirs.shape[-1] != 16:
            raise RuntimeError('Choose the official neutral SMPL-H NPZ with 52 joints and 16 shape coefficients.')
    tool_path = os.pathsep.join(dict.fromkeys((str(Path(ffmpeg).parent), str(Path(ffprobe).parent))))
    return dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0), opencv=cv2.__version__, body_ready=body_ready,
                ffmpeg_path=tool_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['download', 'check', 'probe'])
    parser.add_argument('--target', required=True)
    parser.add_argument('--status', required=True)
    parser.add_argument('--project')
    parser.add_argument('--body', default='')
    args = parser.parse_args()
    if args.action == 'download':
        download(args.target, args.status)
    elif args.action == 'check':
        print(json.dumps(check(args.project, args.target, args.body)), flush=True)
    else:
        import torch
        import torchvision
        print(json.dumps(probe_cuda(torch, torchvision)), flush=True)
