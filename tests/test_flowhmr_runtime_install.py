"""Installer invariants independent of Blender, CUDA, and the network."""
import io
import hashlib
from pathlib import Path
import tarfile
import zipfile
from contextlib import contextmanager
from types import SimpleNamespace
import time
import sys
import types

import pytest
from blender_extension.motius_flowhmr import runtime_bootstrap as bootstrap
from blender_extension.motius_flowhmr import runtime_helper as helper


def make_tar(path, files):
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in files.items():
            data = content.encode(); info = tarfile.TarInfo(name); info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def test_backend_install_is_content_addressed_and_rejects_path_escape(tmp_path):
    source = tmp_path/'backend.tar.gz'
    make_tar(source, {'motius/__init__.py': '', 'blender_extension/motius_flowhmr/worker.py': 'pass'})
    target = bootstrap.extract_backend(source, tmp_path/'home')
    assert (target/'motius/__init__.py').is_file()
    assert bootstrap.extract_backend(source, tmp_path/'home') == target
    invalid = tmp_path/'bad.tar.gz'; make_tar(invalid, {'../escape.py': 'bad'})
    with pytest.raises(RuntimeError, match='Invalid bundled backend'):
        bootstrap.extract_backend(invalid, tmp_path/'home')
    assert not (tmp_path/'home/escape.py').exists()
    assert (target/'motius/__init__.py').is_file()


def test_windows_bootstrap_selects_and_checks_windows_executable(tmp_path, monkeypatch):
    executable = b'windows-uv-test'
    data = io.BytesIO()
    with zipfile.ZipFile(data, 'w') as archive:
        archive.writestr('uv-0.9.11.data/scripts/uv.exe', executable)
    wheel = data.getvalue()
    monkeypatch.setattr(bootstrap.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(bootstrap, 'WINDOWS_UV_SHA256', hashlib.sha256(wheel).hexdigest())
    monkeypatch.setattr(bootstrap, 'WINDOWS_UV_BINARY_SHA256', hashlib.sha256(executable).hexdigest())
    class Response(io.BytesIO):
        headers = {'Content-Length': str(len(wheel))}
    urls = []
    def fetch(url, timeout):
        urls.append(url)
        return Response(wheel)
    monkeypatch.setattr(bootstrap.urllib.request, 'urlopen', fetch)
    result = bootstrap.download_uv(tmp_path, tmp_path/'status.json')
    assert result.name == 'uv-0.9.11.exe' and result.read_bytes() == executable
    assert urls == [bootstrap.WINDOWS_UV_URL]
    assert bootstrap.download_uv(tmp_path, tmp_path/'status.json') == result
    assert len(urls) == 1


@pytest.mark.parametrize(('caps', 'cuda', 'version'), [
    ([(8, 6)], 'cu124', '2.6.0+cu124'),
    ([(8, 9)], 'cu124', '2.6.0+cu124'),
    ([(9, 0)], 'cu124', '2.6.0+cu124'),
    ([(12, 0)], 'cu128', '2.7.1+cu128'),
    ([(8, 9), (12, 0)], 'cu128', '2.7.1+cu128'),
])
def test_gpu_profile_matches_architecture(caps, cuda, version):
    profile = bootstrap.pytorch_profile([{'capability': cap} for cap in caps])
    assert profile['cuda'] == cuda and profile['torch'] == version


def test_gpu_profile_rejects_unverified_mixed_blackwell_and_pre_volta():
    with pytest.raises(RuntimeError, match='mixed GPU'):
        bootstrap.pytorch_profile([{'capability': (6, 1)}, {'capability': (12, 0)}])


def test_nvidia_smi_detects_multiple_windows_gpus(monkeypatch):
    monkeypatch.setattr(bootstrap.platform, 'system', lambda: 'Windows')
    monkeypatch.setattr(bootstrap.shutil, 'which', lambda name: 'nvidia-smi.exe')
    commands = []
    def run(command, **kwargs):
        commands.append(command)
        return SimpleNamespace(stdout='0, GPU-a, NVIDIA GeForce RTX 5090, 12.0, 576.02\n1, GPU-b, NVIDIA GeForce RTX 4090, 8.9, 576.02\n')
    monkeypatch.setattr(bootstrap.subprocess, 'run', run)
    devices = bootstrap.discover_nvidia_gpus()
    assert [d['capability'] for d in devices] == [(12, 0), (8, 9)]
    assert bootstrap.pytorch_profile(devices)['cuda'] == 'cu128'
    assert 'compute_cap' in commands[0][1]


def test_nvidia_smi_failure_does_not_silently_select_incompatible_build(monkeypatch):
    monkeypatch.setattr(bootstrap.shutil, 'which', lambda name: None)
    monkeypatch.setattr(bootstrap.platform, 'system', lambda: 'Linux')
    with pytest.raises(RuntimeError, match='nvidia-smi was not found'):
        bootstrap.discover_nvidia_gpus()


def test_cuda_probe_reports_missing_blackwell_kernels_before_download():
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_capability=lambda i: (12, 0),
                           get_arch_list=lambda: ['sm_80', 'sm_86', 'sm_90'],
                           get_device_name=lambda i: 'RTX 5090')
    torch = SimpleNamespace(cuda=cuda, __version__='2.6.0+cu124')
    with pytest.raises(RuntimeError, match='RTX 5090.*unsupported'):
        helper.probe_cuda(torch, SimpleNamespace())


def test_cuda_probe_executes_blackwell_matmul_and_torchvision_kernel():
    calls = []
    class Tensor:
        def __matmul__(self, other):
            calls.append('matmul')
            return self
    cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                           get_device_capability=lambda i: (12, 0),
                           get_arch_list=lambda: ['sm_80', 'sm_89', 'sm_90', 'sm_120'],
                           get_device_name=lambda i: 'RTX 5090',
                           synchronize=lambda i: calls.append(('synchronize', i)))
    torch = SimpleNamespace(cuda=cuda, __version__='2.7.1+cu128', version=SimpleNamespace(cuda='12.8'),
                            ones=lambda shape, device: Tensor(), tensor=lambda values, device: Tensor())
    vision = SimpleNamespace(ops=SimpleNamespace(nms=lambda boxes, scores, threshold: calls.append('nms')))
    result = helper.probe_cuda(torch, vision)
    assert result['devices'][0]['capability'] == [12, 0]
    assert calls == ['matmul', 'nms', ('synchronize', 0)]


def test_weight_manifest_includes_both_network_and_all_frontends():
    assert {'model.safetensors', 'preprocessor/sam/model.ckpt',
            'preprocessor/vggt_omega_1b_512.pt', 'preprocessor/yolox_l.pth'} <= set(helper.REQUIRED)
    assert helper.REVISION and not helper.REVISION.startswith('main')


def test_weight_progress_counts_download_callbacks(tmp_path, monkeypatch):
    import huggingface_hub
    from huggingface_hub import file_download
    status = tmp_path/'status.json'
    class Api:
        def model_info(self, *args, **kwargs):
            return SimpleNamespace(siblings=[SimpleNamespace(rfilename='model.safetensors', size=100)])
    @contextmanager
    def progress(*args, **kwargs):
        yield SimpleNamespace(update=lambda n: None)
    observed = []
    def download(*args, **kwargs):
        with file_download._get_progress_bar_context() as bar:
            bar.update(37)
        time.sleep(1.1)
        import json
        observed.append(json.loads(status.read_text()))
        for name in helper.REQUIRED:
            path = tmp_path/'weights'/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    monkeypatch.setattr(huggingface_hub, 'HfApi', Api)
    monkeypatch.setattr(huggingface_hub, 'snapshot_download', download)
    monkeypatch.setattr(file_download, '_get_progress_bar_context', progress)
    helper.download(tmp_path/'weights', status)
    assert observed[0]['current'] == 37 and observed[0]['total'] == 100
    assert file_download._get_progress_bar_context is progress


def test_ffmpeg_fallback_avoids_addon_progress_module_shadow(tmp_path, monkeypatch):
    site = tmp_path/'site'; package = site/'progress'; package.mkdir(parents=True)
    (package/'__init__.py').write_text('')
    (package/'bar.py').write_text('MARKER = 17\n')
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.syspath_prepend(str(Path(helper.__file__).parent))
    shadow = types.ModuleType('progress')
    shadow.__file__ = str(Path(helper.__file__).with_name('progress.py'))
    monkeypatch.setitem(sys.modules, 'progress', shadow)
    monkeypatch.delitem(sys.modules, 'progress.bar', raising=False)
    static = types.ModuleType('static_ffmpeg')
    seen = []
    def add_paths():
        import importlib
        seen.append(importlib.import_module('progress.bar').MARKER)
    static.add_paths = add_paths
    monkeypatch.setitem(sys.modules, 'static_ffmpeg', static)
    helper.activate_static_ffmpeg()
    assert seen == [17] and sys.modules['progress'] is shadow


def test_package_contains_bundled_backend_without_model_weights(tmp_path):
    from tools.package_flowhmr_blender import build
    result = build(tmp_path)
    assert len(result) == 1
    with zipfile.ZipFile(tmp_path/result[0]['file']) as package:
        assert {'backend.tar.gz', 'runtime_bootstrap.py', 'runtime_helper.py', 'runtime_requirements.txt'} <= set(package.namelist())
        with tarfile.open(fileobj=io.BytesIO(package.read('backend.tar.gz')), mode='r:gz') as backend:
            names = backend.getnames()
            assert 'motius/pipelines/flowhmr/pipeline.py' in names
            assert 'blender_extension/motius_flowhmr/worker.py' in names
            assert not any(name.endswith(('.ckpt', '.pth', '.pt', '.npz', '.safetensors', '.mp4')) for name in names)
