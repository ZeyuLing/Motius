#!/usr/bin/env python3
"""Freeze the dependencies from a validated NVIDIA 25.01 Python environment.

This copies installed distribution files, not user caches, model weights, pip
configuration, credentials, editable installs or arbitrary site directories.
"""
import argparse
import gzip
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import tarfile

from packaging.requirements import Requirement

ROOTS = ('accelerate', 'addict', 'fvcore', 'huggingface-hub', 'hydra-core',
         'mmcv-lite', 'mmengine', 'opencv-python-headless', 'peft',
         'supervision', 'tokenizers', 'transformers', 'yapf', 'roma', 'timm',
         'trimesh', 'smplx', 'yacs', 'omegaconf', 'lap', 'einops',
         'torchdiffeq', 'webdataset', 'flow-matching')
# These libraries and their native CUDA dependencies come from the pinned
# NVIDIA base image, which is checked against the recorded versions at build.
BASE = {'torch', 'torchvision', 'numpy', 'scipy', 'pip', 'setuptools', 'wheel'}


def normalize(name):
    return name.lower().replace('_', '-').replace('.', '-')


def collect_distributions():
    pending = list(ROOTS)
    seen = set()
    found = {}
    missing = []
    while pending:
        name = pending.pop()
        key = normalize(name)
        if key == 'opencv-python':
            key = 'opencv-python-headless'
            name = 'opencv-python-headless'
        if key in seen or key.startswith('nvidia-'):
            continue
        seen.add(key)
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(name)
            continue
        found[key] = dist
        if key in BASE:
            continue
        for spec in dist.requires or ():
            try:
                req = Requirement(spec)
                if req.marker is None or req.marker.evaluate({'extra': ''}):
                    pending.append(req.name)
            except ValueError:
                raise RuntimeError(f'Invalid requirement in {name}: {spec}')
    return found, missing


def export(output):
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    found, missing = collect_distributions()
    required = {'accelerate', 'mmengine', 'mmcv-lite', 'opencv-python-headless',
                'supervision', 'transformers', 'roma', 'torch', 'torchvision', 'numpy', 'scipy'}
    absent = required - found.keys()
    if absent:
        raise RuntimeError(f'The validated runtime is missing required distributions: {sorted(absent)}')
    included = {}
    packages = {}
    base_versions = {}
    with output.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed, tarfile.open(fileobj=compressed, mode='w') as archive:
        for name, dist in sorted(found.items()):
            packages[name] = dist.version
            if name in BASE:
                if name in {"torch", "torchvision", "numpy", "scipy"}:
                    base_versions[name] = dist.version
                continue
            root = Path(dist.locate_file('')).resolve()
            for entry in sorted(dist.files or (), key=str):
                relative = Path(str(entry))
                if relative.is_absolute() or '..' in relative.parts or '__pycache__' in relative.parts:
                    continue
                if relative.suffix in {'.pyc', '.pyo'} or relative.name == 'direct_url.json':
                    continue
                path = root / relative
                if not path.is_file() or not path.resolve().is_relative_to(root):
                    continue
                key = relative.as_posix()
                if key in included:
                    # Identical shared files are fine; incompatible ownership is not.
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    if included[key] != digest:
                        raise RuntimeError(f'Conflicting distribution file: {key}')
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                included[key] = digest
                info = tarfile.TarInfo(key)
                info.size = path.stat().st_size
                info.mode = 0o755 if os.access(path, os.X_OK) else 0o644
                info.mtime = 0; info.uid = info.gid = 0
                with path.open('rb') as file:
                    archive.addfile(info, file)
    manifest = dict(format='motius.flowhmr.runtime-overlay.v1', python=platform.python_version(),
                    architecture=platform.machine(), base_image='nvcr.io/nvidia/pytorch:25.01-py3',
                    base_versions=base_versions, distributions=packages, missing_optional=sorted(missing),
                    files=len(included), overlay_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    if platform.system() != 'Linux' or manifest['architecture'] != 'x86_64' or not manifest['python'].startswith('3.12.'):
        raise RuntimeError('The NVIDIA 25.01 runtime profile requires Linux x86_64 and Python 3.12')
    output.with_suffix(output.suffix+'.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'archive':str(output),'bytes':output.stat().st_size,'sha256':manifest['overlay_sha256'],
                      'distributions':len(packages),'missing_optional':sorted(missing)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    export(parser.parse_args().output)
