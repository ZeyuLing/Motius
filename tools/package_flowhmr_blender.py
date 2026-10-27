"""Build one Blender 4.2+ extension ZIP with both inference modes."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import tomllib
import zipfile


def backend_archive(root):
    """Bundle the project snapshot, excluding ignored assets, weights and secrets."""
    allowed = {'.py', '.json', '.yaml', '.yml', '.toml', '.txt', '.md', '.cfg', '.ini'}
    listing = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
        '--exclude-standard', '--', 'motius', 'blender_extension/motius_flowhmr'], cwd=root)
    paths = set()
    for part in listing.split(b'\0'):
        if not part:
            continue
        path = root / part.decode()
        if path.is_file() and not path.is_symlink() and '__pycache__' not in path.parts and (
                path.suffix.lower() in allowed or path.name in {'LICENSE', 'NOTICE', 'UPSTREAM_LICENSE'}):
            paths.add(path)
    paths.update(root / name for name in ('pyproject.toml', 'README.md'))
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode='wb', compresslevel=6, mtime=0) as gz, tarfile.open(fileobj=gz, mode='w') as archive:
        for path in sorted(paths):
            info = tarfile.TarInfo(path.relative_to(root).as_posix())
            content = path.read_bytes()
            info.size, info.mtime, info.mode = len(content), 0, 0o644
            archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def build(output_dir):
    root = Path(__file__).resolve().parents[1]
    source = root/'blender_extension/motius_flowhmr'
    manifest = tomllib.loads((source/'blender_manifest.toml').read_text())
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [path for path in sorted(source.rglob('*')) if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc']
    backend = backend_archive(root)
    reports = []
    for mode in ('STANDARD',):
        output = output_dir/f'motius-flowhmr-{manifest["version"]}.zip'
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                relative = str(path.relative_to(source)).replace('\\', '/')
                content = path.read_bytes()
                if relative == 'defaults.py':
                    content = f"# SPDX-License-Identifier: GPL-3.0-or-later\nDEFAULT_MODE = {mode!r}\n".encode()
                info = zipfile.ZipInfo(relative, (2026, 10, 23, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                archive.writestr(info, content)
            info = zipfile.ZipInfo('backend.tar.gz', (2026, 10, 23, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, backend)
        reports.append(dict(file=output.name, default_mode=mode, bytes=output.stat().st_size,
                            sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
    (output_dir/'checksums.json').write_text(json.dumps(reports, indent=2)+'\n')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('dist/flowhmr_blender'))
    args = parser.parse_args()
    for item in build(args.output_dir):
        print(json.dumps(item))
