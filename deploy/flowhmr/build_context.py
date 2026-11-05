#!/usr/bin/env python3
"""Create a transferable Docker build context without checkpoints or credentials."""
import argparse
import gzip
import hashlib
import json
import os
import subprocess
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[2]
ALLOWED = {'.py', '.json', '.yaml', '.yml', '.toml', '.txt', '.md', '.cfg', '.ini'}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def build(overlay, destination):
    overlay=Path(overlay).resolve(strict=True)
    manifest_path=Path(str(overlay)+'.json').resolve(strict=True)
    manifest=json.loads(manifest_path.read_text())
    if digest(overlay)!=manifest['overlay_sha256']:
        raise ValueError('Runtime overlay does not match its recorded checksum')
    destination=Path(destination).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    files=[]
    # Respect repository ignores: do not traverse datasets, caches or weights.
    try:
        listing=subprocess.check_output(['git','ls-files','-z','--cached','--others','--exclude-standard','--',
            'motius','blender_extension/motius_flowhmr'],cwd=ROOT)
        candidates=(ROOT/part.decode() for part in listing.split(b'\0') if part)
    except (OSError,subprocess.CalledProcessError):
        candidates=(path for source in [ROOT/'motius',ROOT/'blender_extension/motius_flowhmr'] for path in source.rglob('*'))
    seen=set()
    for path in candidates:
        if not path.is_file() or path.is_symlink() or '__pycache__' in path.parts:
            continue
        if path.suffix.lower() in ALLOWED or path.name in {'LICENSE','NOTICE','UPSTREAM_LICENSE'}:
            name=path.relative_to(ROOT).as_posix()
            if name not in seen:files.append((path,name));seen.add(name)
    for name in ('Dockerfile','verify_runtime.py','container_python.py','export_overlay.py','build_context.py','README.md'):
        p=ROOT/'deploy/flowhmr'/name
        if p.exists():files.append((p,'Dockerfile' if name=='Dockerfile' else f'deploy/flowhmr/{name}'))
    for relative in ('tools/infer_flowhmr.py', 'pyproject.toml', 'README.md'):
        files.append((ROOT/relative,relative))
    files.extend([(overlay,'runtime-overlay.tar.gz'),(manifest_path,'runtime-overlay.tar.gz.json')])
    code_hash=hashlib.sha256()
    with destination.open('wb') as raw, gzip.GzipFile(fileobj=raw,mode='wb',compresslevel=1,mtime=0) as gz, tarfile.open(fileobj=gz,mode='w') as archive:
        for path,name in sorted(files,key=lambda item:item[1]):
            info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mtime=0;info.uid=info.gid=0
            info.mode=0o755 if os.access(path,os.X_OK) else 0o644
            with path.open('rb') as stream:
                if path in (overlay,manifest_path):
                    archive.addfile(info,stream)
                else:
                    class HashingReader:
                        def __init__(self, source):
                            self.source=source;self.hash=hashlib.sha256()
                        def read(self, count):
                            data=self.source.read(count);self.hash.update(data);return data
                    reader=HashingReader(stream)
                    archive.addfile(info,reader)
                    code_hash.update(name.encode()+b'\0'+reader.hash.digest())
    summary={'archive':str(destination),'bytes':destination.stat().st_size,'sha256':digest(destination),
             'overlay_sha256':manifest['overlay_sha256'],'source_sha256':code_hash.hexdigest(),
             'files':len(files),'base_image':manifest['base_image'],'weights_included':False}
    destination.with_suffix(destination.suffix+'.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.overlay,args.output)
