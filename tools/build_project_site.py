#!/usr/bin/env python3
"""Stage an allowlisted static GitHub Pages site; never publish the repo tree."""
import argparse
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
MEDIA = {
    'logo.png': 'assets/brand/motius-logo-readme.png',
    'generation.mp4': 'site/media/generation.mp4',
    'dance.mp4': 'assets/model_zoo/bailando/bailando_aistpp_waacking_gWA_mWA0_smpl_mesh.mp4',
    'autorig.mp4': 'assets/motion/auto_rigging_demo/motius_multi_character_autorig_004822_960x540_30fps.mp4',
    'characters.mp4': 'assets/motion/fbx_character_demo/004822_skeleton_smpl_mixamo_1440_30fps.mp4',
    'representations.mp4': 'assets/motion/representation_demo/004822_hml_smpl_soma_core_g1_1920_30fps.mp4',
}


def build(destination):
    destination = Path(destination).resolve()
    # Only an empty staging directory is accepted, avoiding stale or accidental
    # publication of other files. Never recursively delete a caller's directory.
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Staging directory must be empty: {destination}')
    required = [ROOT / p for p in MEDIA.values()]
    required += [ROOT / 'site' / p for p in ('index.html', 'style.css', 'app.js')]
    posters = [ROOT / 'site/media' / f'{Path(name).stem}.jpg' for name in MEDIA if name.endswith('.mp4')]
    required += posters
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    (destination / 'media').mkdir(parents=True, exist_ok=True)
    for name in ('index.html', 'style.css', 'app.js'):
        shutil.copy2(ROOT / 'site' / name, destination / name)
    for name, source in MEDIA.items():
        shutil.copy2(ROOT / source, destination / 'media' / name)
    for path in posters:
        shutil.copy2(path, destination / 'media' / path.name)
    (destination / '.nojekyll').touch()
    print(f'Staged {sum(p.is_file() for p in destination.rglob("*"))} files at {destination}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    build(parser.parse_args().output)
