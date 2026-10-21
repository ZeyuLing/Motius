"""Prepare or publish the verified FlowHMR demo to its Hugging Face model cards.

Only README and media files are committed. Each card uses its own checkpoint
prediction, with the same input, frontend cache, inference settings and camera.
Authentication uses the standard Hugging Face credential cache/environment.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS', '1')
from huggingface_hub import CommitOperationAdd, HfApi, get_token, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
LATEST = 'ZeyuLing/Motius-FlowHMR-Latest'
BASE = 'ZeyuLing/Motius-FlowHMR-Base'
MEDIA = ROOT / 'assets/model_zoo/flowhmr'
BASE_PREVIEW = 'https://github.com/user-attachments/assets/791ec4c4-638d-478c-b79d-7ddb09d3582b'
LATEST_PREVIEW = 'https://github.com/user-attachments/assets/ca588fd7-52c8-49a8-af11-e0def808a715'


def media_files(variant):
    stem = 'demo_' + variant.lower() + '_mesh'
    return ('demo_input.mp4', stem+'.mp4', stem+'_preview.png', stem+'.json')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def visual_section(variant, metadata, official_sha):
    repo = BASE if variant == 'Base' else LATEST
    other = LATEST if variant == 'Base' else BASE
    media_url = f'https://huggingface.co/{repo}/resolve/main/media/'
    stem = 'demo_' + variant.lower() + '_mesh'
    video = media_url + stem + '.mp4'
    poster = media_url + stem + '_preview.png'
    description = 'Base flow-matching release' if variant == 'Base' else 'Latest release after PHC+ GRPO post-training'
    return f'''## Visual Results

### Supplied-video demo — {variant}

<video src="{video}" poster="{poster}" controls playsinline preload="metadata" width="100%"></video>

[Input video]({media_url}demo_input.mp4) · [Download demo]({video}) ·
[Render provenance]({media_url}{stem}.json) ·
[Other checkpoint on the same video](https://huggingface.co/{other}#visual-results)

**Checkpoint:** [Motius-FlowHMR-{variant}](https://huggingface.co/{repo}),
the official {description}. This video is an actual {variant} prediction.

- Official checkpoint SHA256: `{official_sha}`.
- Motius inference safetensors SHA256: `{metadata['checkpoint_sha256']}`.
- 750 frames, 25 seconds, 30 fps; seed 0, 20 Euler steps, CFG 1.
- VGGT camera sampling every 100 frames (9 keyframes interpolated to 750).

Base and Latest use the exact same cached tracking boxes, SAM features and camera
estimates (shared frontend SHA256: `{metadata['inference']['shared_frontend_sha256']}`).
Both use the same seed and sampling settings. A single camera framing fits both
predicted sequences, so the display scale and view are identical.

The left panel displays the supplied video with the Motius frontend's tracking
bbox. The right panel reconstructs SMPL-H vertices from the saved predicted
rotations, shape and translation, using Motius's bundled implementation.
Three.js provides smooth normals, depth-tested surfaces, lighting and floor
shadows. The fixed camera fits the complete sequence; horizontal root motion
is centered for display, while vertical motion is preserved. This is a world
view, not a mesh reprojection into the source camera. Original audio is retained.
No external FlowHMR checkout is needed for inference or mesh reconstruction.

Earlier parity-validation clips: [Base, 30 frames]({BASE_PREVIEW}) ·
[Latest, 30 frames]({LATEST_PREVIEW}). These use a different input clip.

'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Commit the prepared cards and media')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/inference/flowhmr/hf_publish')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_by_variant = {}
    validation = json.loads((ROOT/'docs/model_zoo/flowhmr_validation.json').read_text())
    api = HfApi()
    info_by_variant = {}
    for variant, repo in [('Base', BASE), ('Latest', LATEST)]:
        metadata = json.loads((MEDIA/f'demo_{variant.lower()}_mesh.json').read_text())
        if metadata['frames'] != 750 or metadata['fps'] != 30:
            raise ValueError(f'{variant}: expected 750 frames at 30 fps')
        if metadata['video_sha256'] != sha256(MEDIA/f'demo_{variant.lower()}_mesh.mp4'):
            raise ValueError(f'{variant}: video differs from render provenance')
        if metadata['source_sha256'] != sha256(MEDIA/'demo_input.mp4'):
            raise ValueError(f'{variant}: input differs from render provenance')
        if metadata['renderer'] != 'motius-threejs-smooth-mesh':
            raise ValueError(f'{variant}: expected the Three.js render')
        info = api.model_info(repo, files_metadata=True)
        weights = next(item for item in info.siblings if item.rfilename == 'model.safetensors')
        if weights.lfs.sha256 != metadata['checkpoint_sha256']:
            raise ValueError(f'{variant}: inference weights do not match published artifact')
        metadata_by_variant[variant] = metadata
        info_by_variant[variant] = info
    base, latest = metadata_by_variant['Base'], metadata_by_variant['Latest']
    if base['checkpoint_sha256'] == latest['checkpoint_sha256']:
        raise ValueError('Base and Latest must use different checkpoint weights')
    for key in ('shared_frontend_sha256', 'source_video_sha256', 'seed', 'euler_steps', 'cfg_scale', 'vggt_frame_interval'):
        if base['inference'][key] != latest['inference'][key]:
            raise ValueError(f'Comparison settings differ: {key}')
    if base['camera'] != latest['camera']:
        raise ValueError('Base and Latest must use identical camera framing')
    plans = []
    for variant, repo in [('Latest',LATEST), ('Base',BASE)]:
        info = info_by_variant[variant]
        metadata = metadata_by_variant[variant]
        official_sha = validation['full_network'][variant.lower()]['checkpoint_sha256']
        original = Path(hf_hub_download(repo,'README.md',revision=info.sha)).read_text()
        card, replacements = re.subn(r'## Visual Results\n.*?(?=\n## Model Overview)',
                                      lambda _: visual_section(variant,metadata,official_sha),
                                      original, count=1, flags=re.S)
        if replacements != 1:
            raise ValueError(f'{repo} has no recognized Visual Results section')
        card = re.sub(r'<h1 align="center">FlowHMR(?: Base| Latest)? Model Card</h1>',
                      f'<h1 align="center">FlowHMR {variant} Model Card</h1>',card,count=1)
        destination = args.output_dir / variant.lower() / 'README.md'
        destination.parent.mkdir(parents=True,exist_ok=True)
        destination.write_text(card)
        operations = [CommitOperationAdd(path_in_repo='README.md',path_or_fileobj=str(destination))]
        files = media_files(variant)
        operations += [CommitOperationAdd(path_in_repo='media/'+name,path_or_fileobj=str(MEDIA/name)) for name in files]
        plans.append((repo,info.sha,card,operations,files))
        print(f'Prepared {repo}: '+', '.join(op.path_in_repo for op in operations),flush=True)
    if not args.apply:
        print('Preparation only. Use --apply to publish.',flush=True)
        return
    if not get_token():
        raise RuntimeError('No Hugging Face login; run hf auth login before --apply')
    account = api.whoami().get('name')
    print(f'Publishing with authenticated account {account}',flush=True)
    receipts = []
    for repo,parent,card,operations,files in plans:
        commit = api.create_commit(repo_id=repo,repo_type='model',parent_commit=parent,
                                  commit_message='Update supplied-video FlowHMR mesh and tracking demo',operations=operations)
        published = Path(hf_hub_download(repo,'README.md',revision=commit.oid)).read_text()
        if published != card:
            raise RuntimeError(f'{repo}: published card differs from prepared card')
        for name in files:
            downloaded = hf_hub_download(repo,'media/'+name,revision=commit.oid)
            if sha256(downloaded) != sha256(MEDIA/name):
                raise RuntimeError(f'{repo}: published {name} differs from the local artifact')
        receipts.append(dict(repo_id=repo,commit=commit.oid,url=commit.commit_url,
                             changed_files=[op.path_in_repo for op in operations]))
        print(f'Published and verified {repo}: {commit.oid}',flush=True)
    (args.output_dir/'publish_receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')


if __name__ == '__main__':
    main()
