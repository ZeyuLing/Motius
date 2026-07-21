---
license: mit
library_name: motius
tags:
  - motion-generation
  - music-to-dance
  - aistplusplus
  - edge
---

<h1 align="center">EDGE Model Card</h1>

<p align="center">
  <strong>Editable diffusion-based dance generation from music.</strong>
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Tseng_EDGE_Editable_Dance_Generation_From_Music_CVPR_2023_paper.pdf">Paper</a> |
  <a href="https://edge-dance.github.io/">Project Page</a> |
  <a href="https://github.com/Stanford-TML/EDGE">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP">Motius Checkpoint</a> |
  <a href="https://zeyuling-edge-aistpp-preview.static.hf.space/">Interactive Preview</a>
</p>

EDGE is the CVPR 2023 work *EDGE: Editable Dance Generation From Music*.
Motius reproduces the released AIST++ network, cosine DDIM sampler, classifier-
free guidance schedule, long-sequence overlap, Jukebox conditioning contract,
motion decoder, and coordinate conversion without importing an upstream
checkout at runtime.

## Preview

[Open the audio-synchronized Three.js preview](https://zeyuling-edge-aistpp-preview.static.hf.space/).
The scene overlays EDGE's native SMPL-24 skeleton on the SMPL mesh decoded from
the same local rotations. It supports orbit, zoom, timeline seeking, audio
synchronization, NPZ export, and FBX export.

The viewer preserves the generated heading and XZ trajectory. It applies one
clip-wide vertical translation for display and never grounds individual
frames, so jumps, foot slide, and root-height drift remain visible.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Task | Music-to-Dance |
| Dataset | AIST++ |
| Music input | Jukebox layer 66, 4,800D at 30 fps |
| Native motion | EDGE-151 at 30 fps |
| Window | 150 frames with 75-frame overlap |
| Parameters | 49,464,471 |
| Checkpoint | [`ZeyuLing/Motius-EDGE-AISTPP`](https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP) |
| Pipeline | `motius.pipelines.edge.EDGEPipeline` |
| Upstream revision | `Stanford-TML/EDGE@17c3428669ed6733edd9d8c66f7dc62060b8e46d` |
| License | MIT |

## Usage

Install Motius and the official Jukebox feature frontend:

```bash
python -m pip install -e '.[music-to-dance]'
python -m pip install 'jukemirlib @ git+https://github.com/rodrigo-castellon/jukemirlib.git'
```

Generate from raw audio:

```python
from motius.pipelines.edge import EDGEPipeline

pipe = EDGEPipeline.from_pretrained(
    "ZeyuLing/Motius-EDGE-AISTPP",
    device="cuda",
)
result = pipe(
    "music.wav",
    seed=7,
    jukebox_cache_dir="checkpoints/models/edge/jukebox_cache",
)

print(result.joints.shape)       # (frames, 24, 3), Y-up metres
print(result.edge_motion.shape)  # (frames, 151)
print(result.motion_135.shape)    # (frames, 135)
```

The released EDGE checkpoint does not contain the frozen OpenAI Jukebox 5B
frontend. `jukemirlib` downloads its VQ-VAE and level-2 prior on first use.
For reproducible offline setup, the expected files are:

| File | SHA-256 |
| ---- | ------- |
| `vqvae.pth.tar` | `69745413a48e887f8a3fe91b972a6f7f434021a1ce911a99187b331eb48c059a` |
| `prior_level_2.pth.tar` | `89a1dd14f5b2f9b16b3e73b53fa2138cc89fd96bb13249b4267fea471de92672` |

Precomputed `(N,150,4800)` feature windows can be passed directly and avoid
loading Jukebox:

```python
result = pipe(music_features=features, seed=7)
```

## Motion Representation

`EDGE-151` is
`[contacts(4), root_position(3), SMPL24_local_rot6d(144)]`. EDGE stores
PyTorch3D's first-two-rows 6D rotation convention in a Z-up frame. Motius
decodes native joints exactly, then converts to Y-up `motion135` by:

1. applying the inverse 90-degree X basis transform to root position and root
   rotation;
2. preserving the other local rotations;
3. re-encoding rotations in Motius's motion135 first-two-columns convention.

This route uses no IK. Mesh shape can still differ slightly from the released
fixed joint offsets when a different SMPL gender or beta is selected.

## Verification

| Check | Result |
| ----- | ------ |
| Official checkpoint load | Zero missing and zero unexpected tensors |
| Official checkpoint SHA-256 | `28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7` |
| Raw-audio smoke inference | 225 frames from 7.5 seconds, all finite |
| Native FK bone-length temporal deviation | below `0.001 mm` |
| EDGE-to-motion135 fixed-skeleton agreement | below `0.001 mm` maximum |
| Three.js overlay root agreement | below `0.0002 mm` at audited frames |
| Unit/browser tests | 20 passed |

The complete 40-case AIST++ leaderboard run is not reported in this initial
integration card. It will be added only after the common cross-frame-rate
protocol is fixed and the generated set is complete.

## Citation

```bibtex
@inproceedings{tseng2023edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C. Karen},
  booktitle={CVPR},
  year={2023}
}
```
