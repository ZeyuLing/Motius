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
  <a href="https://openaccess.thecvf.com/content/CVPR2023/papers/Tseng_EDGE_Editable_Dance_Generation_From_Music_CVPR_2023_paper.pdf">Paper</a> ·
  <a href="https://edge-dance.github.io/">Project Page</a> ·
  <a href="https://github.com/Stanford-TML/EDGE">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP">Motius Checkpoint</a> ·
  <a href="https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/cases/index.html">M2D Leaderboard + Preview</a>
</p>

EDGE is the CVPR 2023 work *EDGE: Editable Dance Generation From Music*.
Motius reproduces the released AIST++ network, cosine DDIM sampler, classifier-
free guidance schedule, long-sequence overlap, Jukebox conditioning contract,
motion decoder, and coordinate conversion without importing an upstream
checkout at runtime.

<!-- MOTIUS_MODEL_CARD_NAV:START -->
<p align="center">
  <a href="#visual-results">Visual Results</a> ·
  <a href="#model-overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#motion-representation">Motion Representation</a>
</p>
<!-- MOTIUS_MODEL_CARD_NAV:END -->

## Visual Results

<!-- MOTIUS_TASK_DEMOS:START -->

### Task Demos

| Task | Input / condition | Rendered output | More |
| --- | --- | --- | --- |
| Music-to-Dance | Break music | <video src="https://github.com/user-attachments/assets/d03b6962-c83e-4238-9d91-5c558f22883d" controls></video> | [MP4](https://github.com/user-attachments/assets/d03b6962-c83e-4238-9d91-5c558f22883d) · [All cases](https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/cases/index.html?method=edge) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->

| Music | SMPL and native-skeleton preview |
| --- | --- |
| Break | <video src="https://github.com/user-attachments/assets/d03b6962-c83e-4238-9d91-5c558f22883d" controls></video> |

[Open the unified audio-synchronized 40-case M2D comparison](https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/cases/index.html?method=edge).
Every case shows GT, Bailando, and EDGE in one comparison page. The EDGE scene
overlays its native SMPL-24 skeleton on the SMPL mesh decoded from the same
local rotations. It supports orbit, zoom, timeline seeking, audio
synchronization, NPZ export, and FBX export.

The viewer preserves the generated heading and XZ trajectory. It applies one
clip-wide vertical translation for display and never grounds individual
frames, so jumps, foot slide, and root-height drift remain visible.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Music-to-Dance | `infer_music_to_dance` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps motion and Jukebox features |
| Public preview | 30 fps native with synchronized audio |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Tasks | Music-to-Dance |
| Dataset | AIST++ |
| Music input | Jukebox layer 66, 4,800D at 30 fps |
| Native motion | EDGE-151 at 30 fps |
| Window | 150 frames with 75-frame overlap |
| Parameters | 49,464,471 |
| Checkpoint | [`ZeyuLing/Motius-EDGE-AISTPP`](https://huggingface.co/ZeyuLing/Motius-EDGE-AISTPP) |
| Pipeline | `motius.pipelines.edge.EDGEPipeline` |
| Upstream revision | `Stanford-TML/EDGE@17c3428669ed6733edd9d8c66f7dc62060b8e46d` |
| License | MIT |

## Quick Start

Install Motius and the official Jukebox feature frontend:

```bash
python -m pip install -e '.[music-to-dance]'
python -m pip install 'jukemirlib @ git+https://github.com/rodrigo-castellon/jukemirlib.git'
```

Generate from raw audio:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-EDGE-AISTPP",
    device="cuda",
)
result = pipe.infer_music_to_dance(
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
result = pipe.infer_music_to_dance(music_features=features, seed=7)
```

Reproduce the public 40-case package with resumable per-case outputs. Extract
the ten unique music tracks once:

```bash
python tools/infer_edge_aistpp.py \
  --checkpoint ZeyuLing/Motius-EDGE-AISTPP \
  --case-manifest docs/leaderboards/hf_space_music_to_dance/cases/manifest.json \
  --audio-root docs/leaderboards/hf_space_music_to_dance/cases \
  --feature-root outputs/edge/aistpp_jukebox_features \
  --output outputs/edge/aistpp_official_40 \
  --extract-features-only \
  --jukebox-cache-dir checkpoints/models/edge/jukebox_cache
```

Then generate all 40 cases:

```bash
python tools/infer_edge_aistpp.py \
  --checkpoint ZeyuLing/Motius-EDGE-AISTPP \
  --case-manifest docs/leaderboards/hf_space_music_to_dance/cases/manifest.json \
  --feature-root outputs/edge/aistpp_jukebox_features \
  --output outputs/edge/aistpp_official_40 \
  --seed 20260721
```

The manifest mode uses each case's exact duration, assigns a unique deterministic
seed, and preserves EDGE's native 30 fps timeline. Jukebox features are shared
by music id, so the four cases associated with one song do not recompute the
10.3 GB frontend.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Music-to-Dance | [Published results](../leaderboards/hf_space_music_to_dance/music_to_dance_results.json) | AIST++ music-to-dance and normalized uTMR FID |

### Canonical Music-to-Dance Snapshot

| Method | n | FID_k | FID_g | Motius FID (normalized) | Diversity_k | Diversity_g | BeatAlign |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| EDGE | 40 | 38.0561 | 20.6397 | 0.2503 | 3.9275 | 3.4718 | 0.2562 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Motius evaluates the released checkpoint on the same 40 AIST++ cross-modal
cases used by the public Bailando row. Official kinetic/geometric features are
computed after phase-aligned interpolation from EDGE's native 30 fps output to
the Bailando 60 fps feature timeline. BeatAlign uses the native 30 fps timing;
uTMR uses canonical 30 fps SMPL-22 joints and per-sample L2-normalized
embeddings against the fixed 1,320-motion reference pool.

| Result | FID_k | FID_g | uTMR FID | Diversity_k | Diversity_g | BeatAlign |
| ------ | ----: | ----: | --------: | ----------: | ----------: | --------: |
| EDGE official checkpoint | 38.06 | 20.64 | 0.2503 | 3.93 | 3.47 | 0.2562 |
| Bailando reproduction | 28.11 | 9.70 | 0.3138 | 7.73 | 6.31 | 0.2271 |
| Motius GT | 17.16 | 10.66 | 0.1829 | 8.17 | 7.49 | 0.2293 |

Lower is better for FID, higher is better for BeatAlign, and diversity should
be interpreted relative to GT. These are Motius common-protocol measurements,
not values copied from the EDGE paper. EDGE's paper evaluates its own generated
sample population and introduces PFC; its published protocol is not presented
as if it were the same 40-case Bailando benchmark.

### Physical Diagnostics

| Result | Jitter | Dynamic | Penetration | Float | Slide |
| ------ | -----: | ------: | ----------: | ----: | ----: |
| EDGE | 0.00541 | 0.02030 | 0.00000 | 0.32649 | 0.00479 |
| Paired GT | 0.00677 | 0.02276 | 0.00000 | 0.10658 | 0.00330 |

These diagnostics use the shared SMPL-22 subset on the common 60 fps metric
timeline. `Dynamic` measures expressiveness relative to GT rather than an
error to minimize.

### Verification

| Check | Result |
| ----- | ------ |
| Official checkpoint load | Zero missing and zero unexpected tensors |
| Official checkpoint SHA-256 | `28ca4ce167bb17c36869b4d021af8762a34c6df034002f61b3bc1c1d0b1b02c7` |
| Raw-audio smoke inference | 225 frames from 7.5 seconds, all finite |
| Common-protocol generation | 40/40 cases, exact manifest lengths, all finite |
| Seeds | 40/40 unique and deterministic from base seed `20260721` |
| Native FK bone-length temporal deviation | below `0.001 mm` |
| EDGE-to-motion135 fixed-skeleton agreement | below `0.001 mm` maximum |
| Three.js overlay root agreement | below `0.0002 mm` at audited frames |
| Unified Three.js viewer | 40 cases, GT/Bailando/EDGE, zero browser errors |
| Unit/browser tests | 43 passed |

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

## Citation and License

```bibtex
@inproceedings{tseng2023edge,
  title={EDGE: Editable Dance Generation From Music},
  author={Tseng, Jonathan and Castellon, Rodrigo and Liu, C. Karen},
  booktitle={CVPR},
  year={2023}
}
```

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
