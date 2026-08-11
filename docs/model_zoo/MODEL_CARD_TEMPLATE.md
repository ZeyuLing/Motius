# Motius Model Card Template

Use this structure for every public method package. Keep method-specific
details, but do not rename or omit the six required level-two sections.

<h1 align="center">Method Name Model Card</h1>

<p align="center">
  <strong>One sentence describing the method's user-facing capability.</strong>
</p>

<p align="center">
  <a href="PAPER_URL">Paper</a> ·
  <a href="SOURCE_URL">Original GitHub</a> ·
  <a href="CHECKPOINT_URL">Motius Checkpoint</a>
</p>

The opening paragraph identifies the original work, venue, Motius integration
scope, and whether inference is standalone.

## Visual Results

The generated `Task Demos` table contains exactly one row for every public
`infer_{task}` API. Motion-generating tasks use the shared
`motius-threejs-floor-v1` scene and a GitHub-native H.264 video player. Show
SMPL Mesh only when the representation bridge is validated; otherwise show
the method's native skeleton, robot, or body mesh and name that representation
in the render manifest. Every scene uses the same matte floor, neutral grid,
lighting, and camera controls. Matplotlib and method-specific 2D renderers are
not public preview backends. A method-filtered all-case Three.js viewer is
supplementary. Text and audio outputs must use their task-native media, and
every audio-conditioned preview must contain the synchronized audio stream.
Do not let one task's media stand in for another task, and do not repeat one
prompt across every inline row. Inline preview tables show the user input and
result only; dataset case IDs are release metadata and must not be displayed.
Upload MP4s through GitHub's attachment service and record their stable URLs in
`video_attachments.json`; repository-relative MP4s, GIFs, and WebPs are not
valid primary previews because GitHub does not render them as video players.
Every source timeline frame is rendered once. Raising the MP4 container frame
rate by duplicating a sparsely sampled GIF is not valid.

## Model Overview

The generated Task APIs table belongs first in this section, followed by the
generated Frame-Rate Contract. The latter must distinguish the checkpoint's
native training clock from the public preview clock and state any
duration-preserving resampling explicitly.

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | Native checkpoint training FPS |
| Public preview | Playback FPS and any duration-preserving resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only
controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->

| Item | Value |
| --- | --- |
| Tasks | Canonical task names |
| Native representation | Exact tensor or body-model representation and FPS |
| Checkpoint | Exact Hugging Face repository |
| Pipeline | Fully-qualified Pipeline class |
| Release status | Release-ready, restricted runtime, or unregistered |

## Quick Start

Use `Pipeline.from_pretrained` with a literal repository ID and show every
supported `infer_{task}` entrypoint. Explain output type, shape, physical scale,
and optional licensed assets.

For a package with a registered Motius trainer, add a `### Training` subsection
here. It must link the public config and [Training Hub](../training/README.md),
name the required dataset/manifest and native representation, state whether the
recipe trains from scratch or warm-starts, list precision and optimized loss
terms, show single-node/distributed launch with `--auto-resume`, describe
checkpoint cadence, and keep the work directory below `outputs/training/`.
Do not describe a package as trainable merely because its upstream repository
contains training code.

## Evaluation Results

Link the named benchmark protocol, state population size and metric direction,
and report measured values synchronized from the machine-readable Leaderboard
snapshot. Motius/uTMR FID always means per-sample L2-normalized
embedding-space FID. Never publish the historical raw-space value under the
same label, and never replace missing results with fabricated numbers.

## Motion Representation

Describe native inputs and outputs, FPS, skeleton/body model, normalization,
coordinate convention, and any lossy representation bridge used for evaluation
or visualization.

## Citation and License

Include the original paper and source repository, BibTeX when available,
upstream license, vendored attribution path, and restrictions on weights or
body models. For Motius-native methods, state that no external paper or source
repository is claimed.

Synchronize and validate generated content before release:

```bash
python tools/export_t2m_leaderboard_results.py --check
python tools/audit_model_card_media.py
python tools/publish_model_card_videos.py
python tools/sync_model_card_content.py
python tools/audit_model_card_format.py
python tools/audit_model_card_content.py
```
