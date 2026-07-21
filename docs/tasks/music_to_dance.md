# Music-to-Dance on AIST++

Motius exposes music-to-dance methods through one audio/feature input contract
and a shared SMPL motion bridge. Bailando's native output is 60 fps AIST++
SMPL-24 joints; cross-method Motius metrics use canonical 30 fps SMPL-22
joints. EDGE uses a 30 fps contact/root/SMPL-rotation representation and is
available as a checkpoint-verified integration while its full common-protocol
benchmark run is being completed.

[Open the public Music-to-Dance Leaderboard](https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard),
including the audio-synchronized all-case GT/Bailando 3D comparison.
Its Three.js viewer supports free orbit, zoom, view reset, timeline seeking,
and synchronized playback for all 40 cases. Each method is one scene: its
native SMPL-24 skeleton is drawn as a coral X-ray overlay on the position-IK
SMPL Mesh. Only the mesh is affected by IK ambiguity.

## Task Contract

| Item | Definition |
| ---- | ---------- |
| Audio input | File path or mono waveform with an explicit sample rate |
| Bailando feature input | `(T, 438)` at 7.5 fps: MFCC, delta MFCC, chroma, onset, beat, and tempogram |
| Motion output | `(T * 8, 24, 3)` global AIST++ SMPL joint positions at 60 fps |
| Coordinate system | SMPL Y-up world frame, positions in metres |
| Unconditioned start | Released Bailando code pair `(423, 12)` |
| Official AIST++ start | First upper/lower VQ token encoded from the paired GT motion |

Raw audio is resampled to 3,840 Hz before the released 438D Bailando feature
extractor is applied. Passing precomputed features skips that deterministic
front end.

## Inference

```python
from motius.pipelines.bailando import BailandoPipeline

pipeline = BailandoPipeline.from_pretrained(
    "ZeyuLing/Motius-Bailando-AISTPP",
    device="cuda",
)

output = pipeline("music.wav")
print(output.joints.shape)  # (1, frames, 24, 3)
```

Reproduce the released AIST++ generation protocol:

```bash
python tools/infer_bailando_aistpp.py \
  --data-root /path/to/data/aistpp_test_full_wav \
  --music-feature-root /path/to/data/aistpp_music_feat_7.5fps \
  --checkpoint ZeyuLing/Motius-Bailando-AISTPP \
  --output outputs/music_to_dance/bailando/aistpp
```

Each case is stored independently and existing cases are skipped, so the
command can resume after interruption or elastic-worker eviction.

### EDGE

EDGE consumes raw audio through the released Jukebox layer-66 frontend, or
precomputed feature windows shaped `(N,150,4800)`. Its native 151D output is
converted directly to Y-up SMPL joints and `motion135`; no position IK is used.

```python
from motius.pipelines.edge import EDGEPipeline

pipeline = EDGEPipeline.from_pretrained(
    "ZeyuLing/Motius-EDGE-AISTPP",
    device="cuda",
)
result = pipeline(
    "music.wav",
    seed=7,
    jukebox_cache_dir="checkpoints/models/edge/jukebox_cache",
)
```

See the [EDGE model card](../model_zoo/edge.md) for exact Jukebox hashes,
representation conventions, and the interactive skeleton/mesh overlay.

The public viewer uses motion-length MP3 clips derived from the official
[AIST Dance Video Database audio release](https://aistdancedb.ongaaccel.jp/database_download/).
Rebuild the clips and provenance manifest with:

```bash
python tools/build_aistpp_gallery_audio.py \
  --manifest docs/leaderboards/hf_space_music_to_dance/cases/manifest.json \
  --output-dir docs/leaderboards/hf_space_music_to_dance/cases/audio
```

## Representation Bridge

The native output is registered as `aistpp_smpl24_joints`. Its first 22 joints
are the standard SMPL body chain and can be selected exactly:

```python
from motius.motion import convert_motion

smpl22 = convert_motion(
    output.joints[0],
    source="aistpp_smpl24_joints",
    target="smpl22_joints",
)
```

Conversion to `motion135` uses position IK because the AIST++ tensor does not
store joint rotations:

```python
motion135 = convert_motion(
    output.joints[0],
    source="aistpp_smpl24_joints",
    target="motion135",
    model_dir="checkpoints/body_models/smpl",
    source_fps=60,
    target_fps=30,
    gender="male",
)
```

This second route is lossy and reports fit errors when called through the lower-
level `retarget_hml263_clip` API. It is intended for SMPL mesh rendering and
cross-representation tools; official Bailando metrics consume native joints.

The public two-scene overlay can be rebuilt from the native 60 fps outputs and
fitted 30 fps SMPL parameters with:

```bash
python tools/build_smpl_motion_gallery.py \
  --source-manifest outputs/bailando/leaderboard_gallery_source.json \
  --motion 'gt=GT=outputs/bailando/leaderboard_smpl/gt' \
  --motion 'bailando=Bailando=outputs/bailando/leaderboard_smpl/bailando' \
  --skeleton 'gt=GT native joints=path/to/aistpp_test_full_wav' \
  --skeleton 'bailando=Bailando native joints=outputs/bailando/aistpp_official_epoch10' \
  --skeleton-fps 60 --fps 30 --stride 2 \
  --output-dir docs/leaderboards/hf_space_music_to_dance/cases
```

### Coordinate and floor rules

The generation output and qualitative viewer preserve the native AIST++ world
heading and XZ trajectory. They do not move the first pelvis to the origin or
rotate the first pose to face `+Z`. The SMPL preview applies one clip-wide
vertical translation, then aligns the native skeleton to the fitted mesh with
one fixed frame-0 root transform. It never grounds individual frames. A jump or
model-predicted vertical root drift therefore remains visible instead of being
hidden by per-frame foot locking.

The uTMR evaluation path is intentionally different. After 30 fps resampling,
it applies one rigid transform per clip: first-frame pelvis XZ to the origin,
first-frame hip/shoulder facing to `+Z`, and the clip-wide SMPL foot-joint
minimum to `Y=0`. This transform preserves relative trajectory, velocity,
acceleration, and timing.

## Evaluation

The paper protocol generates 40 dances and extracts kinetic/geometric features
from the first 1,200 generated frames. The FID reference pool contains all 1,320
valid AIST++ v1 motion PKLs and uses each complete sequence without cropping. Music
beat alignment uses the complete generated clip and its matching full-rate 60
fps beat channel, not the 7.5 fps model input.

Build the reference pool and evaluate:

```bash
python tools/build_aistpp_reference_features.py \
  --motions-root /path/to/aist_plusplus_final/motions \
  --smpl-skeleton outputs/music_to_dance/aistpp_smpl24_skeleton.npz \
  --ignore-list /path/to/aist_plusplus_final/ignore_list.txt \
  --output outputs/music_to_dance/aistpp_reference_features.npz

python tools/build_aistpp_utmr_reference_embeddings.py \
  --motions-root /path/to/aist_plusplus_final/motions \
  --smpl-skeleton outputs/music_to_dance/aistpp_smpl24_skeleton.npz \
  --ignore-list /path/to/aist_plusplus_final/ignore_list.txt \
  --output outputs/music_to_dance/aistpp_reference_utmr_embeddings.npy \
  --device cuda

python tools/eval_music_to_dance.py \
  --data-root /path/to/data/aistpp_test_full_wav \
  --music-feature-root /path/to/data/aistpp_music_feat_7.5fps \
  --pred-root outputs/music_to_dance/bailando/aistpp \
  --joint-fid \
  --evaluator-artifact ZeyuLing/Motius-Evaluator-AISTPP-Music-to-Dance \
  --output outputs/music_to_dance/bailando/metrics.json
```

The evaluator reports `FID_k`, `FID_g`, `Diversity_k`, `Diversity_g`, and
`BeatAlign`, plus normalized `FID_uTMR` and Motius physical diagnostics on the
common SMPL-22 subset. The released evaluator artifact already contains both
1,320-motion reference pools; the two build commands document how they are
reproduced rather than being required for normal use.
