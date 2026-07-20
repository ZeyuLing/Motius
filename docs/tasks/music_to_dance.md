# Music-to-Dance on AIST++

Motius exposes music-to-dance methods through one audio/feature input contract
and one 60 fps AIST++ SMPL-24 joint output contract. Bailando is the first
released baseline.

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

python tools/eval_music_to_dance.py \
  --data-root /path/to/data/aistpp_test_full_wav \
  --music-feature-root /path/to/data/aistpp_music_feat_7.5fps \
  --pred-root outputs/music_to_dance/bailando/aistpp \
  --reference-features outputs/music_to_dance/aistpp_reference_features.npz \
  --output outputs/music_to_dance/bailando/metrics.json
```

The evaluator reports `FID_k`, `FID_g`, `Diversity_k`, `Diversity_g`, and
`BeatAlign`, plus Motius physical diagnostics on the common SMPL-22 subset.
