---
license: other
library_name: motius
tags:
  - motion-generation
  - text-to-motion
  - music-to-dance
  - humanml3d
  - aistplusplus
  - tm2d
---

<h1 align="center">TM2D Model Card</h1>

<p align="center">
  <strong>One shared motion tokenizer for text-to-motion and music-to-dance.</strong>
</p>

<p align="center">
  <a href="https://openaccess.thecvf.com/content/ICCV2023/papers/Gong_TM2D_Bimodality_Driven_3D_Dance_Generation_via_Music-Text_Integration_ICCV_2023_paper.pdf">Paper</a> |
  <a href="https://garfield-kh.github.io/TM2D/">Project Page</a> |
  <a href="https://github.com/Garfield-kh/TM2D">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-TM2D-HumanML3D-AISTPP">Motius Checkpoint</a> |
  <a href="https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard">T2M Leaderboard</a> |
  <a href="https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard">M2D Leaderboard</a>
</p>

TM2D is the ICCV 2023 work *TM2D: Bimodality Driven 3D Dance Generation via
Music-Text Integration*. The full paper combines music and text. Its released
checkpoint also retains independently usable text-only and music-only branches;
Motius exposes both through one self-contained pipeline without importing an
upstream checkout at runtime.

## Preview

- [HumanML3D all-case T2M comparison](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html)
- [Audio-synchronized AIST++ all-case M2D comparison](https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/cases/index.html)

Both pages use the shared Three.js viewer with orbit, zoom, timeline seeking,
and downloadable motion assets. The AIST++ scene overlays TM2D's native
SMPL-24 joints on the neutral SMPL Mesh obtained from the same generated joint
positions. The T2M page uses the selected HumanML3D caption for every case.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Tasks | T2M, Music-to-Dance |
| Datasets | HumanML3D and AIST++ |
| Native motion | TM2D-287, 24 joints at 60 fps |
| Motion tokens | 1,024-entry VQ codebook, stride 8 |
| Text input | Official word/POS vocabulary, at most 20 lexical tokens, plus target length |
| Music input | AIST++ 438D audio features at 7.5 fps |
| Parameters | 101,267,841 unique parameters |
| Checkpoint | [`ZeyuLing/Motius-TM2D-HumanML3D-AISTPP`](https://huggingface.co/ZeyuLing/Motius-TM2D-HumanML3D-AISTPP) |
| Pipeline | `motius.pipelines.tm2d.TM2DPipeline` |
| Upstream revision | `Garfield-kh/TM2D@98bef9571419b6459927630d5d96f8450898687e` |
| Source checkpoints | VQ-VAE `E0190`; joint Transformer `E0020` |

The Hugging Face artifact contains the VQ encoder, codebook, decoder, text
Transformer, audio Transformer, normalization statistics, and text vocabulary.
`TM2DPipeline.from_pretrained` therefore needs no second checkpoint download.

## Usage

Install TM2D's text and audio dependencies, then download the spaCy English
model used by the released tokenizer:

```bash
python -m pip install -e '.[tm2d]'
python -m spacy download en_core_web_sm
```

Load the complete artifact once:

```python
from motius.pipelines.tm2d import TM2DPipeline

pipe = TM2DPipeline.from_pretrained(
    "ZeyuLing/Motius-TM2D-HumanML3D-AISTPP",
    device="cuda",
)
```

Text-to-motion requires a duration because the released text Transformer uses
an explicit motion-length indicator:

```python
result = pipe.infer_text_to_motion(
    "a person walks forward and turns left",
    duration_seconds=6.0,
    output_fps=30.0,
    seed=7,
)

print(result.joints.shape)       # (180, 24, 3)
print(result.model_motion.shape) # native 60 fps TM2D-287
```

Music-to-dance accepts raw audio or the released 438D feature stream:

```python
result = pipe.infer_music_to_dance("music.wav", seed=7)

print(result.joints.shape)         # (frames, 24, 3), 60 fps
print(result.music_features.shape) # (music frames, 438), 7.5 fps
```

The authors' paired AIST++ protocol samples the first VQ token from the paired
GT motion. Pass `initial_motion=paired_gt_tm2d287` to reproduce that behavior.
The public Motius leaderboard instead uses a deterministic reference-free VQ
seed for every clip, so inference does not consume test motion.

## Evaluation Results

### HumanML3D Text-to-Motion

All rows use the official HumanML3D test split and selected-caption protocol.
R-Precision is computed with retrieval batches of 32 and one deterministic
evaluation pass. uTMR FID is computed in per-sample L2-normalized embedding
space, as required for all new Motius evaluations.

| Evaluator | n | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | -: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | 3,970 | 0.1817 | 0.3004 | 0.3944 | 2.5038 | 5.1678 | 8.6228 |
| MotionStreamer Evaluator | 4,032 | 0.1324 | 0.2073 | 0.2674 | 548.5562 | 25.3085 | 12.2421 |
| Motius Joint-Position Evaluator | 4,032 | 0.2163 | 0.3529 | 0.4427 | 0.6703 | 49.0229 | 46.6538 |

The MotionStreamer result uses the independently canonicalized 30 fps
MotionStreamer-272 bridge. Its large FID is reported rather than hidden: TM2D
generates position-based TM2D-287, while this evaluator is especially sensitive
to its own 272D conversion and training distribution.

Physical diagnostics on 4,042 generated clips, in the shared joint-level metric
units:

| Jitter | Dynamic | Penetration | Float | Slide |
| -----: | ------: | ----------: | ----: | ----: |
| 0.01099 | 0.02368 | 0.00000 | 0.23983 | 0.00597 |

### AIST++ Music-to-Dance

The common 40-case package uses the complete 1,320-motion AIST++ reference pool
for FID and diversity. BeatAlign is evaluated on the full generated timeline;
uTMR uses canonical SMPL-22 joints at 30 fps and normalized embeddings.

| Result | FID_k | FID_g | uTMR FID | Diversity_k | Diversity_g | BeatAlign |
| ------ | ----: | ----: | --------: | ----------: | ----------: | --------: |
| TM2D reference-free | 43.42 | 20.42 | 0.2623 | 3.77 | 3.13 | 0.1903 |
| Motius GT | 17.16 | 10.66 | 0.1829 | 8.17 | 7.49 | 0.2247 |

Physical diagnostics for the same 40 clips:

| Jitter | Dynamic | Penetration | Float | Slide |
| -----: | ------: | ----------: | ----: | ----: |
| 0.01371 | 0.02173 | 0.00000 | 0.26373 | 0.00605 |

These are common-protocol measurements from the converted official checkpoint,
not values copied from the paper. The reference-free initialization is stricter
than the paired GT-token initialization used by the authors' released script.

## Motion Representation

`TM2D-287` is a 24-joint HumanML-style representation:

```text
root angular/planar velocity + root height                     4
root-invariant positions for joints 1..23                     69
continuous 6D local rotations for joints 1..23               138
local joint velocities for joints 0..23                       72
foot contacts                                                   4
                                                               ---
                                                               287
```

The released VQ encoder consumes the first 283 channels; the decoder predicts
all 287. Motius decodes native joints directly with `recover_from_ric`. SMPL
mesh export uses position IK because TM2D's generated joint positions do not
uniquely determine twist. T2M export uses the per-frame minimum-twist solution
to avoid accumulating unconstrained axial rotations. M2D retains temporal
twist stabilization for its continuous dance stream. Neither route runs
position-only pose refinement, which can reduce MPJPE while moving through
unconstrained twist solutions. Every release preview must pass joint, rotation,
and sampled SMPL-surface quality gates.

Across the 40 AIST++ cases, the corrected fit has `24.45 mm` mean joint MPJPE,
`13.81 deg` mean per-case local-rotation jump p99 (`19.94 deg` worst case), and
`1.333x` mean SMPL edge-length ratio p99 (`1.442x` worst case). The viewer also
overlays the native joints so residual fitting error remains visible.

## Reproduction Audit

| Check | Result |
| ----- | ------ |
| Audio encoder and logits vs upstream | maximum absolute error `0` |
| Text encoder and logits vs upstream | maximum absolute error `0` |
| VQ encoder, token indices, and decoder vs upstream | exact parity |
| Official tensor load | zero missing and zero unexpected tensors |
| HumanML3D generation | 4,042/4,042 selected-caption cases |
| AIST++ generation | 40/40 cases, all finite, exact 60 fps lengths |
| Runtime boundary | no import from `ref_repo` or an upstream checkout |

The audited upstream revision contains no license file. Motius records that
fact explicitly instead of assigning an inferred license to the authors' code
or weights. Users are responsible for obtaining any permission required for
their use case.

## Citation

```bibtex
@inproceedings{gong2023tm2d,
  title={TM2D: Bimodality Driven 3D Dance Generation via Music-Text Integration},
  author={Gong, Kehong and Lian, Dongze and Chang, Heng and Guo, Chuan and Jiang, Zihang and Zuo, Xinxin and Mi, Michael Bi and Wang, Xinchao},
  booktitle={ICCV},
  year={2023}
}
```
