---
license: other
library_name: motius
tags:
  - multimodal-generation
  - motion-generation
  - music-generation
  - motion-captioning
  - music-captioning
  - humanml3d
  - aistplusplus
  - unimumo
---

<h1 align="center">UniMuMo Model Card</h1>

<p align="center">
  <strong>One checkpoint for generation and translation across text, music, and motion.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.04534">Paper</a> |
  <a href="https://hanyangclarence.github.io/unimumo_demo/">Project Page</a> |
  <a href="https://github.com/hanyangclarence/UniMuMo">Original GitHub</a> |
  <a href="https://huggingface.co/ClarenceY/unimumo">Original Checkpoint</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-UniMuMo">Motius Checkpoint</a> |
  <a href="https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard">T2M Leaderboard</a> |
  <a href="https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard">M2T Leaderboard</a> |
  <a href="https://huggingface.co/spaces/ZeyuLing/music-to-dance-aistpp-leaderboard">M2D Leaderboard</a> |
  <a href="https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard">D2M Leaderboard</a>
</p>

UniMuMo is the unified text, music, and motion model introduced in *UniMuMo:
Unified Text, Music and Motion Generation*. Motius independently implements its
inference architecture and converts the authors' published weights into one
self-contained safe artifact. Runtime inference does not import an upstream
checkout or download a second text, audio, motion, or caption model.

## Preview

- [HumanML3D all-case text-to-motion comparison](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html)
- [HumanML3D all-case motion caption comparison](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html)
- [Audio-synchronized AIST++ all-case dance comparison](https://zeyuling-music-to-dance-aistpp-leaderboard.static.hf.space/cases/index.html)
- [Motion-synchronized AIST++ all-case music comparison](https://zeyuling-dance-to-music-aistpp-leaderboard.static.hf.space/cases/index.html)

The T2M page compares UniMuMo's generated SMPL Mesh with every released
baseline over all 4,042 selected-caption cases. The M2T page shows the same
animated input SMPL Mesh beside every baseline caption for all 4,400 protocol
samples. The M2D page contains all 40 AIST++ cases, synchronized audio, native
SMPL-24 joints, the fitted SMPL Mesh, orbit, zoom, timeline seeking, and
downloadable motion assets. The D2M page reuses the same 40 dances and lets
the viewer switch between reference and generated audio while the mesh plays.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Tasks | T2M, M2T, Music-to-Dance, Dance-to-Music |
| Additional pipeline routes | Text-to-Music and joint Text-to-Music-Motion |
| Motion representation | HumanML3D-263 at 60 fps |
| Audio representation | Encodec, 32 kHz, four 2,048-entry RVQ codebooks |
| Shared code rate | 50 Hz |
| Generator | 24-layer, 1,024D dual-stream autoregressive Transformer |
| Text conditioning and captioning | T5-base encoder and T5-base captioner |
| Maximum duration | 10 seconds per call |
| Checkpoint | [`ZeyuLing/Motius-UniMuMo`](https://huggingface.co/ZeyuLing/Motius-UniMuMo) |
| Pipeline | `motius.pipelines.unimumo.UniMuMoPipeline` |
| Upstream revision | `hanyangclarence/UniMuMo@a75ddac791ff6806b5bd511d1ce887a1980e20d5` |

The artifact includes both core shards, Encodec, T5 encoder, T5 captioner,
SentencePiece tokenizer, HumanML3D normalization statistics, configuration,
and provenance. `UniMuMoPipeline.from_pretrained` is the only loader needed.

## Usage

Install the UniMuMo dependencies and load the complete artifact:

```bash
python -m pip install -e '.[unimumo]'
```

```python
from motius.pipelines.unimumo import UniMuMoPipeline

pipe = UniMuMoPipeline.from_pretrained(
    "ZeyuLing/Motius-UniMuMo",
    device="cuda",
)
```

Generate synchronized music and motion from two optional text descriptions:

```python
result = pipe.infer_text_to_music_motion(
    music_prompt="an upbeat electronic dance track",
    motion_prompt="a person dances energetically",
    duration_seconds=8.0,
    guidance_scale=4.0,
    seed=7,
)

print(result.waveform.shape, result.sample_rate)  # (256000,), 32000
print(result.motion.shape, result.motion_fps)     # (480, 263), 60.0
print(result.joints.shape)                       # (480, 22, 3)
```

Use the task-specific routes with the same loaded pipeline:

```python
t2m = pipe.infer_text_to_motion(
    "a person walks in a circle",
    duration_seconds=6.0,
    seed=7,
)
t2music = pipe.infer_text_to_music(
    "a quiet piano melody",
    duration_seconds=6.0,
    seed=7,
)
m2d = pipe.infer_music_to_motion(
    "music.wav",
    motion_prompt="a person performs a street dance",
    guidance_scale=3.0,
    seed=7,
)
motion_music = pipe.infer_motion_to_music(
    t2m.motion,
    input_fps=60.0,
    music_prompt="upbeat percussion",
    seed=7,
)
motion_caption = pipe.infer_motion_to_text(t2m.motion, input_fps=60.0)
music_caption = pipe.infer_music_to_text("music.wav")
```

Array audio inputs also require `sample_rate=...`. Motion inputs must be
HumanML3D-263; other Motius representations should first be converted through
the motion representation API.

## Evaluation Results

### HumanML3D Text-to-Motion

UniMuMo exposes text-to-motion as a zero-shot route of its joint music-motion
generator. Motius evaluates all 4,042 selected-caption protocol cases with one
deterministic pass and retrieval groups of 32. HumanML3D Official uses the
4,012 cases for which the released `new_joint_vecs` reference exists; its
retrieval computation uses the largest complete 32-case groups (`n=4,000`).

| Evaluator | n | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | -: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | 4,000 | 0.1000 | 0.1775 | 0.2468 | 1.4849 | 6.6372 | 9.0766 |
| MotionStreamer Evaluator | 4,032 | 0.0655 | 0.1138 | 0.1617 | 373.2192 | 25.6637 | 18.8368 |
| Motius Joint-Position Evaluator | 4,032 | 0.0704 | 0.1471 | 0.2093 | 0.6788 | 54.0101 | 46.7609 |

HumanML3D and MotionStreamer FID use their native evaluator spaces; uTMR FID
uses per-sample L2-normalized embeddings. The weak retrieval scores are
reported as measured: this checkpoint supports T2M, but it was not optimized
as a dedicated HumanML3D text-to-motion model.

The HumanML3D result above is computed from the codec's native 60 fps output
with the phase-aligned `[1::3]` inverse used by the official UniMuMo data
pipeline. A parity audit found and fixed a top-k sampling-order discrepancy;
under the upstream dependency versions, motion codes, sampled tokens, and
decoded features now match the released implementation exactly. The UniMuMo
paper does not report a standalone HumanML3D T2M leaderboard result, so the
remaining low retrieval score is recorded as the zero-shot operating point of
the released joint model rather than treated as a paper-parity target.

Physical diagnostics over all 4,042 generated SMPL-22 joint sequences:

| Slide | Float | Jitter | Dynamic | Penetration |
| ----: | ----: | -----: | ------: | ----------: |
| 21.7377 | 58.7669 | 11.8780 | 54.3552 | 0.0000 |

### HumanML3D Motion-to-Text

The shared M2T protocol contains 4,400 official test motions and temporal
subclips, three references per sample, semantic retrieval groups of 32, and
one deterministic evaluation pass. UniMuMo follows the authors' 10-second
captioning protocol: 20 fps HumanML3D input is padded to 200 frames, linearly
resampled to 60 fps, encoded, and captioned.

| Method | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | BERT raw | R@1 | R@2 | R@3 | MM-Dist |
| ------ | -----: | -----: | ------: | ----: | -------: | --: | --: | --: | ------: |
| UniMuMo | 0.3534 | 0.0457 | 0.2822 | 0.0635 | 0.9006 | 0.5162 | 0.7032 | 0.7984 | 2.9658 |

The lexical metrics use the TM2T token/lemma references. BERT raw is the
unrescaled RoBERTa-large layer-17 cosine score; the corresponding
baseline-rescaled BERTScore is `0.4109`. The paper reports `R@1=0.520`,
`R@3=0.806`, and `MM-Dist=2.958`, which closely matches this independent run.
The leaderboard's raw-reference diagnostic reports BLEU-4 `0.1271`, ROUGE-L
`0.3560`, CIDEr `0.3114`, and raw BERTScore `0.9075`.

### AIST++ Music-to-Dance

The common leaderboard evaluates all 40 public cross-modal cases against the
complete 1,320-motion AIST++ reference pool. FID_k/FID_g and diversity use the
released Bailando 60 fps protocol. uTMR FID uses canonical SMPL-22 joints at
30 fps with per-sample L2-normalized embeddings.

| Result | FID_k | FID_g | uTMR FID | Div_k | Div_g | BeatAlign |
| ------ | ----: | ----: | --------: | ----: | ----: | --------: |
| UniMuMo | 17.7250 | 38.6446 | 0.2823 | 8.8767 | 8.4657 | 0.2430 |
| Motius GT | 17.1589 | 10.6618 | 0.1829 | 8.1666 | 7.4893 | 0.2247 |

Physical diagnostics on the same generated clips:

| Jitter | Dynamic | Penetration | Float | Slide |
| -----: | ------: | ----------: | ----: | ----: |
| 0.00982 | 0.02523 | 0.00000 | 0.19176 | 0.00523 |

For paper parity, a separate first-five-second evaluation gives
`FID_k=10.7721`, `FID_g=27.3115`, and `BeatAlign=0.2430`; the paper reports
BeatAlign `0.24`. The leaderboard uses full generated timelines for every
method and does not mix the shorter parity result into rankings.

### AIST++ Dance-to-Music

Motion-to-music is a zero-shot route of the same released checkpoint. The
authors evaluate on the D2M-GAN 2-second AIST++ split; Motius records those
published numbers separately from its longer common-case diagnostic.

| Protocol | Samples | Beats Coverage | Beats Hit |
| -------- | ------: | -------------: | --------: |
| UniMuMo paper, D2M-GAN 2-second split | paper test split | 93.0% | 88.4% |
| Motius common AIST++ cases, up to 10 seconds | 40 | 108.11% | 39.19% |

The common-case coverage is the generated/reference beat-count ratio
(`640/592`); hit is one-to-one beat matching within `0.1 s` (`232/592`). This
run uses the task's official test default (`CFG=3`, temperature `1.0`,
top-k `250`). The
long-window diagnostic is useful for inspecting complete generated songs but
is not directly comparable to the paper's 2-second protocol. Every WAV,
input-motion package, prompt, seed, and codec output is available in the
[`dance_to_music_aistpp_common40` benchmark folder](https://huggingface.co/ZeyuLing/Motius-UniMuMo/tree/main/benchmarks/dance_to_music_aistpp_common40).

[Open the Dance-to-Music leaderboard and synchronized 40-case SMPL/audio viewer](https://huggingface.co/spaces/ZeyuLing/dance-to-music-aistpp-leaderboard).

## Motion Representation

The native motion stream is HumanML3D-263. Its root velocities, root height,
root-invariant positions, continuous 6D local rotations, local velocities, and
foot contacts are normalized with the authors' published statistics and
encoded jointly with zero-audio embeddings. The motion codec maps 60 fps motion
to the same 50 Hz, four-codebook token clock used by music.

For the AIST++ viewer and evaluator, Motius decodes HumanML3D joints, converts
the common SMPL-22 body directly, and extrapolates AIST++ hand joints 22 and 23
from each elbow-to-wrist direction by `0.35x`. The official feature evaluator
uses 24 joints; uTMR uses only the common SMPL-22 body. SMPL Mesh preview uses
position IK because generated joint positions do not uniquely determine axial
twist. Across all 40 clips, the fitted mesh has `28.16 mm` mean joint MPJPE.

## Reproduction Audit

| Check | Result |
| ----- | ------ |
| Motion-code encoding vs official implementation | Exact equality, zero differing codes |
| Seeded music generation vs official implementation | Exact equality, zero differing codes |
| Seeded motion generation vs official implementation | Exact equality, zero differing codes |
| Motion caption vs official implementation | Exact string equality |
| Official tensor load | 880 core tensors loaded across two safe shards |
| HumanML3D caption generation | 4,400/4,400 protocol samples |
| AIST++ dance generation | 40/40 cases, all finite |
| AIST++ music generation from dance | 40/40 cases, all WAV and codec metadata public |
| HumanML3D zero-shot motion generation | 4,042/4,042 selected-caption cases, all four output representations finite |
| Runtime boundary | No import from `ref_repo` or an upstream checkout |

The audited upstream source and checkpoint declare no license. Motius records
that fact rather than inferring redistribution terms. Users remain responsible
for obtaining permission appropriate to their use case.

## Citation

```bibtex
@article{yang2024unimumo,
  title={UniMuMo: Unified Text, Music and Motion Generation},
  author={Yang, Han and Su, Kun and Zhang, Yutong and Chen, Jiaben and Qian, Kaizhi and Liu, Gaowen and Gan, Chuang},
  journal={arXiv preprint arXiv:2410.04534},
  year={2024}
}
```
