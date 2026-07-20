---
license: other
license_name: s-lab-license-1.0
license_link: https://github.com/lisiyao21/Bailando/blob/master/LICENSE
library_name: motius
tags:
  - evaluation
  - music-to-dance
  - aistplusplus
  - bailando
datasets:
  - yeok/danceba
---

<h1 align="center">AIST++ Music-to-Dance Evaluator</h1>

<p align="center">
  <strong>Checkpoint-free Bailando protocol for dance quality, diversity, and beat alignment.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2203.13055">Bailando Paper</a> |
  <a href="https://github.com/lisiyao21/Bailando">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-Evaluator-AISTPP-Music-to-Dance">Motius Protocol Artifact</a> |
  <a href="https://github.com/ZeyuLing/Motius/blob/main/docs/tasks/music_to_dance.md">Task Protocol</a>
</p>

## Protocol

| Metric | Input | Interpretation |
| ------ | ----- | -------------- |
| `FID_k` | 72D kinetic clip features | Lower is better |
| `FID_g` | 32D geometric clip features | Lower is better |
| `Diversity_k` | Pairwise distance in normalized kinetic space | Compare with GT |
| `Diversity_g` | Pairwise distance in normalized geometric space | Compare with GT |
| `BeatAlign` | Music beats to local minima of mean joint speed | Higher is better |

Generated clips are first-root anchored and cropped to 1,200 frames. Kinetic
and geometric reference features are extracted from each complete AIST++ v1
sequence without cropping. Both sides are normalized with the complete GT
feature-pool statistics before FID and Diversity are computed. Beat alignment
uses the complete generated clip,
`sigma = 3` frames, and the original 60 fps music beat stream truncated to the
motion-velocity sequence length.

The feature extractors are the Bailando/Fairmotion implementations vendored
under `motius.evaluation.metrics.dance_features`. They have no learned
checkpoint. The evaluator additionally reports checkpoint-free Motius physical
diagnostics on joints `0:22`; these diagnostics are not part of the CVPR paper
table.

## Python API

```python
from motius.evaluation import AISTPPMusicDanceEvaluator

evaluator = AISTPPMusicDanceEvaluator.from_pretrained(
    "ZeyuLing/Motius-Evaluator-AISTPP-Music-to-Dance",
)
evaluator.process(
    {
        "name": sequence_id,
        "pred_joints": generated_smpl24,
        "gt_joints": paired_gt_smpl24,
        "music_beats": full_rate_music_features[:, 53].astype(bool),
        "music_fps": 60.0,
        "motion_fps": 60.0,
    }
)
metrics = evaluator.compute()
```

## Reference Pool Audit

The AIST++ v1 metadata declares 1,408 sequences, while the released motion
archive contains 1,365 SMPL PKLs. Applying the official 45-entry ignore list
leaves 1,320 valid motions, all of which are recorded by name in the protocol
artifact. The original Google Storage link expired in
2026 and its replacement GitHub `motions.zip` currently contains only 411
motions. Motius therefore rebuilds the complete pool from the preserved public
archive mirror at [`yeok/danceba`](https://huggingface.co/datasets/yeok/danceba),
revision `637d0aadf69e3e926ba70bfee9ff89571fd18813`. A sample of 86 overlapping
motion files matches the current official GitHub release byte for byte. Source
names, skipped entries, hashes, and the calibrated SMPL-24 skeleton report are
shipped with the evaluator artifact.

The converted official checkpoints reproduce the paper table on all 40
cross-modal evaluation cases:

| Row | FID_k | FID_g | Diversity_k | Diversity_g | BeatAlign |
| --- | ----: | ----: | ----------: | ----------: | --------: |
| Motius GT | 17.16 | 10.66 | 8.17 | 7.49 | 0.2247 |
| GT paper | 17.10 | 10.60 | 8.19 | 7.45 | 0.2374 |
| Motius Bailando | 28.11 | 9.70 | 7.73 | 6.31 | 0.2268 |
| Bailando paper | 28.16 | 9.62 | 7.83 | 6.34 | 0.2332 |

The Motius rows are computed outputs; the paper rows are parity targets and
are not hard-coded by the evaluator.

## Provenance

The implementation follows `lisiyao21/Bailando` revision
`cc90b98bff81c9709570db413c9610c2562e27ca`. Bailando is distributed under the
S-Lab License 1.0; the kinetic/geometric files retain Fairmotion BSD headers.
