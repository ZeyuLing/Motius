---
title: Motion-to-Text · HumanML3D
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Motion-to-Text · HumanML3D

Static leaderboard for motion-to-text captioning on the official HumanML3D
test population. All methods use the same 4,400 samples, three-reference TM2T
language protocol, candidate groups of 32 for semantic retrieval, and one
deterministic evaluation pass.

The all-case explorer exposes the input motion, all human references, and every
TM2T, MotionGPT, MotionGPT3, VerMo, and UniMuMo prediction for all 4,400
evaluated clips. The input is shown as an animated neutral SMPL mesh next to
all five caption outputs on one page.
Large motion chunks are loaded on demand from
[`ZeyuLing/Motius-Leaderboard-Cases`](https://huggingface.co/datasets/ZeyuLing/Motius-Leaderboard-Cases).
The metric table names both BERTScore scales explicitly: raw cosine similarity
and the English layer-17 baseline-rescaled score used by the TM2T protocol.

GT is reported as a reference row under the same protocol. It is excluded from
method rankings, best/second-best styling, and comparison charts.

The static Space consists of:

- `index.html`: responsive leaderboard and protocol layout.
- `leaderboard.js`: sorting, ranking, chart, and qualitative audit logic.
- `m2t_results.json`: versioned metric snapshot.
- `cases/`: all-case SMPL mesh viewer and manifest; binary assets live in the
  public dataset above.
