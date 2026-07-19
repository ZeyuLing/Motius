---
title: T2M HumanML3D Leaderboard
emoji: 🏃
colorFrom: green
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
---

# T2M HumanML3D Leaderboard

Static leaderboard page for HumanML3D official-test text-to-motion evaluation.

The public page contains the benchmark protocol, generated-method comparison
charts, metric tables, and an all-case Three.js comparison covering all 4,042
selected-caption test cases and every released result row with retained output.
The comparison decodes each method's `motion_135` channels and animates a
neutral SMPL body mesh with linear blend skinning; it is not a skeleton-only
preview.
Motion assets are fetched lazily from the public
[`ZeyuLing/Motius-Leaderboard-Cases`](https://huggingface.co/datasets/ZeyuLing/Motius-Leaderboard-Cases)
dataset so the Space itself remains lightweight.
GT, paper-only, and explicitly marked calibration rows are excluded from all
generated-method rankings. Internal result paths and debugging notes stay in
the repository documentation and experiment logs.
