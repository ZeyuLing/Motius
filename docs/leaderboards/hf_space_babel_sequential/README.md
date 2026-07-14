---
title: BABEL Sequential Generation Leaderboard
emoji: 🎬
colorFrom: green
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
---

# BABEL Sequential Generation Leaderboard

Static leaderboard for text-guided sequential human motion generation on the
FlowMDM BABEL validation protocol: 64 compositions, 32 prompts per composition,
and 2,048 generated action segments.

Every method is converted to canonical SMPL-22 joints. Semantic and transition
distribution metrics use the Motius Joint-Position Evaluator; Peak Jerk and AUJ
are computed directly from the same joint trajectories. GT/reference rows are
shown for calibration and excluded from generated-method ranking.
