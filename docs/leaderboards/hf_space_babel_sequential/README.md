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
processed official BABEL validation split: 1,295 episodes, 7,285 LLM-rewritten
action intervals, and 5,990 transition boundaries. Explicit transition labels
are cut at their midpoint, and adjacent short actions are merged to at least 30
frames before caption rewriting.

Every method is converted to neutral zero-beta SMPL-22 joints. Semantic and
transition distribution metrics use the Motius Joint-Position Evaluator; Peak
Jerk and AUJ are computed directly from paired episode trajectories.
GT/reference rows are shown for calibration and excluded from generated-method
ranking. Results from the superseded 64-composition protocol are not carried
into this leaderboard.

| Method | R@1 | R@2 | R@3 | FID | MM-Dist | Transition FID | AUJ Gap |
| ------ | --: | --: | --: | --: | ------: | -------------: | ------: |
| BABEL GT | 0.2939 | 0.4330 | 0.5193 | 0.0000 | 46.5581 | 0.0000 | 0.0000 |
| FlowMDM | 0.1032 | 0.1839 | 0.2496 | 2479.8395 | 57.3074 | 2629.5964 | 55.8724 |

R-Precision uses recall batches of 32 (7,264 paired segments). Distribution
metrics use all 7,285 segments. GT is excluded from ranking.
