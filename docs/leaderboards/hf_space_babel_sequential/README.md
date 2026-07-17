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
transition distribution metrics use the Motius Joint-Position Evaluator. FID
is measured only after per-sample L2 normalization of the uTMR motion
embeddings; Peak Jerk and AUJ are computed directly from paired episode trajectories.
GT/reference rows are shown for calibration and excluded from generated-method
ranking. Results from the superseded 64-composition protocol are not carried
into this leaderboard.

| Method | R@1 | R@2 | R@3 | Normalized FID | MM-Dist | Normalized Transition FID | AUJ Gap |
| ------ | --: | --: | --: | --: | ------: | -------------: | ------: |
| BABEL GT | 0.3614 | 0.5284 | 0.6317 | 0.0000 | 47.8378 | 0.0000 | 0.0000 |
| FlowMDM | 0.2504 | 0.3925 | 0.4818 | 0.0467 | 50.8503 | 0.0555 | 34.4040 |
| MotionStreamer | 0.2130 | 0.3303 | 0.4175 | 0.0610 | 52.0339 | 0.0702 | 76.2889 |
| MotionLab | 0.2580 | 0.3793 | 0.4536 | 0.2011 | 51.3873 | 0.2499 | 25.7259 |
| PRISM (epoch 14) | 0.2453 | 0.3716 | 0.4555 | 0.0574 | 51.8148 | 0.0732 | 106.4423 |

R-Precision uses official BABEL `act_cat` action-group multi-positive recall
batches of 32 (7,264 paired segments). The 7,285 intervals form 1,738 action
groups; synonymous labels are retained and never treated as false negatives.
Distribution metrics use all 7,285 segments. Encoder forward batch size is a
throughput setting independent of the 32-sample recall candidate batch. GT is
excluded from ranking.

PRISM uses `checkpoint-epoch_14` with a fixed 360-frame canvas for every model
call; the complete 1,295-episode run contains no legacy 365-frame calls.
MotionLab uses its official five-frame autoregressive context. Both rows are
converted to the same canonical SMPL-22 joints before evaluation.

The Space also includes a synchronized Three.js neutral-SMPL-mesh comparison
of BABEL GT, FlowMDM, MotionStreamer, PRISM, and MotionLab. PRISM and MotionLab
use their native or fitted SMPL parameters so the viewer does not invent head
or terminal-joint twists from positions alone. Every viewport preserves the
method's global XZ trajectory and exposes its current body-facing direction,
estimated from the hips and shoulders. The cyan arrow is not root velocity, so
backward and sideways actions need not align with it. For
every displayed subclip it preserves the actual GT/FlowMDM Motion-to-Text and
Text-to-Motion Top-3 rankings from the same seed-0, 32-candidate recall batches.
