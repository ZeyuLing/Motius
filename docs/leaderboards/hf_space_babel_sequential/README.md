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
| BABEL GT | 0.3947 | 0.5513 | 0.6327 | 0.0000 | 44.5941 | 0.0000 | 0.0000 |
| FlowMDM | 0.2958 | 0.4217 | 0.5018 | 0.0843 | 46.7698 | 0.1092 | 34.4040 |
| MotionStreamer | 0.2087 | 0.3136 | 0.3955 | 0.1205 | 49.3062 | 0.1664 | 76.2889 |
| MotionLab | 0.1242 | 0.2036 | 0.2647 | 1.3760 | 59.4449 | 1.4963 | 176.6814 |
| PRISM (epoch 12) | 0.4656 | 0.6448 | 0.7249 | 0.5205 | 41.4004 | 0.8037 | 138.7236 |

R-Precision uses official BABEL `act_cat` action-group multi-positive recall
batches of 32 (7,264 paired segments). The 7,285 intervals form 1,738 action
groups; synonymous labels are retained and never treated as false negatives.
Distribution metrics use all 7,285 segments. The encoder forward batch is 32
for the measured run and is independent of the recall candidate batch. GT is
excluded from ranking.

PRISM uses `checkpoint-epoch_12` with a fixed 360-frame canvas for every model
call; the complete 1,295-episode run contains no legacy 365-frame calls.
MotionLab uses its official five-frame autoregressive context. Both rows are
converted to the same canonical SMPL-22 joints before evaluation.

The Space also includes a synchronized Three.js neutral-SMPL-mesh comparison
of BABEL GT, FlowMDM, MotionStreamer, PRISM, and MotionLab. Every viewport preserves the
method's global XZ trajectory and exposes its first-frame facing direction. For
every displayed subclip it preserves the actual GT/FlowMDM Motion-to-Text and
Text-to-Motion Top-3 rankings from the same seed-0, 32-candidate recall batches.
