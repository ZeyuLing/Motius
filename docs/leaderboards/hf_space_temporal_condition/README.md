---
title: Temporal Condition Leaderboard
emoji: ⏱️
colorFrom: green
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Temporal Condition Leaderboard

Unified public leaderboard for temporal-conditioned human motion generation on
the HumanML3D official test split. The page exposes two protocols:

- **Temporal Control:** prediction, motion in-betweening, and adaptive sparse
  keyframe control, with text-on and text-off settings where applicable.
- **TP2M Prefix:** caption-guided continuation from 1, 5, or 9 observed motion
  frames, evaluated in MotionStreamer-272 space.

Every setting includes a visible **GT** reference row. GT semantic metrics are
reused from the matching T2M HumanML3D leaderboard rather than recomputed. For
Temporal Control, condition error and failure rates are zero by identity, and
foot skating is measured with one deterministic pass over the 4,012 temporal
test clips. GT is excluded from ranks, best/second highlighting, and chart
normalization.

The page supports task filtering, method search, sortable metrics, bar and radar
comparisons, and protocol-specific details. Ordered multi-prompt composition is
maintained separately on the BABEL Sequential Generation leaderboard.

The static Space is defined by:

- `index.html`: page structure and responsive styling.
- `leaderboard.js`: interaction, ranking, and chart logic.
- `temporal_control_results.json`: structured Temporal Control result snapshot.
