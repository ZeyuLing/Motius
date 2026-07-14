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

Static leaderboard page for temporal-conditioned human motion generation.

The current public protocol covers HumanML3D TP2M prefix-conditioned generation:
given the text prompt and the first `N` ground-truth motion frames, a method
generates the continuation. Results are reported for 1, 5, and 9 condition
frames. The current measured snapshot uses the MotionStreamer-272 evaluator.
Multi-prompt temporal composition is maintained separately on the BABEL
Sequential Generation leaderboard, where all methods use the Motius
Joint-Position Evaluator.

The page is driven by a single structured result set and provides per-condition
filtering, method search, sortable metrics, generated-method ranking highlights,
bar and radar comparisons, protocol details, and navigation to the T2M
HumanML3D and BABEL Sequential Generation leaderboards. Reference, paper-only,
and explicitly excluded rows do
not participate in rankings.

Internal result paths, rendered examples, and debugging notes stay in the
repository documentation and experiment logs.
