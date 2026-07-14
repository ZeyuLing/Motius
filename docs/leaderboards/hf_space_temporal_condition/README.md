---
title: Temporal Condition Leaderboard
emoji: ⏱️
colorFrom: cyan
colorTo: emerald
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
frames with the MotionStreamer-272 evaluator. Multi-prompt temporal composition
methods are tracked under the same leaderboard family because they test the
same temporal controllability axis.

Internal result paths, rendered examples, and debugging notes stay in the
repository documentation and experiment logs.
