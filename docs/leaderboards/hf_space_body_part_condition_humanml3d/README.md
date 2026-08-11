---
title: Part-Level Motion Control · HumanML3D
emoji: "🎯"
colorFrom: blue
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Part-Level Motion Control · HumanML3D

Static leaderboard for body-part conditional motion generation on the
HumanML3D official test split. The page includes an embedded, synchronized
Three.js comparison over all 4,012 test cases for the left-wrist XYZ
sparse-control setting. The viewer and metric table contain the same five
protocol-compatible methods: OmniControl, CondMDI, MaskControl, MotionLab, and
MotionCanvas.

[ProjFlow](../../model_zoo/projflow.md) is integrated for Cartesian position
masks and has a verified public API/demo. It is intentionally absent from
rotation settings, which the method does not support. Its position rows will be
ranked only after the complete 4,012-case result packs are available; the
single release demo is not substituted for a benchmark score.
