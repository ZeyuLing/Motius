---
title: Motion Repair · BrokenAMASS
emoji: 🧩
colorFrom: green
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Motion Repair · BrokenAMASS

This directory is the source for the public Motion Repair leaderboard. It
compares MoGenDiT, StableMotion, and MotionCanvas on the 299-case
BrokenAMASS Z-up v2 pair-validated split.

- Primary ordering: Motius uTMR R@3.
- Visual population: all 299 cases.
- Viewer: synchronized Three.js SMPL meshes for Clean GT, corrupted input,
  MoGenDiT, StableMotion, and MotionCanvas.
- Support tracks: MoGenDiT method-native adaptive support; StableMotion and
  MotionCanvas oracle-v6 support.

The machine-readable results are in
[`motion_repair_results.json`](motion_repair_results.json). Rebuild the
qualitative assets with
[`tools/build_motion_repair_gallery.py`](../../../tools/build_motion_repair_gallery.py).
