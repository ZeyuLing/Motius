---
title: Motius Dance-to-Music Leaderboard
emoji: 🎧
colorFrom: green
colorTo: red
sdk: static
pinned: false
---

# Motius Dance-to-Music Leaderboard

Static D2M-GAN AIST++ dance-to-music benchmark and synchronized SMPL/audio
case viewer.

- `index.html`: leaderboard and protocol summary.
- `cases/index.html`: all-case interactive SMPL and audio comparison.
- `dance_to_music_results.json`: machine-readable aggregate results.
- `cases/manifest.json`: case-level motion, audio, and beat metrics.

The leaderboard uses the public 86-segment, two-second D2M-GAN protocol. The
paper's `Beats Coverage` is displayed as **Beat Count Ratio (target 100%)**:
it is generated beat bins divided by reference beat bins and is not bounded by
100%. `Beats Hit` remains a higher-is-better metric.
