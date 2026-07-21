---
title: Music-to-Dance AIST++ Leaderboard
emoji: 🎵
colorFrom: green
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Music-to-Dance AIST++ Leaderboard

Static Motius leaderboard for music-conditioned dance generation on the
official 40-case AIST++ cross-modal evaluation package.

The table keeps the released Bailando `FID_k`, `FID_g`, diversity, and beat
protocol for paper comparability, and adds Motius normalized uTMR FID on
canonical 30 fps SMPL-22 joints. The qualitative explorer compares every GT
and generated clip as an aligned SMPL Mesh, synchronized to its official AIST
music clip. Every motion viewport supports drag-to-orbit, wheel zoom, and view
reset.

## Files

- `index.html`: benchmark table, metric comparison, protocol, and audit links.
- `leaderboard.js`: ranking, sorting, charts, and data rendering.
- `music_to_dance_results.json`: machine-readable evaluated results.
- `cases/`: all-case audio-synchronized Three.js SMPL Mesh comparison.
- `cases/audio_manifest.json`: official audio provenance, clip durations, and hashes.

GT is shown as a reference row and is excluded from best/second-best ranking.
Paper-reported values are parity targets, not leaderboard submissions.
