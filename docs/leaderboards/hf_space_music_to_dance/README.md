---
title: Music-to-Dance · AIST++
emoji: 🎵
colorFrom: green
colorTo: red
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Music-to-Dance · AIST++

Static Motius leaderboard for music-conditioned dance generation on the
official 40-case AIST++ cross-modal evaluation package.

The table evaluates Bailando, EDGE, TM2D, and UniMuMo with the released
Bailando `FID_k`, `FID_g`, diversity, and beat protocol, and adds Motius
normalized uTMR FID on canonical 30 fps SMPL-22 joints. The qualitative
explorer compares GT and all four methods across all 40 cases. Each scene
overlays the method's native AIST++ SMPL-24 skeleton on the corresponding SMPL
Mesh and synchronizes playback to the official AIST music. Joint-only outputs
expose position-IK MPJPE; EDGE's rotation-native mesh decode does not use IK.
Every viewport supports drag-to-orbit, wheel zoom, and view reset.

## Files

- `index.html`: benchmark table, metric comparison, protocol, and audit links.
- `leaderboard.js`: ranking, sorting, charts, and data rendering.
- `music_to_dance_results.json`: machine-readable evaluated results.
- `cases/`: all-case audio-synchronized native-skeleton and SMPL-Mesh comparison.
- `cases/audio_manifest.json`: official audio provenance, clip durations, and hashes.

GT is shown as a reference row and is excluded from best/second-best ranking.
Paper-reported values are parity targets, not leaderboard submissions.
