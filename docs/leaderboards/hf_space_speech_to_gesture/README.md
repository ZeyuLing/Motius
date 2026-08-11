---
title: Speech-to-Gesture · BEAT2
emoji: 🗣️
colorFrom: yellow
colorTo: green
sdk: static
app_file: index.html
pinned: false
license: mit
---

# Speech-to-Gesture · BEAT2

Verified-only Motius leaderboard for the fixed BEAT2 English speaker-2 test
protocol. Every ranked row is generated and scored inside Motius; paper-only
numbers are not inserted as submissions.

[Open the public leaderboard](https://huggingface.co/spaces/ZeyuLing/speech-to-gesture-beat2-leaderboard).

The public LoM checkpoint is explicitly marked as visualization-only. Its
measured metrics remain useful for adapter verification, but are not presented
as a reproduction of the paper's private/full evaluation checkpoint.
EMAGE uses the maintained official PantoMatrix audio checkpoint pinned by
revision. On the same 15 paired clips it measures `FGD = 6.199`, `BC = 7.564`,
and `Diversity = 12.476`; its paper values remain a separate reference because
the paper model also consumed text.

GT is displayed as a reference and is excluded from ranking.

The primary visualization uses the same high-resolution Three.js SMPL browser
as the other Motius motion leaderboards, with synchronized audio, timeline
scrubbing, camera controls, and NPZ/FBX export. The browser uses the shared
SMPL-22 projection for direct cross-task visual consistency; the original
55-joint, expression, translation, and shape arrays remain downloadable as NPZ.

The static source in this directory is combined with generated all-case assets
only in an `outputs/` staging directory:

```bash
python tools/publish_speech_to_gesture_hf.py
```
