---
title: Motion Reconstruction · HumanML3D
emoji: ♻️
colorFrom: green
colorTo: gray
sdk: static
pinned: false
license: mit
---

# Motion Reconstruction · HumanML3D

Reconstruction benchmark on the 4,042-motion HumanML3D test split. The public
table combines reconstruction FID and paired embedding distance, SMPL-22
geometry errors, and Motius physical diagnostics. GT is shown first as an
unranked calibration reference.

The embedded Three.js explorer compares GT and all nine released tokenizer or
autoencoder baselines over the same 4,042 paired test motions. This includes
T2M-GPT / MotionGPT, MoMask, MLD / MotionLCM, MoGenTS, MotionGPT3,
MotionStreamer, GoToZero / MotionMillion, PRISM, and VerMo.

The machine-readable result pack is [`reconstruction_results.json`](reconstruction_results.json).
