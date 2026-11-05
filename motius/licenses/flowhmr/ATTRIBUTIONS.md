# FlowHMR integration

This code originates from [FlowHMR](https://github.com/flowhmr/flowhmr),
revision `f12e6a2d46a63d771a66dbb6b5598a1be65b9eea`.
Copyright 2025–2026 FlowHMR authors. The source is restricted to educational,
research and non-profit use; see [UPSTREAM_LICENSE](UPSTREAM_LICENSE).
It is not covered by any more permissive license of the surrounding framework.

The MMDiT architecture is implemented under `motius/models/flowhmr/network`.
Motion representations and SMPL-H geometry live under `motius/motion`; the
objective lives in `motius/trainers/flowhmr`, sampling and video processing in
`motius/pipelines/flowhmr`, and datasets in `motius/datasets/flowhmr_data`.
The native bundle holds the network and checkpoint buffers. Motius owns the
training and inference graphs and does not instantiate the upstream pipeline.
The retired 349-D implementation is not shipped.

VGGT-Omega network definitions originate from `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`,
with namespaced imports; see `VGGT_OMEGA_LICENSE`.
Native vision bundles use the following network sources, with data processing
and inference orchestration kept outside their model packages:
- SAM-3D-Body: `b5c765a0d89d789985e186d396315e7590887b94` (`SAM3D_LICENSE`).
- DINOv3: `6876159a11b4df116f30f667f8c9888617df0751` (`DINOV3_LICENSE.md`).
- YOLOX: `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (`YOLOX_LICENSE`).
The migrated preprocessing helpers retain the Tencent HY-MOTION notices in
`HY_MOTION_LICENSE` and `HY_MOTION_NOTICE`. SMPL-H and its joint regressor remain user supplied. Exported frontend weights
retain the SAM, DINOv3, MHR, YOLOX and VGGT-Omega licenses. Published statistics are copied from FlowHMR.

No external source checkout is imported at runtime. Full training data are not
released upstream. Numerical adapter tests do not establish published benchmark
scores or completion of a full training reproduction.

Fast SAM engineering integration follows
[Fast-SAM-3D-Body](https://github.com/yangtiming/Fast-SAM-3D-Body), revision
`d72aa36913a1673ace029d345f437561a85ec9e2` (MIT; `FAST_SAM_LICENSE`).
GPU image preprocessing and merged [left, right] hand batching are implemented
in `motius/pipelines/flowhmr/sam`, with original SAM networks/checkpoints retained.
Motius's GPU warp additionally reproduces OpenCV's fixed-point interpolation and
uint8 rounding rather than replacing reference pixels with smooth float samples.
Small affine metadata uses the original CPU geometry; RGB warp, flip, scaling
and normalization execute on GPU. No detector-based replacement hand boxes,
sparse camera frames or averaged hand features are used.
