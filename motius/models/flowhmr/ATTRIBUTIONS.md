# FlowHMR integration

This code originates from [FlowHMR](https://github.com/flowhmr/flowhmr),
revision `f12e6a2d46a63d771a66dbb6b5598a1be65b9eea`.
Copyright 2025–2026 FlowHMR authors. The source is restricted to educational,
research and non-profit use; see [UPSTREAM_LICENSE](vendor/UPSTREAM_LICENSE).
It is not covered by any more permissive license of the surrounding framework.

The official 491-D representation, MMDiT, flow matching, losses, data processing,
ODE/SDE sampling and postprocessing are vendored under `vendor/flowhmr`.
Motius changes namespace resolution, makes dataset body assets explicit, removes
checkout-dependent path injection, and adds native bundle/trainer/pipeline adapters.
The retired 349-D implementation is not shipped.

VGGT-Omega is vendored at `48b23c8ce72c9a8fcf987f0d7f43d8e4870759f6`,
with namespaced imports; see `vendor/VGGT_OMEGA_LICENSE`.
The video front end owns namespaced YOLOX, SAM-3D-Body and DINOv3 sources:
- SAM-3D-Body: `b5c765a0d89d789985e186d396315e7590887b94` (`vendor/SAM3D_LICENSE`).
- DINOv3: `6876159a11b4df116f30f667f8c9888617df0751` (`vendor/DINOV3_LICENSE.md`).
- YOLOX: `6ddff4824372906469a7fae2dc3206c7aa4bbaee` (`vendor/YOLOX_LICENSE`).
The migrated preprocessing helpers retain the Tencent HY-MOTION notices in
`vendor/HY_MOTION_LICENSE` and `vendor/HY_MOTION_NOTICE`. SMPL-H and its joint regressor remain user supplied. Exported frontend weights
retain the SAM, DINOv3, MHR, YOLOX and VGGT-Omega licenses. Published statistics are copied from FlowHMR.

No external source checkout is imported at runtime. Full training data are not
released upstream. Numerical adapter tests do not establish published benchmark
scores or completion of a full training reproduction.
