# GEM-SMPL attribution and license boundary

Motius includes the inference implementation from NVIDIA's GEM-SMPL project,
formerly named GENMO:

- Repository: <https://github.com/NVlabs/GENMO>
- Pinned source revision: `16bebf402d8893184249ee206d957b8248cd8310`
- Source license: NVIDIA OneWay Noncommercial License
- Allowed use: noncommercial research and evaluation

The complete upstream license is retained as
`vendor/GENMO_LICENSE`, and upstream third-party notices are retained as
`vendor/GENMO_ATTRIBUTIONS.md`. Motius adds package markers and an isolated
runner; the numerical inference implementation is otherwise source-identical.

The complete Motius artifact uses `gem_smpl.ckpt` from `nvidia/GEM-X` at
revision `5ccf5ca3746c3620aa4016114f069a5f6ae399cd`. Its SHA-256 is
`1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`.
The weights are governed by the NVIDIA Open Model License Agreement.

The artifact also retains the exact official HMR2, ViTPose, and YOLO assets.
Six small body regressors required by GENMO but omitted from its source archive
are retained byte-for-byte from GVHMR revision
`6ec3ca39336c50492c0fae65fba2fb831fc7d866`; its noncommercial license is
retained as `vendor/GVHMR_LICENSE`.
SMPL and SMPL-X body-model files are not redistributed; users must obtain them
from the official Max Planck Institute sites and pass
`body_models_root="checkpoints/body_models"`.
