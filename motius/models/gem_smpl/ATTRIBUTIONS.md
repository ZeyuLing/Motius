# GEM-SMPL attribution and license boundary

This adapter invokes, but does not vendor, NVIDIA's GEM-SMPL runtime (the
project formerly named GENMO):

- Repository: <https://github.com/NVlabs/GENMO>
- Pinned source revision: `16bebf402d8893184249ee206d957b8248cd8310`
- Upstream source license: NVIDIA OneWay Noncommercial License
- Permitted upstream use: research or evaluation only; redistribution must
  retain the complete upstream license and notices.

The official `gem_smpl.ckpt` is obtained from `nvidia/GEM-X` at Hugging Face
revision `5ccf5ca3746c3620aa4016114f069a5f6ae399cd`. The model card governs model
use under the NVIDIA Open Model License Agreement. Its verified SHA-256 is
`1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`.

Runtime-only third-party dependencies remain governed by their own terms.
Notably, SMPL-X model files require separate registration and licensing from
the Max Planck Institute, while HMR2 and ViTPose checkpoints have their own
distribution terms. See the pinned upstream `LICENSE`, `ATTRIBUTIONS.md`, and
`docs/INSTALL.md` before downloading or using these assets.

No upstream source or model weights are distributed in this directory.
