# GEM-X / SOMA-77 attribution and license boundary

This adapter invokes, but does not vendor, NVIDIA's GEM-X runtime:

- Repository: <https://github.com/NVlabs/GEM-X>
- Pinned source revision: `32992550dba114c62243fb55e361311972dce8f9`
- Upstream source license: Apache License 2.0
- Pinned SOMA-X submodule revision:
  `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`

The official `gem_soma.ckpt` is obtained from `nvidia/GEM-X` at Hugging Face
revision `5ccf5ca3746c3620aa4016114f069a5f6ae399cd`. The associated model is
governed by the NVIDIA Open Model License Agreement. Its verified SHA-256 is
`4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e`.

The fixed upstream runtime also uses separately governed components including
SOMA-X, SAM 3D Body, YOLOX, ByteTrack, guided-diffusion, and PyTorch3D-derived
utilities. In particular, SAM 3D Body is subject to the SAM License rather than
Apache 2.0. Review the pinned upstream `ATTRIBUTIONS.md` and each submodule's
license before distribution or commercial deployment.

No upstream source, SOMA assets, or model weights are distributed here.
