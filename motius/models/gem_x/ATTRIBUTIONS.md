# GEM-X / SOMA-77 attribution and license boundary

Motius includes the inference implementation and its public dependency closure:

- GEM-X: <https://github.com/NVlabs/GEM-X>, revision
  `32992550dba114c62243fb55e361311972dce8f9`, Apache-2.0.
- SOMA-X: <https://github.com/NVlabs/SOMA-X>, revision
  `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`, Apache-2.0.
- SAM 3D Body: <https://github.com/facebookresearch/sam-3d-body>, revision
  `b5c765a0d89d789985e186d396315e7590887b94`, SAM License.
- DINOv3: <https://github.com/facebookresearch/dinov3>, revision
  `6876159a11b4df116f30f667f8c9888617df0751`, DINOv3 License.

The complete licenses and attribution files are retained under `vendor/`.
Motius adds package markers, a local DINOv3 resolver, and an isolated
non-rendering runner. The network, preprocessing, decoding, and SOMA forward
implementations remain source-identical.

The complete artifact uses `gem_soma.ckpt` from `nvidia/GEM-X` at revision
`5ccf5ca3746c3620aa4016114f069a5f6ae399cd`. Its SHA-256 is
`4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e`.
The weights are governed by the NVIDIA Open Model License Agreement.

The artifact includes all weights and low-LOD SOMA assets needed by the video
pipeline. It does not clone or import another repository at runtime.
