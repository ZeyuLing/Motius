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

The complete licenses and attribution files are retained under `motius/licenses/gem_x/`.
Motius separates neural networks, video processing, body geometry, inference sampling,
and training objectives. The SOMA geometry dependency is pinned and kept only as
the required representation closure under motion/body_models/soma. Shared SAM3D
and DINOv3 neural networks and their licenses are retained under the native vision
modules and motius/licenses/flowhmr.

GEM-X network and video frontend certificates are independent acceptance gates.
SOMA's CUDA sparse multiplication is not bitwise deterministic even in the original
source; complete postprocessing parity must not be inferred from neural parity.
