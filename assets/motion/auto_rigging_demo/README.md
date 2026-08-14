# Public Auto-Rigging Demo

This directory contains a reproducible structural smoke test for Motius's
static-mesh rigging path. The input is Blender Studio's realistic male from
Human Base Meshes v1.0.0, released under CC0:

```text
https://download.blender.org/demo/bundles/bundles-3.6/human-base-meshes-bundle-v1.0.0.zip
SHA-256: 46a912c0524072ac3b78c35d5d2471df7b8df102394a050ca8cd7184e3393648
```

The source bundle and intermediate GLB/FBX files are reproducibly downloaded
or generated and are not committed. `manifest.json` pins the source, motion,
pipeline, media checksums, and individual validation reports. The final media
are:

- `blender_cc0_male_autorig_004822_640_30fps.mp4`: full 150-frame Blender
  render at 30 fps;
- `blender_cc0_male_autorig_004822_readme.gif`: compact README preview;
- `blender_cc0_male_autorig_visual_qa.png`: rest-skeleton and dominant-weight
  views plus first, middle, worst-deformation, and last poses from front and
  side;

Rebuild everything from the repository root with:

```bash
python tools/build_auto_rigging_demo.py --blender /path/to/blender
```

The build uses the repository's persisted HumanML3D motion `004822`. It
validates the input as unrigged, validates the output skin and canonical
SMPL-22 armature, requires an imported animation Action and measured mesh
deformation, checks source-to-target bone directions and robust edge stretch
on every animation frame, and renders selected front/side pose diagnostics. It
only publishes the media after all gates pass.

These checks establish that the public mesh was unrigged, received a canonical
armature and normalized skin weights, and visibly deforms under the stored
motion. They do not establish production-quality anatomical landmarks, bone
roll, foot contact, or head gaze. The demo retargeter consumes SMPL-22 joint
positions: its bone-direction checks cannot observe twist around a bone, an
ankle-to-foot vector does not define a sole plane, and a neck-to-head vector
does not define a facial direction. Consult
[`docs/motion/rigging.md`](../../../docs/motion/rigging.md) before treating the
output as an accepted character asset.
