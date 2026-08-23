# Multi-Character Auto-Rigging Demo

This directory contains the approved Motius AutoRig demo: three downloaded,
textured, unrigged character meshes with different proportions are bound to
animation-ready skeletons, normalized to Motius's canonical SMPL22 subset, and
driven by the same HumanML3D motion.

## Media

- `motius_multi_character_autorig_004822_960x540_30fps.mp4`: complete
  150-frame, 30 fps synchronized video;
- `motius_multi_character_autorig_004822_800x450_20fps.gif`: optimized
  100-frame, 20 fps looping README preview linked to the full MP4;
- `motius_multi_character_autorig_004822_poster.jpg`: frame 103 reference
  poster;
- `manifest.json`: sources, licenses, pipeline provenance, and artifact hashes;
- `validation.json`: input, rig, texture, deformation, and motion summaries;
- `render.json`: renderer, motion, presentation, and skeleton-overlay contract.

The video contains no text overlays. White joints and cyan limbs visualize the
generated SMPL22-compatible skeletons in motion; they are not reference rigs.

## Character Attribution

| Role | Creator and model | License | Source |
| --- | --- | --- | --- |
| Child | CG-Moon, Little Girl | CC BY 4.0 | `https://sketchfab.com/3d-models/little-girl-rigged-3d-model-2b5c9b749a714dd6b0d04c7de83c254e` |
| Big head | Suushimi, Running boy | CC BY-NC 4.0 | `https://sketchfab.com/3d-models/running-boy-0c80a9edd3514af0902349233d2c8d8f` |
| High weight | kornasale, Yōkai Project: Oni | CC BY 4.0 | `https://sketchfab.com/3d-models/yokai-project-oni-62f0e50b3f3543febaddd4834b05a84d` |

The source GLBs are not redistributed. The committed demo media contains
third-party character content and retains the corresponding attribution and
non-commercial restriction; it is not automatically covered by the license of
Motius's source code.

## Pipeline Provenance

Skeleton and skin inference use the upstream Make-It-Animatable backend:

```text
https://github.com/jasongzy/Make-It-Animatable
https://huggingface.co/spaces/jasongzy/Make-It-Animatable
```

Motius performs the public API integration, Mixamo-compatible-to-SMPL22 bone
normalization, texture-preserving export, motion retargeting, foot/head
orientation priors, validation, skeleton visualization, and final Blender
render. This provenance distinction is intentional: the diverse-character
media is not presented as output of Motius's simpler local `template` fitter.

See [`docs/motion/rigging.md`](../../../docs/motion/rigging.md) for commands,
privacy notes, method limits, and the separate fully local template backend.
