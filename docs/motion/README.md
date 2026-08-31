<!-- MOTIUS_DOCS_NAV:START -->
<p><a href="../../README.md"><strong>Motius</strong></a> / <a href="../README.md">Documentation</a> / <strong>Motion Toolkit</strong></p>
<p>
  <a href="../getting_started.md">Quickstart</a> ·
  <a href="../tasks/README.md">Tasks</a> ·
  <a href="../datasets/README.md">Datasets</a> ·
  <a href="../model_zoo/README.md">Models</a> ·
  <a href="../training/README.md">Training</a> ·
  <a href="../evaluator_zoo/README.md">Evaluators</a> ·
  <a href="../leaderboards/README.md">Benchmarks</a> ·
  <strong>Motion I/O</strong>
</p>
<!-- MOTIUS_DOCS_NAV:END -->

# Motion Representation Toolkit

Motius supports HumanML3D-263, AIST++ SMPL-24 joints, MotionStreamer-272,
HY-Motion-201, DART276, InterHuman-262, SMPL-22 `motion135`, and named Unitree
G1 layouts as motion representations. Instead of coupling every model to every
other model's tensor layout, Motius uses SMPL body motion as the shared
interchange layer:

```text
source representation -> SMPL-22 motion135 -> target representation
```

The bridge makes model-native outputs reusable across evaluators, renderers,
and pipelines while keeping each model's native representation intact. Start
with:

- [Representation reference](representations.md): channel layouts, coordinate
  frames, frame rates, and 6D rotation conventions.
- [Conversion guide](conversion.md): Python/CLI usage and the supported route
  matrix, including shape-aware SMPL to HumanML3D conversion.
- [Retargeting guide](retargeting.md): HML263 to SMPL, SMPL to SOMA, and SMPL
  to Unitree G1.
- [Representation-to-FBX export](fbx.md): export every public representation
  to a user-provided rigged character through the SMPL-22 bridge, using
  Autodesk FBX SDK without Blender or Blender as an optional backend.
- [Automatic character rigging](rigging.md): fit and skin an upright humanoid
  GLB/GLTF/FBX/OBJ/PLY/STL with a canonical SMPL-22 armature before retargeting.
- [Physical evaluation](../evaluation/physical_metrics.md): checkpoint-free
  Slide, Float, Jitter, Dynamic, and Penet metrics on canonical SMPL-22 joints.

The key rule is simple: conversion is not assumed to be lossless. Motius tells
you when a route drops shape, estimates twist with IK, changes coordinates, or
requires a particular skeleton. SOMA and Unitree G1 are retargeting targets,
not members of the body-representation interchange layer.
