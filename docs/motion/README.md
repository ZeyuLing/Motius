# Motion Representation Toolkit

Motius supports HumanML3D-263, MotionStreamer-272, HY-Motion-201, DART276, and
SMPL-22 `motion135` as first-class motion representations. Instead of coupling
every model to every other model's tensor layout, Motius uses SMPL body motion
as the shared interchange layer:

```text
source representation -> SMPL-22 motion135 -> target representation
```

The bridge makes model-native outputs reusable across evaluators, renderers,
and pipelines while keeping each model's native representation intact. Start
with:

- [Representation reference](representations.md): channel layouts, coordinate
  frames, frame rates, and 6D rotation conventions.
- [Conversion guide](conversion.md): Python/CLI usage and the supported route
  matrix.
- [Retargeting guide](retargeting.md): HML263 to SMPL, SMPL to SOMA, and SMPL
  to Unitree G1.

The key rule is simple: conversion is not assumed to be lossless. Motius tells
you when a route drops shape, estimates twist with IK, changes coordinates, or
requires a particular skeleton. SOMA and Unitree G1 are retargeting targets,
not members of the body-representation interchange layer.
