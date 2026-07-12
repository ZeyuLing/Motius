# Motion Toolkit

Motius keeps model-native motion arrays intact and exposes explicit conversion
and retargeting steps around them. Start with:

- [Representation reference](representations.md): channel layouts, coordinate
  frames, frame rates, and 6D rotation conventions.
- [Conversion guide](conversion.md): Python/CLI usage and the supported route
  matrix.
- [Retargeting guide](retargeting.md): HML263 to SMPL, SMPL to SOMA, and SMPL
  to Unitree G1.

The key rule is simple: conversion is not assumed to be lossless. Motius tells
you when a route drops shape, estimates twist with IK, changes coordinates, or
requires a particular skeleton.
