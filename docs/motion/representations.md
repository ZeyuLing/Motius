# Motion Representations

Static metadata is available from `motius.motion.representation.SPECS`.

| Name | Shape | Native FPS | Layout | 6D rotation convention |
| ---- | ----: | ---------: | ------ | ---------------------- |
| `hml263` | `(T, 263)` | 20 | root velocity/height, RIC joints, local rotations, velocities, contacts | HumanML3D feature protocol |
| `ms272` | `(T, 272)` | 30 | root velocity, heading delta, joints, velocities, local rotations | first two **rows**, `R[:2, :].reshape(6)` |
| `motion135` | `(T, 135)` | usually 30 | root translation + 22 local rotations | first two **columns** flattened row-wise, `R[:, :2].reshape(6)` |
| `hymotion201` | `(T, 201)` | 30 | `motion135` + 22 pelvis-relative joints | same as `motion135` |
| `dart276` | `(T, 276)` | 20 | pose, joints, velocities, root orientation/translation | first two **columns** flattened row-wise |
| `g1_38` | `(T, 38)` | 30 | root XY velocity/height, root rotation, 29 joint angles | first two **columns** flattened row-wise |

## The Two 6D Layouts

Two layouts in this repository were historically both described as
"row-major". They are not interchangeable:

```text
motion135 / HY-Motion / DART: [R00, R01, R10, R11, R20, R21]
MS272:                        [R00, R01, R02, R10, R11, R12]
```

Use the format-specific converter. Do not reshape or permute by intuition.
`motius.motion.representation.rotation` uses `convention="row"` for the first
layout because it is `R[:, :2].reshape(6)`. MS272 has a dedicated decoder.

## HY-Motion-201

```text
[0:3]     absolute root translation
[3:9]     root/global rotation 6D
[9:135]   21 body local rotations 6D
[135:201] 22 pelvis-relative joint positions
```

The pelvis position at `[135:138]` is zero by construction. The 198-dimensional
training variant sometimes used inside Motius removes only this redundant
triplet; it is not a second public HY-Motion representation.

## DART276 Length And Coordinates

DART stores velocity targets, so encoding a `T`-frame SMPL clip produces
`T-1` DART frames. `equal_length=True` reconstructs the final frame while
decoding. Native DART is Z-up. The default DART-to-motion135 bridge applies the
released MBench coordinate transform and floor alignment.

## Frame Rate

Most representation converters preserve the input frame count and sampling.
They do not silently resample. The exception is the HML263-to-SMPL retargeter,
whose defaults explicitly resample 20 fps input to 30 fps output. Track FPS in
your dataset metadata whenever a target representation can be used at multiple
rates.
