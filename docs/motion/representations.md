# Motion Representations

Static metadata is available from `motius.motion.representation.SPECS`.

## SMPL-22 Naming

`SMPL-22` is the canonical pelvis-to-wrist body skeleton used by Motius: the
root plus the first 21 articulated body joints. These joint names, their order,
and their kinematic hierarchy are shared by SMPL and SMPL-H. Motius therefore
uses `SMPL-22`, rather than `SMPL-H`, as the representation name. A specific
SMPL-H model may still be named when documenting the implementation used to
materialize joints or meshes.

| Name | Shape | Native FPS | Layout | 6D rotation convention |
| ---- | ----: | ---------: | ------ | ---------------------- |
| `hml263` | `(T, 263)` | 20 | root velocity/height, RIC joints, local rotations, velocities, contacts | HumanML3D feature protocol |
| `ms272` | `(T, 272)` | 30 | root velocity, heading delta, joints, velocities, local rotations | first two **rows**, `R[:2, :].reshape(6)` |
| `motion135` | `(T, 135)` | usually 30 | root translation + 22 local rotations | first two **columns** flattened row-wise, `R[:, :2].reshape(6)` |
| `hymotion201` | `(T, 201)` | 30 | `motion135` + 22 pelvis-relative joints | same as `motion135` |
| `dart276` | `(T, 276)` | 20 | pose, joints, velocities, root orientation/translation | first two **columns** flattened row-wise |
| `g1_38` | `(T, 38)` | 30 | root XY velocity/height, root rotation, 29 joint angles | first two **columns** flattened row-wise |

## Same-Motion Visual Comparison

All three panels below use HumanML3D test case `004822`: *A person walks
forward at an average pace, swaying their arms and torso with swagger.* This
keeps the source motion fixed while changing only the representation and target
body.

![HumanML3D-263, SMPL motion135, and Unitree G1-38D](../../assets/motion/representation_demo/004822_hml_smpl_g1.gif)

The synchronized [Three.js viewer](../../assets/motion/representation_demo/index.html)
uses the following routes:

```text
HumanML3D-263 -> official joint decode -> SMPL-22 joints
SMPL motion135 -> SMPL-H skinning -> animated SMPL surface mesh
SMPL motion135 -> GMR inverse kinematics -> G1 qpos -> MuJoCo visual meshes
```

The display recenters each body at its initial ground position and aligns the
anatomical body-forward direction to viewer `+z`. It preserves the body
proportions and articulated motion produced by each route; it does not claim
that G1 retargeting is lossless.

GMR's coordinate bridge is explicit: SMPL Y-up `[x, y, z]` becomes MuJoCo
Z-up `[z, x, y]`, so SMPL `+z` body-forward becomes G1 `+x` body-forward.
`GMR_Z_UP_FROM_Y_UP` and its inverse `GMR_Y_UP_FROM_Z_UP` are exported from
`motius.motion.retarget`; renderers should use these matrices instead of
assuming a generic Z-up axis permutation.

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
The HML263-to-SMPL retargeter and SMPL-to-HML263 encoder expose `src_fps` and
`dst_fps` because the protocols use different native rates. Integer downsampling
uses phase-aligned striding by default; other ratios use linear joint
interpolation. Track FPS and crop phase in dataset metadata whenever a target
representation can be used at multiple rates.
