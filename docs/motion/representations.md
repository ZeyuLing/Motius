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
| `interhuman262` | `(T, 2, 262)` | 30 | per person: global joints, global velocities, 21 local rotations, contacts | first two **columns** flattened row-wise |
| `g1_38` | `(T, 38)` | 30 | Unitree G1 root XY velocity/height, root rotation, 29 joint angles | first two **columns** flattened row-wise |
| `ardy_core330` | `(T, 330)` | 20 | Core-27 root, heading, positions, rotations, velocities, contacts | global rotations via ARDY `matrix_to_cont6d` |
| `ardy_g1_414` | `(T, 414)` | 25 | Unitree G1 explicit root, heading, positions, rotations, velocities, contacts | global rotations via ARDY `matrix_to_cont6d` |

## Core-330 And Unitree G1 Explicit 414D

Both ARDY release formats expose an explicit motion tensor while the model tokenizer
uses a hybrid explicit-root and latent-body representation internally:

```text
root XYZ | global heading (cos, sin) | root-local non-root joints
         | global joint rotations 6D | global joint velocities | contacts
```

The widths are 330 for Core-27 and 414 for Unitree G1. Checkpoint statistics contain
four additional local-root velocity/height channels used inside the tokenizer;
those make the stored statistics 334/418 wide but do not change public motion
tensor shapes.

Exact native decoding requires the `motion_rep` bundled with the checkpoint,
because it owns the skeleton, FPS, and normalization statistics:

```python
joints = convert_motion(
    features,
    "ardy_core330",
    "joints",
    motion_rep=pipe.bundle.motion_rep,
    is_normalized=True,
)
```

`ardy_g1_414` additionally converts exactly to MuJoCo qpos-36 for the Unitree
G1 robot. It is not a separate body model from Unitree G1; it is ARDY's
explicit tensor for that robot skeleton. Core-27 is not SMPL-22, so Motius
exposes named joint-position bridges in both directions:
`ardy_core27_to_smpl22_joints` and `smpl22_joints_to_ardy_core27_joints`.
They are suitable for viewers and joint-position evaluators, but they do not
recover SMPL twist, shape, or a valid `motion135` sequence.

## InterHuman-262

Each person contributes 262 channels:

```text
[0:66]    22 global SMPL-22 joint positions
[66:132]  22 global joint displacements
[132:258] 21 non-root local rotations in 6D
[258:262] left/right heel and toe contacts
```

A two-person clip has shape `(T, 2, 262)`. Both tracks must remain in the same
canonical world frame. Motius canonicalizes the pair with person 1's first
frame, then places person 2 with the official relative yaw and root offset;
canonicalizing each person independently would destroy the interaction.

The position channels decode exactly. InterHuman does not store root rotation,
body shape, or joint twist completely, so an SMPL mesh is recovered with the
documented position-IK bridge and is necessarily non-unique.

The paired representation can be encoded from SMPL-22 joint tracks or paired
`motion135`:

```python
from motius.motion import convert_motion, motion135_to_interhuman262

joints_pair = convert_motion(motion_interhuman, "interhuman262", "joints")
motion_interhuman = motion135_to_interhuman262(
    motion135_pair,                 # (T, 2, 135)
    bone_offsets=smpl22_offsets,    # (22, 3), same FK skeleton as motion135
    source_coordinates="y_up",
)
```

Going back to SMPL is intentionally split into two levels:

```text
InterHuman-262 -> exact SMPL-22 joint positions
InterHuman-262 -> position-IK -> approximate SMPL motion135 / mesh
```

The first route is deterministic and evaluator-safe. The second route is for
mesh previews and must report the IK fit error.

## Same-Motion Visual Comparison

All five panels below use HumanML3D test case `004822`: *A person walks
forward at an average pace, swaying their arms and torso with swagger.* This
keeps the source motion fixed while changing only the representation and target
body.

![HumanML3D-263, SMPL motion135, SOMA-30, Core-27, and Unitree G1](../../assets/motion/representation_demo/004822_hml_smpl_soma_core_g1.gif)

The synchronized [Three.js viewer](../../assets/motion/representation_demo/index.html)
uses the following routes:

```text
HumanML3D-263 -> official joint decode -> SMPL-22 joints
SMPL motion135 -> SMPL-H skinning -> animated SMPL surface mesh
SMPL motion135 -> SOMA30 rotation transfer -> SOMA77 LBS mesh
SMPL motion135 global rotations -> Core-27 visual rotation bridge -> Core LBS mesh
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

## Two-Person InterHuman Preview

The InterHuman preview is a representation demo, not a model-generation demo.
It uses InterX clip `G021T002A012R014` in both panels: one person steps forward
and points while the other leans back. This avoids precision-contact actions
such as high-fives or hand holding. The same motion is converted to paired
InterHuman skeletons on the left and rendered from the original GT SMPL-H pose
on the right. The shared canonical frame is preserved for both people.

![GT InterX to InterHuman skeleton and SMPL mesh representation comparison](../../assets/motion/interhuman_representation_demo/interx_smplh_gt_G021T002A012R014_skeleton_smpl_mesh.gif)

[Open the synchronized Three.js viewer](../../assets/motion/interhuman_representation_demo/index.html).

The builder reads InterX `smplh_52_2p/P1` and `P2` GT arrays, including
`raw_betas`/`betas` and `gender` when available, extracts SMPL-22 joints,
converts them with `joints_pair_to_interhuman262`, decodes exact
`InterHuman-262` joint positions, and renders the original GT body pose for the
SMPL mesh preview. It also writes centered `data.js`, `smpl_pair_vertices.u16`,
`smpl_pair_normals.i8`, and `smpl_indices.u32` for the browser viewer.

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
