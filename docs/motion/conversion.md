# Motion Conversion

Motius composes cross-representation conversion around SMPL-22 `motion135`:

```text
source representation -> SMPL-22 motion135 -> target representation
```

For example, an HY-Motion-201 sequence can expose its exact `motion135` prefix
and then be encoded for the MotionStreamer-272 evaluator. A HumanML3D-263
sequence first requires IK because its position features do not uniquely retain
SMPL joint rotations. The supported route table below records these fidelity
differences explicitly.

## Python API

```python
import numpy as np
from motius.motion import convert_motion, smpl_to_humanml263

# Exact native position decode.
joints = convert_motion(motion_hml263, "hml263", "joints")
joints = convert_motion(motion_ms272, "ms272", "joints")
joints_pair = convert_motion(motion_interhuman, "interhuman262", "joints")
smpl22_from_ardy = convert_motion(
    ardy_features,
    "ardy_330",
    "smpl22_joints",
    motion_rep=ardy_pipe.bundle.motion_rep,
    is_normalized=True,
)

# HY-Motion's transform channels are an exact prefix.
motion135 = convert_motion(motion_hy201, "hymotion201", "motion135")

# FK routes require the skeleton used by the target protocol.
offsets = np.load("path/to/smpl22_bone_offsets.npy")
motion201 = convert_motion(
    motion135,
    "motion135",
    "hymotion201",
    bone_offsets=offsets,
)

# A complete API-level SMPL-H -> HumanML3D conversion. First follow the
# body-model setup in README.md; no evaluation script or external repo is used.
motion_hml263 = smpl_to_humanml263(
    global_orient,
    body_pose,
    transl,
    betas=betas,
    gender="female",
    model_type="smplh",
    model_path="checkpoints/body_models",
    src_fps=20,
    coordinate_system="amass",  # AMASS Z-up -> HumanML3D Y-up
)
```

Download and install the licensed model parameters using the
[SMPL body-model setup](../../README.md#smpl-body-model-setup). The API accepts
either the documented directory root or a direct `.npz`/`.pkl` model file.

Lower-level functions live in:

- `motius.motion.representation.humanml`
- `motius.motion.representation.motion272`
- `motius.motion.representation.hymotion`
- `motius.motion.representation.dart276`
- `motius.motion.representation.rotation`

## CLI

```bash
python tools/convert_motion.py input.npy output.npy \
  --src hml263 --dst joints

python tools/convert_motion.py motion135.npy motion201.npz \
  --src motion135 --dst hymotion201 \
  --bone-offsets assets/my_smpl22_offsets.npy

python tools/convert_motion.py dart276.npy motion135.npy \
  --src dart276 --dst motion135

python tools/convert_motion.py amass_clip.npz hml263.npy \
  --src smpl --dst hml263 \
  --smpl-model-dir checkpoints/body_models --model-type smplh --gender female \
  --src-fps 120 --dst-fps 20 --coordinate-system amass
```

For an input NPZ with multiple arrays, add `--input-key motion`. NPZ outputs
store the converted array under `motion` and record source/target names.

## Supported Routes

| Source | Target | Semantics |
| ------ | ------ | --------- |
| HML263 | joints | Native RIC decode; deterministic |
| HML263 | motion135 | SMPL IK; lossy; needs SMPL assets |
| SMPL parameters | joints | Shape-aware FK; uses explicit beta, gender, and model file |
| SMPL parameters | HML263 | Shape-aware FK followed by the official HumanML3D protocol |
| joints | HML263 | Official skeleton retargeting, canonicalization, IK, velocities, and contacts |
| MS272 | joints | Native stored-position decode |
| MS272 | motion135 | Recovers root and rotations; subject shape is unavailable |
| HY-Motion-201 | motion135 | Exact prefix extraction |
| HY-Motion-201 | joints | Uses explicit stored RIC joints by default |
| motion135 | joints | FK; requires explicit `(22,3)` offsets |
| motion135 | HY-Motion-201 | FK appends 22 RIC joints; requires offsets |
| motion135 | MS272 | Uses the bundled canonical MS272 evaluator skeleton by default |
| motion135 | HML263 | FK plus official HumanML3D encoding; requires explicit SMPL-22 offsets |
| DART276 | joints | Native DART decode, optional MBench coordinate conversion |
| DART276 | motion135 | Coordinate/floor bridge; retains DART temporal sampling |
| InterHuman-262 | joints | Exact stored global-position decode; preserves the shared pair frame |
| paired joints | InterHuman-262 | Official pair-aware canonicalization; requires 21 non-root local rotations |
| paired motion135 | InterHuman-262 | FK plus pair-aware encoding; requires explicit SMPL-22 offsets |
| ARDY-330 | ARDY-27 joints | Exact native ARDY-27 decode; requires the checkpoint `motion_rep` |
| ARDY-27 joints | SMPL-22 joints | Named joint-position bridge; not an SMPL pose or mesh recovery |
| SMPL-22 joints | ARDY-27 joints | Named joint-position bridge for ARDY skeleton viewers and joint evaluators |
| Unitree G1 explicit 414D | G1 joints | Exact native Unitree G1 decode; requires the checkpoint `motion_rep` |
| Unitree G1 explicit 414D | G1 qpos-36 | Exact MuJoCo root pose plus 29-DOF export |
| G1-38 | G1 qpos-36 | Exact root quaternion + 29-DOF decode |
| G1 qpos-36 | G1-38 | Optional root canonicalization and XY velocity encoding |

Routes can compose through `motion135`, for example DART276 to MS272. HML263
to MS272 first performs the lossy SMPL IK step.

InterHuman-262 intentionally does not expose a direct exact `motion135` decode:
the representation omits root rotation and does not uniquely determine twist.
Use its exact joint decode followed by `retarget_hml263_clip(...,
rotation_init="position_ik")` when an SMPL mesh or `motion135` approximation is
required.

ARDY-330 follows the same honesty rule. The official ARDY repository does not
provide ARDY-to-SMPL retargeting. Motius exposes
`ardy_core27_to_smpl22_joints` and `smpl22_joints_to_ardy_core27_joints` for
joint-level comparison until a validated rotation/mesh retargeter reports its
fitting error.

## HumanML3D Protocol Controls

Official HML263 encoding includes first-frame canonicalization, skeleton
retargeting, inverse kinematics, velocities, and foot-contact extraction. Motius
implements those operations in `joints_to_hml263`; its input must already be a
20-fps, metric, Y-up SMPL-22 joint sequence. `smpl_to_humanml263` additionally
materializes shape-aware joints and exposes frame-rate and coordinate conversion
explicitly.

With the same beta, gender, body-model file, frame selection, and coordinate
convention, the SMPL-H FK path is numerically equivalent to the body-model joint
output. The repository regression fixture reproduces official HumanML3D sample
`004822` within `1e-4`, including exact foot-contact channels. Different body
shape is an expected source-skeleton difference rather than an HML263 encoder
error.
