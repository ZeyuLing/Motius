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
from motius.motion.representation.convert import convert_motion

# Exact native position decode.
joints = convert_motion(motion_hml263, "hml263", "joints")
joints = convert_motion(motion_ms272, "ms272", "joints")

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
```

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
```

For an input NPZ with multiple arrays, add `--input-key motion`. NPZ outputs
store the converted array under `motion` and record source/target names.

## Supported Routes

| Source | Target | Semantics |
| ------ | ------ | --------- |
| HML263 | joints | Native RIC decode; deterministic |
| HML263 | motion135 | SMPL IK; lossy; needs SMPL assets |
| MS272 | joints | Native stored-position decode |
| MS272 | motion135 | Recovers root and rotations; subject shape is unavailable |
| HY-Motion-201 | motion135 | Exact prefix extraction |
| HY-Motion-201 | joints | Uses explicit stored RIC joints by default |
| motion135 | joints | FK; requires explicit `(22,3)` offsets |
| motion135 | HY-Motion-201 | FK appends 22 RIC joints; requires offsets |
| motion135 | MS272 | Uses the bundled canonical MS272 evaluator skeleton by default |
| DART276 | joints | Native DART decode, optional MBench coordinate conversion |
| DART276 | motion135 | Coordinate/floor bridge; retains DART temporal sampling |
| G1-38 | G1 qpos-36 | Exact root quaternion + 29-DOF decode |
| G1 qpos-36 | G1-38 | Optional root canonicalization and XY velocity encoding |

Routes can compose through `motion135`, for example DART276 to MS272. HML263
to MS272 first performs the lossy SMPL IK step.

## Why There Is No Generic Joints-To-HML263 Call

Official HML263 encoding includes first-frame canonicalization, skeleton
retargeting, inverse kinematics, velocities, and foot-contact extraction.
Those choices are part of a dataset/evaluator protocol, not a shape-only tensor
conversion. Motius therefore does not hide them behind a misleading generic
function.
