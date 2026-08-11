# Motion Repair · Source-Only D3 Pipeline

Motius includes a training-free Motion Repair pipeline implementing the frozen
policy used by the paper's formal evaluation:

```text
global SMPL-22 rotations: D3 Whittaker smoothing, lambda = 3
root translation:         D3 Whittaker smoothing, lambda = 3
ground correction:        prefix_q98, 10-frame prefix, 30-frame transition
```

The pipeline consumes only the corrupted motion and fixed SMPL-22 bone offsets.
It does not read a clean target, a corruption label, a learned artifact, a
model, or a checkpoint. The fixed policy must not be retuned on the formal test
set.

## Input contract

- `motion135`: a finite NumPy array with shape `[T, 135]` and a floating dtype;
- layout: root XYZ translation followed by 22 local row-convention 6D rotations;
- coordinates: right-handed, Y-up, metres;
- `bone_offsets`: fixed SMPL-22 offsets with shape `[22, 3]`, in metres.

## Python API

```python
import numpy as np

from motius.pipelines.motionrepair import MotionRepairPipeline

motion135 = np.load("corrupted_motion135.npy")
bone_offsets = np.load("smpl22_offsets.npy")

pipeline = MotionRepairPipeline()
result = pipeline.infer_motion_repair(motion135, bone_offsets)

repaired = result.motion135
rotation_support = result.rotation_support      # [T, 22]
translation_support = result.translation_support  # [T]
```

The same operation is available as `motius.motion.repair_motion135`. The
configuration entrypoint is `configs/motionrepair/d3_source_only.py`.

## Evaluation support

The returned support is computed from the float64 source and repaired values
with the frozen `1e-12` tolerance. Use `rotation_support` and
`translation_support` as the method-native support when serializing results for
the fixed-support protocol. A dense source-only rule does not consume an oracle
mask; it reports the cells and frames it actually changes.

This utility operates on `motion135`. Dataset-specific conversions (for
example, Z-up SMPL axis-angle records) must be performed before and after this
pipeline by the relevant representation adapter.
