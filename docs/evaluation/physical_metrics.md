# Physical Motion Metrics

Motius provides checkpoint-free, joint-level physical plausibility metrics for
motions represented on the shared SMPL-22 skeleton. The implementation follows
the non-VLM MBench motion-quality protocol used by the public HumanML3D
leaderboard.

## Protocol

Inputs must be world-space SMPL-22 joints in metres with a Y-up coordinate
frame. Finite differences are measured per frame, so compare methods only after
they have been converted to the same skeleton, coordinate system, FPS, and frame
selection policy.

| Metric | Meaning | Leaderboard unit | Interpretation |
| ------ | ------- | ---------------- | -------------- |
| `Slide` | Horizontal foot displacement during detected contact | mm/frame | Lower is better |
| `Float` | Fraction of frames with implausible unsupported foot motion | % | Lower is better |
| `Jitter` | Global plus root-relative mean joint acceleration | 1,000 x raw | Lower is better |
| `Dynamic` | Global plus root-relative mean joint velocity | 1,000 x raw | Compare with GT; do not minimize |
| `Penet` | Foot depth below the estimated floor | mm | Diagnostic under the min-foot floor |

The floor is the per-clip minimum foot height. This removes constant root-height
offsets introduced by representation conversion, but it also makes `Penet`
nearly degenerate. Mesh self-intersection and NRDF `PoseQ` require additional
models or geometry dependencies and are not part of this checkpoint-free API.

## Python API

Evaluate canonical joints directly:

```python
import numpy as np

from motius.evaluation.metrics import (
    compute_physical_metrics,
    table_scaled_physical_metrics,
)

joints = np.load("outputs/evaluation/example/joints.npy")  # (T, 22, 3), metres
raw = compute_physical_metrics(joints)
display = table_scaled_physical_metrics(raw)
```

Use a supported motion representation:

```python
from motius.evaluation.metrics import physical_metrics_from_motion

ms272_metrics = physical_metrics_from_motion(motion_272, "ms272")

motion135_metrics = physical_metrics_from_motion(
    motion_135,
    "motion135",
    bone_offsets=smpl22_rest_offsets,
)
```

Aggregate per-clip rows without changing the protocol:

```python
from motius.evaluation.metrics import aggregate_physical_metrics

summary = aggregate_physical_metrics(
    compute_physical_metrics(clip) for clip in all_joint_clips
)
```

## Fair Comparison Checklist

1. Convert every method to the same SMPL-22 joints and Y-up world frame.
2. Resample every motion to the same FPS before computing finite differences.
3. Apply the same test IDs, duration crop, and failed-sample policy.
4. Report GT as a reference row, not as a ranked method.
5. Interpret `Dynamic` by distance to GT and keep `Penet` diagnostic when using
   the per-clip minimum-foot floor.

The implementation lives in
[`motius/evaluation/metrics/physical.py`](../../motius/evaluation/metrics/physical.py).
