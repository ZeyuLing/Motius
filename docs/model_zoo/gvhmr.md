# GVHMR

GVHMR recovers camera-relative and world-grounded SMPL motion from monocular
video through gravity-view coordinates.

- Paper: <https://arxiv.org/abs/2409.06662>
- Official source: <https://github.com/zju3dv/GVHMR>
- Pinned source revision: `6ec3ca39336c50492c0fae65fba2fb831fc7d866`
- Official checkpoint: `gvhmr_siga24_release.ckpt`
- Checkpoint SHA-256:
  `4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91`
- License: upstream non-commercial research terms

**Tasks:** Monocular Motion Capture

## Motius integration

`GVHMRBundle` runs the pinned official environment through a subprocess and
hashes the actual release checkpoint used by every run. The setup applies
exact-source maintenance patches for degenerate two-view estimates and
optional render skipping; neither patch changes network inference.
`GVHMRPipeline` converts documented `hmr4d_results.pt` fields into the shared
`MonocularCaptureResult`.

```bash
bash tools/setup_gvhmr_env.sh
```

The setup pins the source under `outputs/tmp/gvhmr/`. Download the official
GVHMR, YOLO, ViTPose, HMR2, SMPL, and SMPL-X assets according to upstream
instructions; Motius does not redistribute them.

```python
from motius.models.gvhmr import GVHMRBundle
from motius.pipelines.gvhmr import GVHMRPipeline

bundle = GVHMRBundle(
    runtime_root="outputs/tmp/gvhmr/upstream",
    python_executable="outputs/tmp/gvhmr/conda-env/bin/python",
)
pipeline = GVHMRPipeline(bundle)
result = pipeline(
    "input.mp4",
    "outputs/inference/monocular_capture/gvhmr/run_001",
)
```

Pass a dense `(frames, 4)` `bbox_xyxy` array to evaluate a specific person
track. Motius writes the official `bbx.pt` cache and applies GVHMR's native
192:256, 1.2x crop conversion.

The adapter preserves `smpl_params_incam`, `smpl_params_global`, and
`K_fullimg`. Geometry is materialized with the fixed official
`make_smplx("supermotion")`, `smplx2smpl_sparse`, and neutral SMPL-24 joint
regressor. The official demo does not emit `camera_to_world`; Motius leaves it
unset rather than fabricating it.

## Evaluation Results

Parser, immutable checkpoint provenance, bbox injection, runtime maintenance,
and geometry-conversion paths are covered by tests.

### 3DPW Test

Protocol: `3dpw_test_camera_v1`, one inference item per official person track.
The run covers all 24 test sequences and all 37 person tracks. Missing 2D
annotations are interpolated only to form dense crops and remain excluded from
metrics.

| Coverage | MPJPE ↓ | PA-MPJPE ↓ | Accel ↓ |
|---:|---:|---:|---:|
| 100.00% | 62.65 mm | 47.66 mm | 5.47 m/s² |

- Artifact:
  `outputs/evaluation/monocular_capture/3dpw_test/gvhmr_gtcrop_v1/metrics.json`
- Metric samples: 35,515 pose frames and 35,441 acceleration frames
- Body: SMPL-24 joints plus materialized SMPL mesh
- Runtime revision: `6ec3ca39336c50492c0fae65fba2fb831fc7d866`
- Checkpoint SHA-256:
  `4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91`

The detector-driven official `Tracker.get_one_track` path is retained as a
diagnostic only. It covers 65.35% of valid person-frames and obtains 86.85 mm
MPJPE, 59.35 mm PA-MPJPE, and 7.19 m/s² acceleration error; it is not eligible
for complete-split ranking.

GVHMR requires a CUDA 12-compatible host driver (driver major at least 525).
Older DHC_DC A100 nodes are rejected by preflight.
