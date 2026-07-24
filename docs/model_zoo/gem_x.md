# GEM-X

GEM-X is NVIDIA's monocular whole-body motion estimator using the native
SOMA-77 parametric body model.

- Paper lineage: <https://arxiv.org/abs/2505.01425>
- Official source: <https://github.com/NVlabs/GEM-X>
- Pinned source revision: `32992550dba114c62243fb55e361311972dce8f9`
- Pinned SOMA-X revision: `e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`
- Official artifact: <https://huggingface.co/nvidia/GEM-X>
- Checkpoint SHA-256:
  `4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e`
- Source license: Apache-2.0
- Weight license: NVIDIA Open Model License

**Tasks:** Monocular Motion Capture

## Motius integration

`GemXBundle` and `GemXPipeline` run the pinned official environment out of
process. The numeric exporter invokes the official SOMA layer and preserves:

- 77-joint axis-angle pose;
- identity coefficients;
- body-part scale parameters;
- camera/world root translation;
- native camera/world SOMA-77 joints.

```bash
DOWNLOAD_WEIGHTS=1 bash motius/models/gem_x/setup_runtime.sh
```

SOMA LFS assets and SAM-3D-Body assets remain separately governed and are not
redistributed.

```python
from motius.models.gem_x import GemXBundle
from motius.pipelines.gem_x import GemXPipeline

bundle = GemXBundle(
    runtime_root="outputs/tmp/gem_x/upstream",
    checkpoint="outputs/tmp/gem_x/upstream/inputs/pretrained/gem_soma.ckpt",
)
pipeline = GemXPipeline(bundle)
result = pipeline.run(
    "input.mp4",
    "outputs/inference/monocular_capture/gem_x/run_001",
    original_fps=30.0,
)
```

## Cross-model evaluation

Motius never converts SOMA-77 into fake SMPL vertices. Against SMPL benchmarks,
GEM-X is evaluated only on `common_hmr15_named_v1`, selected by audited joint
names. Cross-topology PVE and SMPL foot-vertex metrics are unavailable.

## Evaluation Results

The protocol-locked target-crop evaluation is complete on all 24 3DPW Test
videos and all 37 person tracks, with 100% coverage of the 35,515 official
valid person-frames:

| Protocol | PA-MPJPE ↓ | MPJPE ↓ | Accel ↓ | Coverage |
|---|---:|---:|---:|---:|
| `3dpw_test_camera_v1` | 53.20 mm | 84.38 mm | 5.616 m/s² | 100% |

The prediction artifact is
`outputs/evaluation/monocular_capture/3dpw_test/gem_x_gtcrop_v1/`, including
`metrics.json` and `audit.json`.

The pinned official demo does not estimate visual odometry. Without an
externally supplied camera trajectory it emits an identity camera trajectory,
which Motius records as
`official_demo_identity_fallback_without_external_vo`. The camera-space
`body_params_incam` and SOMA-77 joints are valid for the 3DPW common-joint
protocol; world-space metrics are unavailable for this run. PVE is also
unavailable because SOMA-77 and SMPL do not share a vertex topology.

For provenance, the earlier detector-driven diagnostic run completed all 24
videos but emitted one track per video and covered only 65.35% of valid
person-frames. Its unranked values were PA-MPJPE 65.49 mm, MPJPE 107.42 mm,
and acceleration error 6.374 m/s². That artifact remains at
`outputs/evaluation/monocular_capture/3dpw_test/gem_x/metrics.json`.
