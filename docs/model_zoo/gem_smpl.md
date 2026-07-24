# GEM-SMPL

GEM-SMPL is NVIDIA's SMPL release of **GEM: A Generalist Model for Human
Motion**, originally published as GENMO. It supports video motion estimation
and multimodal generation; this adapter exposes the official monocular-video
estimation path.

- Paper: <https://arxiv.org/abs/2505.01425>
- Official source: <https://github.com/NVlabs/GENMO>
- Pinned source revision: `16bebf402d8893184249ee206d957b8248cd8310`
- Official artifact: `nvidia/GEM-X/gem_smpl.ckpt`
- Checkpoint SHA-256:
  `1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`
- Source license: NVIDIA OneWay Noncommercial
- Weight license: NVIDIA Open Model License

**Tasks:** Monocular Motion Capture

## Motius integration

The official environment runs out of process. Motius does not import or copy
the upstream model implementation. `GemSmplBundle` owns immutable source and
checkpoint provenance; `GemSmplPipeline` invokes the pinned demo and converts
its `smpl_params.pt` output into `MonocularCaptureResult`.

```bash
DOWNLOAD_WEIGHTS=1 bash motius/models/gem_smpl/setup_runtime.sh
```

Additional licensed SMPL-X and upstream detector assets listed by the setup
script remain user supplied.

```python
from motius.models.gem_smpl import GemSmplBundle
from motius.pipelines.gem_smpl import GemSmplPipeline

bundle = GemSmplBundle(
    runtime_root="outputs/tmp/gem_smpl/upstream",
    checkpoint="outputs/tmp/gem_smpl/upstream/inputs/pretrained/gem_smpl.ckpt",
)
pipeline = GemSmplPipeline(bundle)
result = pipeline.run(
    "input.mp4",
    "outputs/inference/monocular_capture/gem_smpl/run_001",
    original_fps=30.0,
)
```

The adapter preserves camera/global SMPL parameters and named SMPL-24 joints
materialized with the official body layer. Vertices are not claimed unless
explicitly exported, so PVE remains unavailable in the current adapter.

## Evaluation Results

The verified 3DPW Test target-crop run covers 100% of valid person-frames.
It obtains PA-MPJPE 46.45 mm, MPJPE 64.46 mm, and acceleration error
5.713 m/s². Each inference item uses one official 3DPW person track with a
1.2x square crop derived from `poses2d` and five-frame crop smoothing.
The metric artifact is
`outputs/evaluation/monocular_capture/3dpw_test/gem_smpl_gt_crop_v1/metrics.json`.
PVE remains unavailable because the current artifact does not export vertices.
