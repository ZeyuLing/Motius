# HYMotion-V2M

HYMotion-V2M converts a tracked monocular video into global SMPL-H motion using
SAM-3D-Body image tokens, camera conditioning, and a flow-matching motion
generator.

- Paper: <https://arxiv.org/abs/2512.23464>
- Official source: <https://github.com/Tencent-Hunyuan/HY-Motion-1.0>
- Motius runtime source: the first-class `HyMotionV2MBundle` and
  `HyMotionV2MPipeline` under `motius/`
- Runtime revision: `motius_hymotion_v2m_release_v1`
- Evaluated checkpoint SHA-256:
  `ed301af6b6fc6bc22dc69d8c7f48c1d6b2fff31d48a1717c8f5bc5ea92bc71df`
- Native output: 52-joint SMPL-H rotation, translation, shape, and joints at
  30 FPS

**Tasks:** Monocular Motion Capture

## Motius integration

The implementation is self-contained inside Motius and does not import a
reference repository. The video path is:

`video -> YOLOX/ByteTrack -> SAM-3D-Body tokens -> HYMotion-V2M -> SMPL-H`.

The generated result is converted into the shared pickle-free
`MonocularCaptureResult` contract without relabeling SMPL-H as SMPL. Camera and
world availability are reported explicitly.

```bash
bash tools/bootstrap_run_3dpw_hymotion_v2m_taiji.sh 8
```

## Evaluation Results

The target-conditioned benchmark path completed all 24 3DPW Test videos and
all 37 official person tracks. Each track uses its dense official 2D crop as
input; the model returns the exact source duration after inference on its fixed
360-frame canvas. The audit found no missing track, truncated output, or
non-finite numeric array.

| Protocol | Coverage | MPJPE | PA-MPJPE | Acceleration |
| --- | ---: | ---: | ---: | ---: |
| 3DPW Test camera, per-target crop | 100.00% | 270.65 mm | 139.25 mm | 6.118 m/s² |

- Full metric artifact:
  `outputs/evaluation/monocular_capture/3dpw_test/hymotion_v2m_gtcrop_v1/metrics.json`
- Completeness and finite-value audit:
  `outputs/evaluation/monocular_capture/3dpw_test/hymotion_v2m_gtcrop_v1/audit.json`
- Full prediction root:
  `outputs/evaluation/monocular_capture/3dpw_test/hymotion_v2m_gtcrop_v1/predictions/`

These metrics remain diagnostic and are not ranking eligible. HYMotion-V2M
currently uses an identity camera trajectory when a video-only camera estimator
is unavailable, so camera-space root motion is not comparable with methods that
recover camera motion. The earlier detector-driven, single-person result under
`outputs/evaluation/monocular_capture/3dpw_test/hymotion_v2m/` is retained only
for provenance.

## License and release scope

Model code, checkpoints, SAM-3D-Body assets, and SMPL-H files remain governed by
their respective upstream licenses. Motius releases the adapter, model card,
checkpoint provenance, and generated numeric contract; it does not redistribute
restricted third-party assets.
