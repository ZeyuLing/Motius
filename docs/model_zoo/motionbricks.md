<h1 align="center">MotionBricks Model Card</h1>

<p align="center"><strong>Native PyTorch pose-conditioned generation for SOMA humans and Unitree G1.</strong></p>
<p align="center">
  <a href="https://arxiv.org/abs/2604.24833">Paper</a> ·
  <a href="https://github.com/NVlabs/motionbricks">Official source</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-motionbricks-soma-rp-v1">SOMA checkpoint</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-motionbricks-g1-rp-v1">G1 checkpoint</a>
</p>

## Model Overview

Motius integrates the independent MotionBricks release at commit
`54fc563aeaaf819a753d72cc9fdc6713989cdd43`. The implementation uses native PyTorch
modules and loads converted official SafeTensors weights. It does not require
ONNX Runtime, the upstream Python package, Newton or TensorRT. The optional full
CUDA pose/scene runtime uses Warp; SOMA scene loading also uses USD.

| Item | SOMA RP v1 | G1 RP v1 |
| --- | --- | --- |
| Skeleton | SOMA 30 joints | Unitree G1 29 DOF |
| Public representation | 366D motion features | 36D qpos |
| Native clock | 30 fps | 30 fps |
| Conditions | 4 recent poses + up to 4 target slots | Same |
| Raw generation length | 24–64 frames | Same |
| Default trimmed length | 18–58 frames | Same |
| Bundle | `MotionBricksReleaseBundle` | Same |
| Pipeline | `MotionBricksReleasePipeline` | Same |
| Task registration | Capability API; no canonical benchmark | Same |

The native Bundle owns **pose, root, encoder, decoder and quantizer**. Neural
layers preserve the official checkpoint architecture and equations; this is an
attributed source adaptation and framework refactor, not a claim that the model
was independently invented. Root-length selection, pose-token sampling and
condition assembly live in the pipeline. Motion features, normalization, G1
kinematics and SOMA pose conversion live under `motius.motion`. There is no
vendored upstream runtime tree or upstream `Predictor`/`supporting_nets` wrapper.
The prior ONNX adapter has been removed.

## Quick Start

Install Motius and its core PyTorch/SafeTensors dependencies from this repository
(version 0.3.23 or later). The new release does not require the legacy
`motionbricks` extra.

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-motionbricks-g1-rp-v1",
    bundle_kwargs={"device": "cpu"},
)
# Both arrays: (N, 4, 36), four consecutive poses sampled at 30 Hz.
# Inputs must be real G1 poses placed in the same Z-up coordinate space.
result = pipe.infer_g1_inbetweening(context_qpos, target_qpos)
qpos = result["qpos"][0]  # (T,36), MuJoCo WXYZ root quaternion
```

The G1 convenience method accepts WXYZ quaternions by default and returns the
same convention. Set `quaternion_order="xyzw"` for the upstream convention.
It enables all eight conditioning slots. Use the raw tensor dictionary API for
partial target masks:

```python
pipe = Pipeline.from_pretrained("ZeyuLing/motius-motionbricks-soma-rp-v1")
result = pipe.infer_pose_inbetweening(inputs)
motion = result["motions"][0]  # (T,366), SOMA model-space features
raw = pipe.infer_raw(inputs)   # native torch tensors, including boundary frames
pipe.save_pretrained("outputs/motionbricks_export")
```

### Input and Output Contract

| Input tensor | SOMA | G1 | Type |
| --- | --- | --- | --- |
| `global_root_values` | `(N,8,5)` | — | FP32 |
| `local_root_values` | `(N,8,4)` | — | FP32 |
| `local_poses` | `(N,8,267)` | — | FP32 |
| `context_qpos`, `target_qpos` | — | `(N,4,36)` each | FP32, XYZW in raw API |
| `has_global_root_values`, `has_local_root_values`, `has_local_poses` | `(N,8)` each | Same | Binary mask |
| `allowed_pred_num_tokens` | `(N,11)` | Same | Binary flags for counts 6–16 |

SOMA is Y up in meters. Root features contain XYZ position and heading cosine/sine;
local root features contain heading rate, XZ velocity and height. Local pose
features contain 29 joint positions and 30 six-dimensional rotations. Its 366D
output contains root (5), positions (87), rotations (180), velocities (90) and
foot contacts (4). These are not SMPL parameters. G1 is Z up with XYZW quaternions
in `infer_raw` and `infer_pose_inbetweening`.

`infer_pose_inbetweening` returns `motions` and `foot_contacts` lists, `lengths`,
`pred_num_tokens`, `fps=30`, `representation`, and `endpoint_corrected=False`.
The default removes three boundary frames from each end. Set
`trim_boundaries=False` to retain all valid raw frames. Padded frames after
`4 * pred_num_tokens` are not valid motion.

### Full CUDA Runtime and Scene Control

Install `pip install -e '.[motionbricks-runtime]'`. The full runtime preserves the
native neural implementation and adds world-space alignment, official boundary
decoding, G1 root filtering, clamped foot contacts, endpoint correction, SOMA local
pose reconstruction/IK, generated-motion recording and playback. These attributed
algorithms live in Motius motion services and pipelines; the runtime engine and
headless scene entry point are native. No upstream package is imported.

```python
pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-motionbricks-g1-rp-v1",
    bundle_kwargs={"device": "cuda"},
)
from motius.configuration.motionbricks_runtime import MotionBricksInferenceConfig
with pipe.create_runtime() as engine:
    context, target, output = engine.create_input(), engine.create_input(), engine.create_output()
    # Warp arrays: (4,36) recent poses and (1..4,36) target poses, XYZW, Z up.
    context.update_from_poses(context_poses, stream=engine.stream)
    target.update_from_poses(target_poses, stream=engine.stream)
    engine.generate_animation(context, target, output,
        config=MotionBricksInferenceConfig(correct_generated_diff=True))
    corrected_qpos = output.poses.numpy()[:output.num_frames]
```

The default allows 8–14 tokens; explicit counts can range from 6 to 16. Output
contains `4 * token_count - 6` frames at 30 Hz. The first and last poses are
corrected toward the last context and first target poses. Set
`correct_generated_diff=False` to retain predicted endpoints. Full runtime
inference disables TF32/autocast and restores the caller's precision settings.
SOMA `create_input`/`create_output` take its USD skeleton; `load_scene` handles this
and the Y-up model to Z-up world transform automatically.

Download optional, separately licensed official clips/documents:

```bash
PYTHONPATH=. python tools/prepare_motionbricks_scene_assets.py
```

```python
from pathlib import Path
import numpy as np
from motius.pipelines.motionbricks.scene_motion import LocomotionCommand

assets = Path("checkpoints/models/motionbricks_scene_assets")
with pipe.load_scene(assets / "scenes/g1_gym.scene.json", assets_root=assets, seed=42) as scene:
    frame = scene.step(LocomotionCommand(
        facing_direction=np.array([1., 0., 0.]),
        movement=np.array([1., 0., 0.]),
        requested_mode="directional_2d",
    ), delta_time=1/30, auto_interact=True)
    pose, state = frame["pose"], frame["state"]
    # scene.interact() presses the nearest interaction; interact(primitive_id)
    # explicitly selects a primitive for scripted control.
```

Use `soma_gym.scene.json` with the SOMA checkpoint. Controls include linear,
directional and crouching locomotion, style modes, idle transitions, target
blending, velocity-matched sequences and sitting/standing. Automatic interaction
blocks repeat activation until the character leaves the interaction region.
`seed` fixes idle clip selection. Rendering, mesh skinning, GUI/input-device
handling, physical simulation and text-to-motion are outside this headless API.
Raw APIs remain uncorrected; they do not silently switch to scene generation.
A canonical benchmark and new-release training protocol are not registered.

## Checkpoint Format

Each native Hub artifact contains `model.safetensors` with five neural roots,
`normalization.npz`, `motionbricks_release.json`, `model_index.json`, model card,
license notices and the independent validation certificate. All official neural
keys are consumed and loaded strictly; no arbitrary YAML targets are executed.
The source is `nvidia/MotionBricks`, release `3cb362d`, with per-file SHA256 and
size verification against pinned official metadata.

To convert independently:

```bash
PYTHONPATH=. python tools/prepare_motionbricks_release.py --model soma --output outputs/motionbricks_soma
PYTHONPATH=. python tools/prepare_motionbricks_release.py --model g1 --output outputs/motionbricks_g1
```

## Evaluation Results

Validation uses a separate official source checkout pinned to
`54fc563aeaaf819a753d72cc9fdc6713989cdd43`; it does not call Motius as its own
reference. All numbers describe tested cases, not universal numerical guarantees.

- CPU and H200 GPU FP32 raw generation: 8 cases per device, batch sizes 1/2,
  predicted/forced lengths, 6/16-token extremes and partial target masks. All valid
  outputs, contacts and token counts are bitwise equal (maximum absolute error 0).
- Full CUDA pose generation: official TensorRT FP32 versus native PyTorch, TF32
  disabled, 10 cases using real G1 CSV/SOMA BVH clips and the official USD skeleton. Tests cover
  world translation/yaw, 1/2/4 target poses, correction on/off and generated
  context reuse. Pose tolerance is `atol=2e-4, rtol=2e-4`; contact tolerance is
  `atol=5e-4, rtol=2e-4`. Measured maximum errors are `1.58e-5` for pose
  components and `2.35e-4` for contacts. TensorRT is not bitwise equal.
- Continuous character replay compares native generation/control with independent
  official PyTorch models and official pose/state/playback code. The certificate
  records 990 frames per rig, with bitwise-equal poses (maximum error 0), covering
  movement/style transitions and all four gym primitives, including sitting/standing.
  TensorRT floating-point differences can grow during closed-loop feedback; this
  replay certificate is explicitly for the PyTorch reference backend.
- 86 official headless algorithm tests pass with assertions unchanged and only
  import/mock namespaces adapted. Another 64 cases compare manual/automatic
  interaction dispatch against unchanged official dispatch methods.

G1 raw cases use an official walking CSV; SOMA raw cases use deterministic
synthetic poses. Full runtime/replay uses real official clips. These tests verify
implementation agreement, not motion quality or a leaderboard score. Other GPUs,
PyTorch versions and reduced precision are not covered by these measurements.

```bash
PYTHONPATH=. python tools/verify_motionbricks_release_parity.py \
  --source /path/to/pinned/official/motionbricks --device cuda \
  --g1-csv /path/to/official/Neutral_walk_forward_002__A057.csv
PYTHONPATH=. python tools/verify_motionbricks_runtime_parity.py --source /path/to/pinned/official/motionbricks
PYTHONPATH=. python tools/verify_motionbricks_scene_parity.py --source /path/to/pinned/official/motionbricks
PYTHONPATH=. python tools/verify_motionbricks_runtime_tests.py --source /path/to/pinned/official/motionbricks
PYTHONPATH=. python tools/verify_motionbricks_dispatch_parity.py --source /path/to/pinned/official/motionbricks
```

Only the TensorRT comparison requires original ONNX graphs and TensorRT; the
reference checkout and graphs stay outside the installed Motius runtime.
Hub artifacts include `validation.json` (CPU), `validation_cuda.json`,
`validation_runtime.json`, `validation_scene.json`, `validation_runtime_tests.json`
and `validation_dispatch.json`. Native weight tensors are unchanged by this
runtime update; checkpoint metadata/cards and evidence are updated together.

## Legacy G1 Navigation

The earlier `ZeyuLing/motius-motionbricks-g1` repository remains a distinct legacy
artifact. It is not the RP-v1 native checkpoint. It uses `MotionBricksBundle`,
`MotionBricksPipeline`, 413D/414D/418D features and optional MuJoCo dependencies:

```python
pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-motionbricks-g1",
    bundle_kwargs={"device": "cuda"},
    controller="random",
)
result = pipe.infer_g1_qpos_generation(steps=240, seed=1234)
# Equivalent legacy navigation entry point:
# result = pipe.infer_g1_realtime_navigation(steps=240, seed=1234)
```

The older source/training/rollout certificates apply to the GR00T preview at
`45c03facbf848e0c743091b7d9c17952096ff908`, not the new SOMA release.

## Citation and License

MotionBricks: *Scalable Real-Time Motions with Modular Latent Generative Model
and Smart Primitives*, Wang et al., ACM TOG / SIGGRAPH 2026,
[DOI 10.1145/3811334](https://doi.org/10.1145/3811334).

- Source code: Apache-2.0, retaining NVIDIA and dependency notices.
- Weights: [NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/).
- Official BVH/CSV samples: separate NVIDIA Sample Data Evaluation License;
  samples are not included in the native checkpoint artifacts.
- [Source attribution](../../motius/licenses/motionbricks/ATTRIBUTIONS.md).

[Model Zoo](README.md) · [Task Registry](../tasks/README.md)
