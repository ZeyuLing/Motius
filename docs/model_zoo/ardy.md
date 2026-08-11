<h1 align="center">ARDY Model Card</h1>

<p align="center">
  <strong>Streaming autoregressive diffusion for text-driven and kinematically controlled motion.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.08741">Paper</a> ·
  <a href="https://doi.org/10.1145/3811284">ACM TOG</a> ·
  <a href="https://research.nvidia.com/labs/sil/projects/ardy/">Project Page</a> ·
  <a href="https://github.com/nv-tlabs/ardy">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-ardy-330-horizon40">Motius Checkpoint</a>
</p>

ARDY is NVIDIA's *Autoregressive Diffusion with Hybrid Representation for
Interactive Human Motion Generation* (SIGGRAPH 2026). It generates motion in
short autoregressive horizons, accepts text changes while a sequence is
running, and combines text with root paths, waypoints, full-body keyframes, or
sparse joint position and rotation constraints.

This Motius integration vendors the inference runtime under its Apache-2.0
license, loads the original NVIDIA checkpoints directly, and exposes batch and
stateful streaming APIs without a runtime dependency on a reference checkout.

<!-- MOTIUS_MODEL_CARD_NAV:START -->
<p align="center">
  <a href="#visual-results">Visual Results</a> ·
  <a href="#model-overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#motion-representation">Motion Representation</a>
</p>
<!-- MOTIUS_MODEL_CARD_NAV:END -->

## Visual Results

<!-- MOTIUS_TASK_DEMOS:START -->

### Task Demos

| Task | Input / condition | Rendered output | More |
| --- | --- | --- | --- |
| Text-to-Motion | a person walks forward, turns right, and starts jogging | <video src="https://github.com/user-attachments/assets/b4b5c83f-d19a-4d7e-81f2-b29604198152" controls></video> | [MP4](https://github.com/user-attachments/assets/b4b5c83f-d19a-4d7e-81f2-b29604198152) |
| Sequential Text-to-Motion | Continue twice: a person walks forward, turns right, and starts jogging<br><sub>Two stateful 40-frame autoregressive horizons</sub> | <video src="https://github.com/user-attachments/assets/b4903132-f291-4ea3-9660-3e680dbf6fdf" controls></video> | [MP4](https://github.com/user-attachments/assets/b4903132-f291-4ea3-9660-3e680dbf6fdf) |
| Kinematic Motion Control | a person walks forward, turns right, and starts jogging<br><sub>Five root-position and heading constraints over 80 frames</sub> | <video src="https://github.com/user-attachments/assets/fbaf6520-6824-43c7-ad28-ee820882c459" controls></video> | [MP4](https://github.com/user-attachments/assets/fbaf6520-6824-43c7-ad28-ee820882c459) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->

These previews render the native ARDY-330 body mesh and joint layout. They do
not use the experimental ARDY-to-SMPL joint bridge, because that bridge cannot
recover body shape or twist and would visibly misrepresent the generated
motion.

The text and sequential animations above are Motius pipeline outputs. The
kinematic-control animation is the official root-trajectory example mirrored
locally so GitHub can render it inline; the
[project page](https://research.nvidia.com/labs/sil/projects/ardy/) contains
the complete interactive capability set.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Sequential Text-to-Motion | `infer_sequential_text_to_motion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) |
| Kinematic Motion Control | `infer_kinematic_motion_control` | [Benchmark and examples](../leaderboards/kinematic_motion_control.md) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | ARDY-330: 20 fps; Unitree G1: 25 fps |
| Public preview | ARDY-330: 20 fps native; G1 and control previews retain their checkpoint clock |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | Two-stage autoregressive diffusion with explicit root and latent body streams |
| Tasks | Text-to-Motion, Sequential Text-to-Motion, Kinematic Motion Control |
| Venue | ACM TOG 45(4), SIGGRAPH 2026, Article 86 |
| Native skeletons | ARDY-27 and Unitree G1 |
| Native FPS | ARDY-330 20 fps; G1 25 fps |
| Text encoder | LLM2Vec with Meta-Llama-3-8B-Instruct |
| Checkpoint | [`ZeyuLing/motius-ardy-330-horizon40`](https://huggingface.co/ZeyuLing/motius-ardy-330-horizon40) |
| Pipeline | `motius.pipelines.ardy.ARDYPipeline` |
| Upstream revision | `nv-tlabs/ardy@693f74d13b3d04a0a22ce127ee79c929dd89756b` |

[Kinematic Motion Control benchmark protocol](../leaderboards/kinematic_motion_control.md)

### Checkpoints

| Alias | Skeleton | FPS | Generated horizon | Official checkpoint |
| ----- | -------- | --: | ----------------: | ------------------- |
| `core` / `core40` | ARDY-27 | 20 | 40 frames | [`nvidia/ARDY-Core-RP-20FPS-Horizon40`](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon40) |
| `core8` | ARDY-27 | 20 | 8 frames | [`nvidia/ARDY-Core-RP-20FPS-Horizon8`](https://huggingface.co/nvidia/ARDY-Core-RP-20FPS-Horizon8) |
| `g1` / `g152` | Unitree G1 | 25 | 52 frames | [`nvidia/ARDY-G1-RP-25FPS-Horizon52`](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) |
| `g18` | Unitree G1 | 25 | 8 frames | [`nvidia/ARDY-G1-RP-25FPS-Horizon8`](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon8) |

NVIDIA has not released an ARDY SMPL-X checkpoint. The official release only
contains ARDY-27 and Unitree G1 checkpoints; the upstream README lists a
SOMA checkpoint as coming soon. SMPL-X text-to-motion support belongs to
NVIDIA's separate KIMODO-SMPLX release, not to ARDY.

The Motius release packages the Horizon-40 ARDY-330 checkpoint together with
the complete LLM2Vec/Llama text stack. It does not fetch the gated base model
or either PEFT adapter during loading. NVIDIA's other variants remain listed
above for provenance, but are not aliases of this Motius artifact.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.ardy.ARDYPipeline` |
| Bundle | `motius.models.ardy.ARDYBundle` |
| Runtime | `motius.models.ardy.network` |
| Representation API | `motius.motion.representation.ardy` |

## Quick Start

### Installation

ARDY-330 checkpoint inference requires the ARDY optional dependencies:

```bash
python -m pip install -e ".[ardy]"
```

The exact upstream local text stack requires Python 3.10 or newer:

```bash
python -m pip install -e ".[ardy,ardy-text]"
```

`ardy-text` pins the upstream-tested `transformers==5.8.1` and `peft>=0.19`.
After downloading the Motius snapshot, loading is fully local. The denoiser
can also run with `text_encoder=False` and externally computed LLM2Vec features
of shape `(B, tokens, 4096)`.

Use the Horizon-40 ARDY-330 model for the strongest default generation and
constraint following. Horizon-8 is intended for lower-latency replanning.

### Text To Motion

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-ardy-330-horizon40",
    bundle_kwargs={
        "device": "cuda",
        "text_encoder_mode": "local",
    },
)

motion = pipe.infer_text_to_motion(
    ["a person walks forward, turns right, and starts jogging"],
    [160],
    num_denoising_steps=4,
    seed=1234,
)

features = motion["features"][:, :160]       # (1, 160, 330)
joints = motion["posed_joints"][:, :160]    # (1, 160, 27, 3)
```

Batch generation accepts one length per prompt. Returned arrays are padded to
the longest item; use `motion["lengths"]` to crop each sample.

### Streaming Prompt Updates

```python
sequence = pipe.infer_sequential_text_to_motion(
    [
        "walk forward at a relaxed pace",
        "quickly sidestep to the left",
    ],
    num_denoising_steps=4,
)
segments = sequence["segments"]
state = sequence["state"]
```

The call returns one generated segment per prompt and an `ARDYStreamState`
containing normalized motion history. Pass `state=sequence["state"]` into the
next call to continue without resetting the motion.

### Kinematic Constraints

```python
root_path = pipe.root2d_constraint(
    frame_indices=[0, 20, 40, 60],
    root_2d=[[0.0, 0.0], [0.4, 0.2], [0.9, 0.3], [1.4, 0.0]],
    global_root_heading=[0.0, 0.2, 0.1, 0.0],  # radians
)

motion = pipe.infer_kinematic_motion_control(
    ["a person follows a curved path"],
    [80],
    constraints=[root_path],
    num_denoising_steps=4,
)
```

`fullbody_keyframe_constraint` accepts native global joint positions and
rotation matrices. `end_effector_constraint` additionally accepts native joint
names. Existing official constraint JSON can be loaded with
`pipe.load_constraints(path)`.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Sequential Text-to-Motion | [Published results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | BABEL semantic and transition metrics; normalized uTMR FID |
| Kinematic Motion Control | [Published results](../leaderboards/kinematic_motion_control.md) | Native-skeleton protocol; rows remain pending |

### Canonical HumanML3D Semantic Results

| Evaluator | Variant | n | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HumanML3D Official | Default | — | — | — | — | — | — | — | Not measured |
| MotionStreamer Evaluator | Default | — | — | — | — | — | — | — | Not measured |
| Motius Joint-Position Evaluator | Default | — | — | — | — | — | — | — | Not measured |

<!-- MOTIUS_CANONICAL_METRICS:END -->


### Released Rigplay Model

The ARDY paper reports the following text-only results for the default Core
Horizon-40 FSQ model with 10 denoising steps on the Bones Rigplay test set.

| R-Precision | FID | Foot skating |
| ----------: | --: | ------------: |
| 65.47% | 0.027 | 0.264 m/s |

Its constrained-motion evaluation reports 0.250 m/s foot skating, 2.23-degree
joint rotation error, 0.025 m sparse joint-position error, 0.023 m full-body
keyframe error, 0.015 m trajectory error, and 0.024 m waypoint error.

### HumanML3D Paper Benchmark

The paper's separately trained HumanML3D benchmark model reports R-Precision
0.729, FID 0.044, skating ratio 6.28%, constraint error 4.15 cm, and 0.15 s
latency. That benchmark model is not one of the four released Core/G1 Rigplay
checkpoints above. Motius lists this paper-only result in the
Text-to-Motion · HumanML3D benchmark as an official-paper row, separate from
the released checkpoint rows.

Motius's HumanML3D, MotionStreamer, and joint-position evaluator rows are not
reported yet: the released ARDY-27 output first needs a validated SMPL-22
retargeting protocol. This card keeps the official native metrics separate
instead of presenting an unverified cross-skeleton score.

## Motion Representation

ARDY's hybrid latent representation is internal to the tokenizer. The public
pipeline returns the exact explicit checkpoint representation:

| Field | ARDY-330 | Unitree G1 explicit 414D |
| ----- | -------: | -----: |
| Root position | 3 | 3 |
| Global root heading `(cos, sin)` | 2 | 2 |
| Root-local non-root joint positions | 78 | 99 |
| Global joint rotations, 6D | 162 | 204 |
| Global joint velocities | 81 | 102 |
| Foot contacts | 4 | 4 |
| **Total** | **330** | **414** |

The checkpoint normalization files contain four additional local-root channels
used by the tokenizer, so their stored widths are 334 and 418. Those four
statistics are not extra output channels.

Use `split_ardy_features` for named slices and pass the checkpoint's exact
`motion_rep` object to `convert_motion` for denormalization and joint decoding:

```python
from motius.motion import convert_motion
from motius.motion.representation import split_ardy_features

parts = split_ardy_features(motion["features"], "ardy_330")
joints = convert_motion(
    motion["features"],
    "ardy_330",
    "joints",
    motion_rep=pipe.bundle.motion_rep,
    is_normalized=True,
)
```

ARDY-330 is the public Motius name for NVIDIA ARDY's released 27-joint human
skeleton representation. The official Hugging Face repositories keep `Core` in
their artifact names; Motius treats that as an upstream checkpoint alias, not as
a separate body-model family. Unitree G1 is the same robot skeleton family used
elsewhere in Motius; ARDY's G1 checkpoint simply uses its own 414D explicit
tensor. ARDY-27 is not SMPL-22. NVIDIA's official ARDY repository does not
include an ARDY-to-SMPL or SMPL-to-ARDY rotation retargeter, so Motius does not
silently truncate or rename joints when crossing skeletons.

For joint-position visualization and evaluator smoke tests, Motius provides a
named bridge:

```python
from motius.motion import convert_motion, smpl22_joints_to_ardy_core27_joints

smpl22_joints = convert_motion(
    motion["features"],
    "ardy_330",
    "smpl22_joints",
    motion_rep=pipe.bundle.motion_rep,
    is_normalized=True,
)
ardy27_joints = smpl22_joints_to_ardy_core27_joints(smpl22_joints)
```

These bridges map between ARDY-27 and SMPL-22 joint positions. They do not
recover SMPL twist, body shape, a valid `motion135` rotation sequence, or a
full ARDY-330 feature tensor from SMPL. SMPL mesh rendering and leaderboard
evaluation must use a separately validated position-IK bridge and report its
fitting error. Unitree G1 output can be exported exactly to MuJoCo qpos-36.

## Citation and License

The adapted runtime preserves NVIDIA's Apache-2.0 notices and the vendored
LLM2Vec MIT attribution. See `motius/models/ardy/LICENSE` and
`motius/models/ardy/ATTRIBUTIONS.md`. Checkpoint and dataset licenses are
separate from the source-code license.

```bibtex
@article{zhao2026ardy,
  title     = {ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation},
  author    = {Zhao, Kaifeng and Petrovich, Mathis and Zhang, Haotian and Wang, Tingwu and Tang, Siyu and Rempe, Davis},
  journal   = {ACM Transactions on Graphics (TOG)},
  year      = {2026},
  volume    = {45},
  number    = {4},
  articleno = {86},
  doi       = {10.1145/3811284}
}
```

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
