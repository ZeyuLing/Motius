<p align="center">
  <img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/brand/motius-logo-readme.png" width="620" alt="Motius logo">
</p>

<h1 align="center">Motius</h1>

<p align="center">
  <strong>A modular training, evaluation, and inference framework for human motion generation.</strong>
</p>

<p align="center">
  <a href="#model-zoo">Model Zoo</a> |
  <a href="#evaluator-zoo">Evaluator Zoo</a> |
  <a href="#motion-representation-toolkit">Motion Toolkit</a> |
  <a href="docs/getting_started.md">Getting Started</a> |
  <a href="docs/architecture.md">Architecture</a> |
  <a href="docs/development.md">Development Guide</a>
</p>

Motius packages motion-generation methods as consistent model bundles,
trainers, pipelines, evaluators, and visualization utilities. The public repo
is being opened method by method: the reusable core is available now, and each
released method will ship with a model card, checkpoint path, evaluation
results, and qualitative SMPL renders.

Motius also ships an explicit [Motion Toolkit](docs/motion/README.md) for
converting HML263, MotionStreamer-272, HY-Motion-201, DART276, and SMPL
`motion135`, plus SMPL/SOMA/G1 retargeting. Its documentation records skeleton,
coordinate, FPS, and 6D rotation conventions for every route.

## Model Zoo

Task labels use a controlled vocabulary: `T2M`, `M2T`, `TP2M`,
`Multi-Prompt T2M`, `Motion Control`, and `Kinematic Control`. Properties such
as zero-shot, streaming, latent, or autoregressive are described in the model
cards rather than treated as separate tasks.

| Method | Tasks | Motion Rep. | Checkpoint | Card | References |
| ------ | ---- | ----------- | ---------- | ---- | ---------- |
| MDM | T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) | [Model Card](docs/model_zoo/mdm.md) | [Paper](https://arxiv.org/abs/2209.14916) / [Code](https://github.com/GuyTevet/motion-diffusion-model) |
| T2M-GPT | T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) | [Model Card](docs/model_zoo/t2mgpt.md) | [Paper](https://arxiv.org/abs/2301.06052) / [Code](https://github.com/Mael-zys/T2M-GPT) |
| MoMask | T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d) | [Model Card](docs/model_zoo/momask.md) | [Paper](https://arxiv.org/abs/2312.00063) / [Code](https://github.com/EricGuo5513/momask-codes) |
| MoGenTS | T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) | [Model Card](docs/model_zoo/mogents.md) | [Paper](https://arxiv.org/abs/2409.17686) / [Code](https://github.com/weihaosky/mogents) |
| MotionGPT | T2M, M2T | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d) | [Model Card](docs/model_zoo/motiongpt.md) | [Paper](https://arxiv.org/abs/2306.14795) / [Code](https://github.com/OpenMotionLab/MotionGPT) |
| FlowMDM | T2M, Multi-Prompt T2M, TP2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d) | [Model Card](docs/model_zoo/flowmdm.md) | [Paper](https://arxiv.org/abs/2402.15509) / [Code](https://github.com/BarqueroGerman/FlowMDM) |
| MotionMillion | T2M | MotionStreamer-272 | [7B](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272) / [3B](https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272) | [Model Card](docs/model_zoo/motionmillion.md) | [Paper](https://arxiv.org/abs/2507.07095) / [Code](https://github.com/VankouF/MotionMillion-Codes) |
| MotionStreamer | T2M, Multi-Prompt T2M, TP2M | MotionStreamer-272 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) | [Model Card](docs/model_zoo/motionstreamer.md) | [Paper](https://arxiv.org/abs/2503.15451) / [Code](https://github.com/zju3dv/MotionStreamer) |
| HY-Motion T2M | T2M | HY-Motion-201 | [Full](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0) / [Lite](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) | [Model Card](docs/model_zoo/hymotion_t2m.md) | [Paper](https://arxiv.org/abs/2512.23464) / [Code](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) |
| KIMODO | T2M, Multi-Prompt T2M, TP2M, Kinematic Control | SOMA / G1 / SMPL-X | [SOMA-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp) / [G1-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp) / [G1-SEED](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed) / [SMPLX-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp) | [Model Card](docs/model_zoo/kimodo.md) | [Paper](https://arxiv.org/abs/2603.15546) / [Code](https://github.com/nv-tlabs/kimodo) |
| MLD | T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) | [Model Card](docs/model_zoo/mld.md) | [Paper](https://arxiv.org/abs/2212.04048) / [Code](https://github.com/ChenFengYe/motion-latent-diffusion) |
| MotionLCM | T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) | [Model Card](docs/model_zoo/motionlcm.md) | [Paper](https://arxiv.org/abs/2404.19759) / [Code](https://github.com/Dai-Wenxun/MotionLCM) |
| ViMoGen | T2M | DART276 | [HF](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) | [Model Card](docs/model_zoo/vimogen.md) | [Paper](https://arxiv.org/abs/2510.26794) / [Code](https://github.com/MotrixLab/ViMoGen) |
| DART | T2M, Motion Control | DART276 | [HF](https://huggingface.co/ZeyuLing/motius-dart-humanml3d) | [Model Card](docs/model_zoo/dart.md) | [Paper](https://arxiv.org/abs/2410.05260) / [Code](https://github.com/zkf1997/DART) |

## Evaluator Zoo

Motius model cards report text-to-motion metrics with three evaluator views:
HumanML3D official metrics, MotionStreamer Evaluator metrics, and the Motius
joint-position evaluator trained on unified SMPL-22 joints. Historical
contrastive-evaluator rows are not part of the public Evaluation tables.
G1-native methods additionally use the robot-specific TMR-G1 evaluator.

| Evaluator | Purpose | Motion Rep. | Checkpoint | Card | Reference |
| --------- | ------- | ----------- | ---------- | ---- | --------- |
| HumanML3D Official | Standard T2M leaderboard metrics on the selected-caption HumanML3D test protocol | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-humanml3d-official) | [Evaluator Card](docs/evaluator_zoo/humanml3d_official.md) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) / [Code](https://github.com/EricGuo5513/text-to-motion) |
| MotionStreamer Evaluator | Cross-representation semantic evaluator for SMPL-aligned T2M results | MotionStreamer-272 | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-motionstreamer-272) | [Evaluator Card](docs/evaluator_zoo/motionstreamer.md) | [Paper](https://arxiv.org/abs/2503.15451) / [Code](https://github.com/zju3dv/MotionStreamer) |
| Motius Joint-Position Evaluator | Motius-trained TMR reproduction for unified SMPL-22 joint positions | SMPL-22 joints66 | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66) | [Evaluator Card](docs/evaluator_zoo/motius_joint_position.md) | [TMR Paper](https://arxiv.org/abs/2305.00976) / [TMR Code](https://github.com/Mathux/TMR) |
| Motius TMR-G1 Evaluator | Robot-native text-motion evaluator for Unitree G1 generation | G1-38D | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-g1-38d-tmr) | [Evaluator Card](docs/evaluator_zoo/g1_tmr.md) | [TMR Paper](https://arxiv.org/abs/2305.00976) / [TMR Code](https://github.com/Mathux/TMR) |

## Motion Representation Toolkit

Motius provides first-class support for the motion representations used by
different model families and makes them interoperable through a shared
**SMPL-22 body-motion bridge**. A source representation is converted to SMPL
`motion135` (root translation plus 22 local joint rotations), then encoded into
the representation required by the target model, evaluator, or renderer.

| Representation | Shape | Used by | Relationship to the SMPL bridge |
| -------------- | ----: | ------- | --------------------------------- |
| **SMPL-22 `motion135`** | `(T, 135)` | Canonical interchange, FK, mesh rendering | Central bridge: translation + 22 local 6D rotations |
| **HumanML3D-263** | `(T, 263)` | HumanML3D-based T2M models | Native decode plus official SMPL-22 joint encoder |
| **MotionStreamer-272** | `(T, 272)` | MotionStreamer and MotionMillion | Converts to and from SMPL-22 motion |
| **HY-Motion-201** | `(T, 201)` | HY-Motion models | Contains `motion135` as an exact prefix plus 22 joint positions |
| **DART276** | `(T, 276)` | DART and ViMoGen | Bridges through SMPL parameters and joints with explicit coordinate conversion |
| **Unitree G1-38D** | `(T, 38)` | G1-native generation and evaluation | SMPL body motion is retargeted through GMR; G1 qpos decode is exact |

### Same-Motion Representation Demo

The preview below uses one official HumanML3D test motion for every panel. The
left panel decodes HumanML3D-263 to SMPL-22 joints, the center renders the
skinned SMPL surface, and the right animates the Unitree G1 MJCF visual meshes
after GMR retargeting. All three are aligned to the same initial body heading.

![HumanML3D, SMPL, and Unitree G1 representation comparison](assets/motion/representation_demo/004822_hml_smpl_g1.gif)

[Open the synchronized Three.js viewer](assets/motion/representation_demo/index.html)
or read the [representation protocol](docs/motion/representations.md).

The shared bridge lets a model trained with one representation feed evaluators,
visualizers, or pipelines built for another. Conversion is exact where the
source preserves the required SMPL state; position-only recovery uses IK and is
necessarily lossy.

The generic API lives at
[`convert_motion`](motius/motion/representation/convert.py), with a matching
[`tools/convert_motion.py`](tools/convert_motion.py) CLI.

SMPL-parameter routes require locally licensed body-model files. Follow the
[SMPL body-model setup](#smpl-body-model-setup) before using these routes.

```python
from motius.motion import convert_motion, smpl_to_humanml263

# HY-Motion-201 -> SMPL-22 bridge -> MotionStreamer-272
smpl_motion = convert_motion(motion_hy201, "hymotion201", "motion135")
motion_ms272 = convert_motion(smpl_motion, "motion135", "ms272")

# Shape-aware SMPL-H -> official HumanML3D-263.
motion_hml263 = smpl_to_humanml263(
    global_orient,
    body_pose,
    transl,
    betas=betas,
    gender="female",
    model_type="smplh",
    model_path="checkpoints/smpl_models",
    src_fps=20,
    coordinate_system="amass",
)
```

```bash
python tools/convert_motion.py input.npy output.npy \
  --src hymotion201 --dst ms272
```

### SMPL Body-Model Setup

`model_path` is a local filesystem path, not a remote URL. SMPL+H parameters
cannot be redistributed with Motius, so download them from the
[official MANO / SMPL+H download page](https://mano.is.tue.mpg.de/download.php):

1. Register or sign in and accept the model license. Redirecting to the sign-in
   page before authentication is expected.
2. In **Downloads**, download **Extended SMPL+H model** for the genders you
   need. The separate MANO hand package is not required for Motius's SMPL-22
   joint conversion.
3. Extract the archive and arrange the files in either supported layout:

```text
checkpoints/smpl_models/
└── smplh/
    ├── female/model.npz
    ├── male/model.npz
    └── neutral/model.npz       # if downloaded
```

The standard `smplx` layout is also accepted:

```text
checkpoints/smpl_models/
└── smplh/
    ├── SMPLH_FEMALE.pkl
    ├── SMPLH_MALE.pkl
    └── SMPLH_NEUTRAL.pkl       # if available
```

Pass the directory root as `model_path="checkpoints/smpl_models"`, or pass one model
file directly. Verify the installation before conversion:

```bash
python - <<'PY'
from motius.motion.skeleton import resolve_smpl_model_path

path = resolve_smpl_model_path(
    "checkpoints/smpl_models", model_type="smplh", gender="female"
)
print(path)
PY
```

The printed path must be the downloaded female SMPL+H file. Select the same
`gender` used by the source motion; `betas` are evaluated against that model's
shape space. Keep these licensed files out of Git.

See the [representation reference](docs/motion/representations.md),
[conversion guide](docs/motion/conversion.md), and
[retargeting guide](docs/motion/retargeting.md) for channel layouts, 6D rotation
conventions, FPS behavior, required assets, and lossiness guarantees. SOMA and
Unitree G1 are documented separately as retargeting targets rather than body
representation interchange formats.

## What Is Included

| Area | Purpose |
| ---- | ------- |
| `motius.registry` | Central registries for models, bundles, trainers, pipelines, datasets, hooks, evaluators, and visualizers. |
| `motius.models` | `ModelBundle` abstraction and model utility functions. |
| `motius.trainers` | Reusable trainer base classes for method-specific training logic. |
| `motius.pipelines` | Pipeline base classes for inference and task-facing APIs. |
| `motius.runner` | Accelerate-based distributed training runner and train loops. |
| `motius.datasets` | Dataset bases and reusable transform primitives. |
| `motius.hooks` | Checkpoint, EMA, logging, and learning-rate scheduler hooks. |
| `motius.evaluation` | Evaluator base interfaces. |
| `motius.motion` | Representation specs/converters, SMPL-22 FK, and optional SOMA/G1 retargeting. |
| `motius.visualization` | File and TensorBoard visualization bases. |
| `configs/_base_` | Minimal runtime config templates. |
| `tools/` | Command-line training entry points. |

## Quick Start

```bash
python -m pip install -e ".[dev]"
```

Run a lightweight import and registration check:

```bash
python - <<'PY'
import motius

motius.register_all_modules()
print("Motius core import OK")
PY
```

Run the current smoke tests:

```bash
pytest -q
```

## Documentation

The detailed architecture, extension points, and package conventions live in
the formal documentation:

- [Architecture](docs/architecture.md)
- [Getting Started](docs/getting_started.md)
- [Development Guide](docs/development.md)
- [Motion representations and retargeting](docs/motion/README.md)

## Release Status

Motius is an early public release. APIs may still change while research-specific
method code is separated from reusable framework code. New methods will be
added through scoped Model Zoo entries and model cards.
