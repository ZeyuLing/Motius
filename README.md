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
  <a href="docs/model_zoo/release_policy.md">Release Policy</a> |
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

| Method | Task | Motion Rep. | Checkpoint | Card | References |
| ------ | ---- | ----------- | ---------- | ---- | ---------- |
| MDM | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) | [Model Card](docs/model_zoo/mdm.md) | [Paper](https://arxiv.org/abs/2209.14916) / [Code](https://github.com/GuyTevet/motion-diffusion-model) |
| T2M-GPT | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) | [Model Card](docs/model_zoo/t2mgpt.md) | [Paper](https://arxiv.org/abs/2301.06052) / [Code](https://github.com/Mael-zys/T2M-GPT) |
| MoMask | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-momask-humanml3d) | [Model Card](docs/model_zoo/momask.md) | [Paper](https://arxiv.org/abs/2312.00063) / [Code](https://github.com/EricGuo5513/momask-codes) |
| MoGenTS | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mogents-humanml3d) | [Model Card](docs/model_zoo/mogents.md) | [Paper](https://arxiv.org/abs/2409.17686) / [Code](https://github.com/weihaosky/mogents) |
| MotionGPT | Text-to-Motion / Motion-to-Text | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d) | [Model Card](docs/model_zoo/motiongpt.md) | [Paper](https://arxiv.org/abs/2306.14795) / [Code](https://github.com/OpenMotionLab/MotionGPT) |
| FlowMDM | Text-to-Motion / Multi-Prompt T2M | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-flowmdm-humanml3d) | [Model Card](docs/model_zoo/flowmdm.md) | [Paper](https://arxiv.org/abs/2402.15509) / [Code](https://github.com/BarqueroGerman/FlowMDM) |
| MotionMillion | Zero-Shot Text-to-Motion | MotionStreamer-272 | [7B](https://huggingface.co/ZeyuLing/hftrainer-gotozero-7b-train-humanml272) / [3B](https://huggingface.co/ZeyuLing/hftrainer-gotozero-3b-train-humanml272) | [Model Card](docs/model_zoo/motionmillion.md) | [Paper](https://arxiv.org/abs/2507.07095) / [Code](https://github.com/VankouF/MotionMillion-Codes) |
| MotionStreamer | Streaming Text-to-Motion / TP2M | MotionStreamer-272 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motionstreamer-humanml272) | [Model Card](docs/model_zoo/motionstreamer.md) | [Paper](https://arxiv.org/abs/2503.15451) / [Code](https://github.com/zju3dv/MotionStreamer) |
| HY-Motion T2M | Text-to-Motion | HY-Motion-201 | [Full](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0) / [Lite](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) | [Model Card](docs/model_zoo/hymotion_t2m.md) | [Paper](https://arxiv.org/abs/2512.23464) / [Code](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) |
| KIMODO | Text + Kinematic Control | SOMA / G1 / SMPL-X | [SOMA-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp) / [G1-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp) / [G1-SEED](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed) / [SMPLX-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp) | [Model Card](docs/model_zoo/kimodo.md) | [Paper](https://arxiv.org/abs/2603.15546) / [Code](https://github.com/nv-tlabs/kimodo) |
| MLD | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) | [Model Card](docs/model_zoo/mld.md) | [Paper](https://arxiv.org/abs/2212.04048) / [Code](https://github.com/ChenFengYe/motion-latent-diffusion) |
| MotionLCM | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) | [Model Card](docs/model_zoo/motionlcm.md) | [Paper](https://arxiv.org/abs/2404.19759) / [Code](https://github.com/Dai-Wenxun/MotionLCM) |
| ViMoGen | Text-to-Motion | DART276 | [HF](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) | [Model Card](docs/model_zoo/vimogen.md) | [Paper](https://arxiv.org/abs/2510.26794) / [Code](https://github.com/MotrixLab/ViMoGen) |
| DART | Autoregressive Text-to-Motion / Motion Control | DART276 | [HF](https://huggingface.co/ZeyuLing/motius-dart-humanml3d) | [Model Card](docs/model_zoo/dart.md) | [Paper](https://arxiv.org/abs/2410.05260) / [Code](https://github.com/zkf1997/DART) |

## Evaluator Zoo

Motius model cards report text-to-motion metrics with three evaluator views:
HumanML3D official metrics, MotionStreamer Evaluator metrics, and the Motius
joint-position evaluator trained on unified SMPL-22 joints. Historical
contrastive-evaluator rows are not part of the public Evaluation tables.

| Evaluator | Purpose | Motion Rep. | Checkpoint | Card | Reference |
| --------- | ------- | ----------- | ---------- | ---- | --------- |
| HumanML3D Official | Standard T2M leaderboard metrics on the selected-caption HumanML3D test protocol | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-humanml3d-official) | [Evaluator Card](docs/evaluator_zoo/humanml3d_official.md) | [Paper](https://openaccess.thecvf.com/content/CVPR2022/html/Guo_Generating_Diverse_and_Natural_3D_Human_Motions_From_Text_CVPR_2022_paper.html) / [Code](https://github.com/EricGuo5513/text-to-motion) |
| MotionStreamer Evaluator | Cross-representation semantic evaluator for SMPL-aligned T2M results | MotionStreamer-272 | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-motionstreamer-272) | [Evaluator Card](docs/evaluator_zoo/motionstreamer.md) | [Paper](https://arxiv.org/abs/2503.15451) / [Code](https://github.com/zju3dv/MotionStreamer) |
| Motius Joint-Position Evaluator | Motius-trained TMR reproduction for unified SMPL-22 joint positions | SMPL-22 joints66 | [HF](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66) | [Evaluator Card](docs/evaluator_zoo/motius_joint_position.md) | [TMR Paper](https://arxiv.org/abs/2305.00976) / [TMR Code](https://github.com/Mathux/TMR) |

## Motion Representation Toolkit

Motius exposes representation metadata, deterministic decoders, skeleton-aware
converters, SMPL-22 forward kinematics, and optional SOMA/G1 retargeting under
[`motius.motion`](motius/motion). The generic conversion API is
[`convert_motion`](motius/motion/representation/convert.py); the same routes are
available from [`tools/convert_motion.py`](tools/convert_motion.py).

```python
from motius.motion.representation.convert import convert_motion

joints = convert_motion(motion_hml263, "hml263", "joints")
motion135 = convert_motion(motion_hy201, "hymotion201", "motion135")
motion272 = convert_motion(motion135, "motion135", "ms272")
```

```bash
python tools/convert_motion.py input.npy output.npy \
  --src hml263 --dst joints
```

| Source | Public targets | Conversion note |
| ------ | -------------- | --------------- |
| HumanML3D-263 | joints, `motion135`, MS272 | SMPL outputs use IK and are lossy |
| MotionStreamer-272 | joints, `motion135` | Native joints; SMPL subject shape is not retained |
| HY-Motion-201 | joints, `motion135` | Stored joints and exact 135-d prefix |
| `motion135` | joints, HY-Motion-201, MS272 | FK routes require the target skeleton or bone offsets |
| DART276 | joints, `motion135`, MS272 | Explicit DART/MBench coordinate conversion |
| G1-38 | MuJoCo qpos-36 | Exact root-pose and 29-DOF decode |

See the [representation reference](docs/motion/representations.md),
[conversion guide](docs/motion/conversion.md), and
[retargeting guide](docs/motion/retargeting.md) for channel layouts, 6D rotation
conventions, FPS behavior, required assets, and lossiness guarantees.

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
