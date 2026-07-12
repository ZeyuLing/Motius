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
  <a href="docs/getting_started.md">Getting Started</a> |
  <a href="docs/architecture.md">Architecture</a> |
  <a href="docs/development.md">Development Guide</a>
</p>

Motius packages motion-generation methods as consistent model bundles,
trainers, pipelines, evaluators, and visualization utilities. The public repo
is being opened method by method: the reusable core is available now, and each
released method will ship with a model card, checkpoint path, evaluation
results, and qualitative SMPL renders.

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
| HY-Motion T2M | Text-to-Motion | HY-Motion-201 / SMPL motion_135 | [Full](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0) / [Lite](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) | [Model Card](docs/model_zoo/hymotion_t2m.md) | [Paper](https://arxiv.org/abs/2512.23464) / [Code](https://github.com/Tencent-Hunyuan/HY-Motion-1.0) |
| KIMODO | Text + Kinematic Control | SOMA / G1 / SMPL-X | [SOMA-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp) / [SMPLX-RP](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp) | [Model Card](docs/model_zoo/kimodo.md) | [Paper](https://arxiv.org/abs/2603.15546) / [Code](https://github.com/nv-tlabs/kimodo) |
| MLD | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) | [Model Card](docs/model_zoo/mld.md) | [Paper](https://arxiv.org/abs/2212.04048) / [Code](https://github.com/ChenFengYe/motion-latent-diffusion) |
| MotionLCM | Text-to-Motion | HumanML3D-263 | [HF](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) | [Model Card](docs/model_zoo/motionlcm.md) | [Paper](https://arxiv.org/abs/2404.19759) / [Code](https://github.com/Dai-Wenxun/MotionLCM) |
| ViMoGen | Text-to-Motion | DART276 | [HF](https://huggingface.co/ZeyuLing/hftrainer-vimogen-1.3b-humanml3d) | [Model Card](docs/model_zoo/vimogen.md) | [Paper](https://arxiv.org/abs/2510.26794) / [Code](https://github.com/MotrixLab/ViMoGen) |
| DART | Autoregressive Text-to-Motion / Motion Control | SMPL-H motion_135 | Pending HF artifact | [Model Card](docs/model_zoo/dart.md) | [Paper](https://arxiv.org/abs/2410.05260) / [Code](https://github.com/zkf1997/DART) |

## Evaluator Zoo

Motius model cards report text-to-motion metrics with three evaluator views:
HumanML3D official metrics, MotionStreamer Evaluator metrics, and the Motius
joint-position evaluator trained on unified SMPL-H joints. Historical
contrastive-evaluator rows are not part of the public Evaluation tables.

| Evaluator | Purpose | Motion Rep. | Checkpoint | Card | Reference |
| --------- | ------- | ----------- | ---------- | ---- | --------- |
| HumanML3D Official | Standard T2M leaderboard metrics on the selected-caption HumanML3D test protocol | HumanML3D-263 | Official evaluator assets | [Evaluator Card](docs/evaluator_zoo/humanml3d_official.md) | HumanML3D / T2M Leaderboard protocol |
| MotionStreamer Evaluator | Cross-representation semantic evaluator for SMPL-aligned T2M results | MotionStreamer-272 | Pending public artifact | [Evaluator Card](docs/evaluator_zoo/motionstreamer.md) | [Paper](https://arxiv.org/abs/2503.15451) / [Code](https://github.com/zju3dv/MotionStreamer) |
| Motius Joint-Position Evaluator | Source-fair evaluator trained on HYMotion Data + MotionHub with unified SMPL-H joint positions | SMPL-H joints66 | Pending public artifact | [Evaluator Card](docs/evaluator_zoo/motius_joint_position.md) | Motius universal evaluator |

### Preview Gallery

<p align="center">
  <img src="assets/model_zoo/mdm/mdm_humanml3d_001840_roundhouse_kick_smpl_mesh_1024_30fps.gif" width="56%" alt="MDM roundhouse-kick SMPL mesh demo">
</p>

<p align="center">
  <sub>1024px / 30fps GIF demo, MDM on HumanML3D test sample 001840: "someone executes a roundhouse kick with their left foot." MP4 source is kept under <code>assets/model_zoo/mdm/</code>.</sub>
</p>

<!-- GitHub README only renders inline MP4 players for uploaded GitHub attachment URLs.
     Until those attachment URLs are available, model cards use verified 30fps GIF demos
     and keep MP4 sources under assets/. -->

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

## Release Status

Motius is an early public release. APIs may still change while research-specific
method code is separated from reusable framework code. New methods will be
added through scoped Model Zoo entries and model cards.
