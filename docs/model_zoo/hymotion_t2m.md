<h1 align="center">HY-Motion T2M Model Card</h1>

<p align="center">
  <strong>Large-scale flow-matching text-to-motion generation, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.23464">Paper</a> ·
  <a href="https://hunyuan.tencent.com/motion">Project Page</a> ·
  <a href="https://github.com/Tencent-Hunyuan/HY-Motion-1.0">Original GitHub</a> ·
  <a href="https://huggingface.co/tencent/HY-Motion-1.0">Original Weights</a>
</p>

HY-Motion T2M is Tencent Hunyuan's billion-parameter flow-matching text-to-3D
motion model. Motius packages both the human-motion checkpoints and the
G1-native 339k checkpoint with the MMDiT motion transformer,
classifier-free guidance embeddings, normalization statistics, frozen Qwen3 /
CLIP-L text encoders, official smoothing, and an ODE-based inference pipeline.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/df30bbaa-9d4e-4c5e-bcb4-1028b8dc4f3b" controls></video> | [MP4](https://github.com/user-attachments/assets/df30bbaa-9d4e-4c5e-bcb4-1028b8dc4f3b) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=hymotion1b) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/df30bbaa-9d4e-4c5e-bcb4-1028b8dc4f3b" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/1a9072c2-f8d3-4ee3-9ebf-ed79d776c755" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/f2953667-dee4-4b71-9bf4-c00d29e7f700" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps (HY-Motion-201 or G1-38, artifact-dependent) |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | HY-Motion 1.0 T2M, DiT + flow matching |
| Tasks | Text-to-Motion |
| Motion representation | HY-Motion-201 or G1-38 at 30 fps |
| Text encoder | Qwen3-8B token context + CLIP-L sentence embedding |
| Pipeline | `motius.pipelines.hymotion_t2m.HyMotionT2MPipeline` |

Processed checkpoints:

| Variant | Checkpoint | Contents |
| ------- | ---------- | -------- |
| Full | [`ZeyuLing/Motius-HYMotion-T2M-1.0`](https://huggingface.co/ZeyuLing/Motius-HYMotion-T2M-1.0) | motion transformer, mean/std, text encoder tree |
| Lite | [`ZeyuLing/Motius-HYMotion-T2M-1.0-Lite`](https://huggingface.co/ZeyuLing/Motius-HYMotion-T2M-1.0-Lite) | same artifact layout |
| G1 | [`ZeyuLing/Motius-HYMotion-G1`](https://huggingface.co/ZeyuLing/Motius-HYMotion-G1) | 339k transformer, G1-38 stats, CFG embeddings, text encoder tree |

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.hymotion_t2m.HyMotionT2MPipeline` |
| Bundle | `motius.models.hymotion_t2m.HyMotionT2MBundle` |
| Runtime | `motius.models.hymotion_t2m.network` |

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-HYMotion-T2M-1.0-Lite",
    device="cuda",
)

out = pipe.infer_text_to_motion(
    ["a person practices tai chi with slow controlled movements"],
    num_frames=[180],
)

motion_201 = out["latent"]
keypoints3d = out.get("keypoints3d")
```

The G1 checkpoint uses the same API and returns its native generative tensor
plus an exact MuJoCo state decode:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-HYMotion-G1",
    device="cuda",
)
out = pipe.infer_text_to_motion("a robot walks forward and waves", num_frames=180)

motion_g1_38 = out["g1_38"]
qpos_g1_36 = out["g1_qpos"]
```

The pipeline pads inference to the 360-frame training length, integrates the
flow-matching ODE, truncates to the requested length, and applies the official
temporal smoothing by default.

### Training

Motius provides a native HY-Motion T2M trainer and public
[configuration](../../configs/hymotion_t2m/train_hymotion_t2m.py). The recipe
trains the 201D flow-matching motion transformer and may either initialize it
from scratch or warm-start its motion weights with
`MOTIUS_PRETRAINED_WEIGHTS`. It does not import the original HY-Motion
repository.

The G1-native recipe is
[`train_hymotion_g1.py`](../../configs/hymotion_t2m/train_hymotion_g1.py). It
keeps the 0.46B architecture and trains directly in canonical `g1_38`, without
an SMPL round trip. Its manifest motion entries are `(T, 38)` arrays produced
with `motius.motion.representation.g1.encode_g1_motion`; point
`MOTIUS_MOTION_STATS` at statistics computed from those exact arrays.

Prepare a `ManifestTextMotionDataset` containing physical-scale
HY-Motion-201 arrays and cached Qwen3/CLIP text features. The manifest schema
and feature keys are documented in the
[training data-format guide](../training/prism_tmr_hymotion_t2m.md). Point
`MOTIUS_MOTION_STATS` to the matching 201D `Mean.npy` and `Std.npy`; changing
statistics changes the training space.

```bash
MOTIUS_DATA_ROOT=/path/to/hymotion201 \
MOTIUS_TRAIN_MANIFEST=train.json \
MOTIUS_MOTION_STATS=/path/to/hymotion201/stats \
bash tools/dist_train.sh configs/hymotion_t2m/train_hymotion_t2m.py 8 \
  --work-dir outputs/training/hymotion_t2m \
  --auto-resume
```

To warm-start only the network weights, set
`MOTIUS_PRETRAINED_WEIGHTS=/path/to/motion_transformer.safetensors`. For an
explicit full-state resume, use
`--load-from CHECKPOINT --load-scope full`.

| Training item | Released recipe |
| --- | --- |
| Precision | BF16 through Accelerate |
| Batch size | 8 per process |
| Optimizer | AdamW, learning rate `1e-4` |
| Objective | Masked Smooth L1 flow-velocity loss in normalized HY-Motion-201 space |
| Schedule | 100 epochs, gradient norm clipped to `1.0` |
| Checkpoints | Every epoch, latest state saved, five historical checkpoints retained |
| Outputs | `outputs/training/hymotion_t2m` |

See the [Training Hub](../training/README.md) for distributed launch, effective
batch size, output layout, and resume semantics.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Text-to-Motion · Unitree G1 | [Published results](../leaderboards/hf_space_t2m_unitree_g1/g1_results.json) | Fixed 1,024-case G1 protocol with TMR-G1 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Full | 3,970 | 0.561 | 0.761 | 0.853 | 0.103 | 2.532 | 10.031 | Measured |
| MotionStreamer Evaluator | Full | 4,042 | 0.7899 | 0.9177 | 0.9516 | 13.5194 | 14.7748 | 27.6694 | Measured |
| Motius Joint-Position Evaluator | Full | 4,034 | 0.7158 | 0.8600 | 0.9122 | 0.0147 | 27.1607 | 54.0940 | Measured |
| HumanML3D Official | Lite | 3,970 | 0.488 | 0.674 | 0.772 | 0.085 | 3.179 | 9.539 | Measured |
| MotionStreamer Evaluator | Lite | 4,042 | 0.7877 | 0.9100 | 0.9487 | 10.2711 | 14.8351 | 27.5298 | Measured |
| Motius Joint-Position Evaluator | Lite | 4,034 | 0.7126 | 0.8607 | 0.9143 | 0.0157 | 27.3560 | 54.3186 | Measured |

G1 protocol: fixed 1,024-case split, retrieval group size 32, and normalized
TMR-G1 embedding-space FID.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| Motius TMR-G1 | G1 | 1,024 | 0.7086 | 0.8380 | 0.8935 | 0.0587 | 20.5128 | 36.0106 | Measured |

## Motion Representation

The Full and Lite human checkpoints generate `HY-Motion-201` at 30 fps. The G1
checkpoint generates canonical `g1_38` at 30 fps: root XY velocity and height,
row-convention root rotation 6D, and 29 Unitree G1 joint angles. It decodes
exactly to MuJoCo `qpos-36`; no SMPL retargeting or IK is involved.

Human artifacts expose `rot6d`, `transl`, and optional `keypoints3d` helpers.
The G1 artifact exposes `g1_38` and `g1_qpos`. These decoded views do not change
the artifact's native representation.

## Citation and License

```bibtex
@article{wen2025hymotion,
  title={HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation},
  author={Wen, Yuxin and Shuai, Qing and Kang, Di and Li, Jing and Wen, Cheng and Qian, Yue and Jiao, Ningxin and Chen, Changhai and Chen, Weijie and Wang, Yiran and Guo, Jinkun and An, Dongyue and Liu, Han and Tong, Yanyu and Zhang, Chao and Guo, Qing and Chen, Juan and Zhang, Qiao and Zhang, Youyi and Yao, Zihao and Zhang, Cheng and Duan, Hong and Wu, Xiaoping and Chen, Qi and Cheng, Fei and Dong, Liang and He, Peng and Zhang, Hao and Lin, Jiaxin and Zhang, Chao and Fan, Zhongyi and Li, Yifan and Hu, Zhichao and Liu, Yuhong and Linus and Jiang, Jie and Li, Xiaolong and Bao, Linchao},
  journal={arXiv preprint arXiv:2512.23464},
  year={2025}
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
