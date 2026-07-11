<h1 align="center">KIMODO Model Card</h1>

<p align="center">
  <strong>Controllable human and humanoid motion generation through text and kinematic constraints.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.15546">Paper</a> |
  <a href="https://research.nvidia.com/labs/sil/projects/kimodo/">Project Page</a> |
  <a href="https://github.com/nv-tlabs/kimodo">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp">SOMA-RP Checkpoint</a>
</p>

KIMODO is NVIDIA's kinematic motion diffusion model for text-driven and
constraint-driven motion authoring. This Motius release packages the native
KIMODO runtime, skeleton assets, motion-representation utilities, and a unified
pipeline facade for text-to-motion, multi-prompt transitions, full-body
keyframes, end-effector controls, root paths, and prefix-conditioned TP2M.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | KIMODO, two-stage kinematic motion diffusion |
| Tasks | Text-to-Motion, multi-prompt stitching, kinematic control, TP2M |
| Motion representations | SOMA, Unitree G1, SMPL-X, plus `motion_135` TP2M bridge |
| Text encoder | LLM2Vec / Meta-Llama-3 local encoder tree |
| Default model | `Kimodo-SOMA-RP-v1` |
| Pipeline | `motius.pipelines.kimodo.KIMODOPipeline` |

Processed checkpoints:

| Variant | Native Skeleton | Checkpoint |
| ------- | --------------- | ---------- |
| SOMA-RP | SOMA | [`ZeyuLing/hftrainer-kimodo-soma-rp`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-soma-rp) |
| G1-RP | Unitree G1 | [`ZeyuLing/hftrainer-kimodo-g1-rp`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp) |
| G1-SEED | Unitree G1 | [`ZeyuLing/hftrainer-kimodo-g1-seed`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed) |
| SMPLX-RP | SMPL-X | [`ZeyuLing/hftrainer-kimodo-smplx-rp`](https://huggingface.co/ZeyuLing/hftrainer-kimodo-smplx-rp) |

The G1 and SMPL-X artifacts reuse the shared text-encoder tree from the SOMA-RP
artifact to avoid duplicating large LLM2Vec / Meta-Llama files.

## Usage

```python
from motius.pipelines.kimodo import KIMODOPipeline

pipe = KIMODOPipeline.from_pretrained(
    "ZeyuLing/hftrainer-kimodo-soma-rp",
    device="cuda",
)

motion = pipe.text_to_motion(
    "a person walks forward and waves.",
    num_frames=150,
)
```

Use the same pipeline for constraint-based generation:

```python
root_path = pipe.root2d_constraint(
    frame_indices=[0, 30, 60, 90],
    smooth_root_2d=[[0.0, 0.0], [0.5, 0.2], [1.0, 0.2], [1.5, 0.0]],
)

motion = pipe.constrained_motion(
    "a person follows a curved walking path",
    num_frames=120,
    constraints=[root_path],
)
```

TP2M takes a `motion_135` prefix motion and returns native KIMODO debug arrays
plus a generated `motion_135` bridge:

```python
samples = pipe.infer_tp2m(
    ["a person keeps walking forward"],
    [gt_motion_135],
    condition_frames=5,
)
```

## Evaluation Results

Protocol: HumanML3D official-test selected-caption protocol. KIMODO SMPL-X
output is converted through `SMPL-X -> SMPL motion_135 -> evaluator input`.
For FID and MM-Dist, lower is better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D-263 | 2,478 | 0.314 | 0.482 | 0.593 | 1.843 | 4.281 | 9.149 |
| MotionStreamer-272 | 7,392 | 0.323 | 0.460 | 0.541 | 143.917 | 21.707 | 25.316 |

## TP2M Results

Protocol: HumanML3D TP2M official-test selected-caption splits scored with
MotionStreamer-272. Each row uses the standard min/max length filter.

| Condition Frames | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ----------------: | ------: | --: | --: | --: | --: | ------: | --------: |
| 1 | 3,968 | 0.525 | 0.690 | 0.769 | 82.560 | 19.301 | 26.158 |
| 5 | 3,968 | 0.538 | 0.699 | 0.775 | 80.381 | 19.199 | 26.154 |
| 9 | 3,968 | 0.531 | 0.704 | 0.772 | 79.122 | 19.166 | 26.202 |

## Motion Representation

KIMODO operates in native skeleton spaces rather than HumanML3D-263. The Motius
runtime exposes native arrays such as `local_rot_mats`, `global_rot_mats`,
`posed_joints`, `root_positions`, `smooth_root_pos`, `foot_contacts`, and
`global_root_heading`.

For TP2M and cross-method evaluation, Motius also supports a `motion_135` bridge:
root translation `(3)` plus 22 local joint rotations in row-major 6D `(132)`.
The public pipeline uses the same row-major 6D convention when creating prefix
constraints and when exporting generated `motion_135`.

## Qualitative Results

Validated SMPL previews will be added after the public render pass is rebuilt
from the SMPL-X/SOMA-to-SMPL bridge. This card avoids shipping unverified render
assets and keeps the release focused on reproducible loading and metrics.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.kimodo.KIMODOPipeline` |
| Bundle | `motius.models.kimodo.KIMODOBundle` |
| Runtime | `motius.models.kimodo.network` |

## Citation

```bibtex
@article{rempe2026kimodo,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and Petrovich, Mathis and Yuan, Ye and Zhang, Haotian and Peng, Xue Bin and Jiang, Yifeng and Wang, Tingwu and Iqbal, Umar and Minor, David and de Ruyter, Michael and Li, Jiefeng and Tessler, Chen and Lim, Edy and Jeong, Eugene and Wu, Sam and Hassani, Ehsan and Huang, Michael and Yu, Jin-Bey and Chung, Chaeyeon and Song, Lina and Dionne, Olivier and Kautz, Jan and Yuen, Simon and Fidler, Sanja},
  journal={arXiv preprint arXiv:2603.15546},
  year={2026}
}
```
