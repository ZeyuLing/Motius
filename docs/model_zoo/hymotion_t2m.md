<h1 align="center">HY-Motion T2M Model Card</h1>

<p align="center">
  <strong>Large-scale flow-matching text-to-motion generation, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.23464">Paper</a> |
  <a href="https://hunyuan.tencent.com/motion">Project Page</a> |
  <a href="https://github.com/Tencent-Hunyuan/HY-Motion-1.0">Original GitHub</a> |
  <a href="https://huggingface.co/tencent/HY-Motion-1.0">Original Weights</a>
</p>

HY-Motion T2M is Tencent Hunyuan's billion-parameter flow-matching text-to-3D
motion model. This Motius release packages the MMDiT motion transformer,
classifier-free guidance embeddings, normalization statistics, frozen Qwen3 /
CLIP-L text encoders, official smoothing, and an ODE-based inference pipeline.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | HY-Motion 1.0 T2M, DiT + flow matching |
| Tasks | Text-to-Motion |
| Motion representation | HY-Motion-201 at 30 fps |
| Text encoder | Qwen3-8B token context + CLIP-L sentence embedding |
| Pipeline | `motius.pipelines.hymotion_t2m.HyMotionT2MPipeline` |

Processed checkpoints:

| Variant | Checkpoint | Contents |
| ------- | ---------- | -------- |
| Full | [`ZeyuLing/hftrainer-hymotion-t2m-1.0`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0) | motion transformer, mean/std, text encoder tree |
| Lite | [`ZeyuLing/hftrainer-hymotion-t2m-1.0-lite`](https://huggingface.co/ZeyuLing/hftrainer-hymotion-t2m-1.0-lite) | same artifact layout |

## Usage

```python
from motius.pipelines.hymotion_t2m import HyMotionT2MPipeline

pipe = HyMotionT2MPipeline.from_pretrained(
    "ZeyuLing/hftrainer-hymotion-t2m-1.0-lite",
    device="cuda",
)

out = pipe({
    "caption": ["a person practices tai chi with slow controlled movements"],
    "num_frames": [180],
})

motion_201 = out["latent"]
keypoints3d = out.get("keypoints3d")
```

The pipeline pads inference to the 360-frame training length, integrates the
flow-matching ODE, truncates to the requested length, and applies the official
temporal smoothing by default.

## Evaluation Results

Protocol: HumanML3D official-test selected-caption protocol. For FID and
MM-Dist, lower is better.

| Evaluator | Variant | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Full | 0.561 | 0.761 | 0.853 | 0.103 | 2.532 | 10.031 | Measured |
| MotionStreamer Evaluator | Full | 0.737 | 0.881 | 0.929 | 16.021 | 14.789 | 27.187 | Measured |
| Motius Joint-Position Evaluator | Full | - | - | - | - | - | - | Pending |

Ground-truth sanity rows:

| Evaluator | Variant | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------- | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | GT real | 0.522 | 0.725 | 0.823 | 0.000 | 2.691 | 9.876 |
| MotionStreamer Evaluator | GT real | 0.706 | 0.857 | 0.911 | 0.000 | 15.007 | 27.367 |

The Lite checkpoint follows the same artifact and inference path; its public
metric row will be refreshed after the render/eval assets are regenerated.

## Motion Representation

HY-Motion T2M has a single public motion representation in this release:
`HY-Motion-201` at 30 fps. The generated tensor is returned as
`out["latent"]`.

The pipeline may expose decoded helper tensors such as `rot6d`, `transl`, or
`keypoints3d` for visualization/evaluation adapters, but those helpers are not
separate HY-Motion checkpoint variants and should not be listed as the model's
motion representation.

## Qualitative Results

Validated SMPL previews will be added after the public render assets are
rebuilt through the shared SMPL-H visualization path. This card avoids shipping
unverified preview media.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.hymotion_t2m.HyMotionT2MPipeline` |
| Bundle | `motius.models.hymotion_t2m.HyMotionT2MBundle` |
| Runtime | `motius.models.hymotion_t2m.network` |

## Citation

```bibtex
@article{wen2025hymotion,
  title={HY-Motion 1.0: Scaling Flow Matching Models for Text-To-Motion Generation},
  author={Wen, Yuxin and Shuai, Qing and Kang, Di and Li, Jing and Wen, Cheng and Qian, Yue and Jiao, Ningxin and Chen, Changhai and Chen, Weijie and Wang, Yiran and Guo, Jinkun and An, Dongyue and Liu, Han and Tong, Yanyu and Zhang, Chao and Guo, Qing and Chen, Juan and Zhang, Qiao and Zhang, Youyi and Yao, Zihao and Zhang, Cheng and Duan, Hong and Wu, Xiaoping and Chen, Qi and Cheng, Fei and Dong, Liang and He, Peng and Zhang, Hao and Lin, Jiaxin and Zhang, Chao and Fan, Zhongyi and Li, Yifan and Hu, Zhichao and Liu, Yuhong and Linus and Jiang, Jie and Li, Xiaolong and Bao, Linchao},
  journal={arXiv preprint arXiv:2512.23464},
  year={2025}
}
```
