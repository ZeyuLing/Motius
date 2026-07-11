<h1 align="center">T2M-GPT Model Card</h1>

<p align="center">
  <strong>Discrete text-to-motion generation, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2301.06052">Paper</a> |
  <a href="https://mael-zys.github.io/T2M-GPT/">Project Page</a> |
  <a href="https://github.com/Mael-zys/T2M-GPT">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d">Motius Checkpoint</a>
</p>

T2M-GPT is the text-to-motion baseline from *T2M-GPT: Generating Human Motion
from Textual Descriptions with Discrete Representations* (Zhang et al., CVPR
2023). This Motius release packages the VQ-VAE motion tokenizer, the
cross-conditional GPT sampler, CLIP text encoder loading, and HumanML3D-263
denormalization behind a consistent pipeline.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | T2M-GPT, autoregressive discrete motion tokens |
| Task | Text-to-Motion |
| Venue | CVPR 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Tokenizer | HumanVQVAE, 512-code codebook |
| Checkpoint | [`ZeyuLing/hftrainer-t2mgpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-t2mgpt-humanml3d) |
| Pipeline | `motius.pipelines.t2mgpt.T2MGPTPipeline` |

The checkpoint artifact contains `vq.safetensors`, `gpt.safetensors`,
`clip.safetensors`, `t2mgpt_config.json`, `Mean.npy`, and `Std.npy`.

## Usage

```python
from motius.pipelines.t2mgpt import T2MGPTPipeline

pipe = T2MGPTPipeline.from_pretrained(
    "ZeyuLing/hftrainer-t2mgpt-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale. If `lengths` is omitted, the GPT
decides sequence length through its EOS token.

## Evaluation Results

Protocol: HumanML3D official test split, native 263-dim motion, first caption,
model-chosen length. For FID and MM-Dist, lower is better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | ---: | ---: | ---: | ---: | ------: | --------: |
| HumanML3D-263 | 3,940 | 0.470 | 0.660 | 0.761 | 0.176 | 3.238 | 9.563 |
| MotionStreamer-272 | 4,042 | 0.552 | 0.706 | 0.779 | 25.491 | 19.091 | 25.595 |

## Motion Representation

T2M-GPT generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The VQ-VAE downsamples motion tokens by a factor of four frames. A 196-frame
motion therefore maps to at most 49 discrete tokens from a 512-entry codebook.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.t2mgpt.T2MGPTPipeline` |
| Bundle | `motius.models.t2mgpt.T2MGPTBundle` |
| Network | `motius.models.t2mgpt.network` |

The inference path intentionally keeps the GPT in training mode during token
sampling, matching the released T2M-GPT sampling distribution.

## Citation

```bibtex
@inproceedings{zhang2023t2m,
  title={T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations},
  author={Zhang, Jianrong and Zhang, Yangsong and Cun, Xiaodong and Huang, Yong and Zhang, Yong and Zhao, Hongwei and Lu, Hongtao and Shen, Xi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
