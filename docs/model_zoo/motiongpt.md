<h1 align="center">MotionGPT Model Card</h1>

<p align="center">
  <strong>Motion-language generation with discrete motion tokens, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2306.14795">Paper</a> |
  <a href="https://motion-gpt.github.io/">Project Page</a> |
  <a href="https://github.com/OpenMotionLab/MotionGPT">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d">Motius Checkpoint</a>
</p>

MotionGPT is the motion-language baseline from *MotionGPT: Human Motion as a
Foreign Language* (Jiang et al., NeurIPS 2023). This Motius release packages
the HumanML3D motion tokenizer, FLAN-T5-base-style language model with motion
tokens, HumanML3D statistics, and task-facing text-to-motion / motion-to-text
pipeline methods without requiring the original checkout.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MotionGPT, language modeling over text and motion tokens |
| Tasks | Text-to-Motion, Motion-to-Text |
| Venue | NeurIPS 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Language backbone | FLAN-T5-base-style encoder-decoder with motion tokens |
| Motion tokenizer | VQ-VAE, 512-code codebook |
| Checkpoint | [`ZeyuLing/hftrainer-motiongpt-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motiongpt-humanml3d) |
| Pipeline | `motius.pipelines.motiongpt.MotionGPTPipeline` |

The checkpoint artifact contains `motiongpt_s3_h3d.tar`,
`assets/meta/mean.npy`, `assets/meta/std.npy`, `deps/flan-t5-base/`, and
`model_index.json`.

## Usage

```python
from motius.pipelines.motiongpt import MotionGPTPipeline

pipe = MotionGPTPipeline.from_pretrained(
    "ZeyuLing/hftrainer-motiongpt-humanml3d",
    bundle_kwargs={"local_files_only": False},
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale. The same pipeline also exposes
`infer_m2t` for captioning denormalized HumanML3D-263 motions.

## Evaluation Results

Protocol: HumanML3D official-test caption protocol, HumanML3D-263 generation
converted through the shared SMPL/MotionStreamer path. For FID and MM-Dist,
lower is better.

| Evaluator | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | - | - | - | - | - | - | Pending |
| MotionStreamer Evaluator | 0.494 | 0.635 | 0.694 | 23.681 | 19.678 | 25.541 | Measured |
| Motius Joint-Position Evaluator | - | - | - | - | - | - | Pending |

Physical diagnostics:

| Slide | Float | Jitter | Dynamic |
| ----: | ----: | -----: | ------: |
| 3.878 | 10.884 | 5.168 | 21.061 |

## Motion Representation

MotionGPT generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The VQ-VAE converts normalized HumanML3D features into discrete motion tokens.
MotionGPT then treats those tokens as a language vocabulary item alongside text
tokens.

## Qualitative Results

Validated SMPL previews will be added to this card once the public qualitative
assets are rendered through the shared SMPL-H visualization path. The current
release keeps the model card focused on reproducible checkpoint loading and
numeric evaluation rather than shipping unverified preview media.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motiongpt.MotionGPTPipeline` |
| Bundle | `motius.models.motiongpt.MotionGPTBundle` |
| Runtime | `motius.models.motiongpt.network.mGPT.archs` |

Only the inference-time MotionGPT modules required by the bundle are included
in this public package.

## Citation

```bibtex
@inproceedings{jiang2023motiongpt,
  title={MotionGPT: Human Motion as a Foreign Language},
  author={Jiang, Biao and Chen, Xin and Liu, Wen and Yu, Jingyi and Yu, Gang and Chen, Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2023}
}
```
