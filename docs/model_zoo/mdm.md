<h1 align="center">MDM Model Card</h1>

<p align="center">
  <strong>Human Motion Diffusion Model, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2209.14916">Paper</a> |
  <a href="https://openreview.net/forum?id=SJ1kSyO2jwu">OpenReview</a> |
  <a href="https://guytevet.github.io/mdm-page/">Project Page</a> |
  <a href="https://github.com/GuyTevet/motion-diffusion-model">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d">Motius Checkpoint</a>
</p>

MDM is the text-to-motion baseline from *Human Motion Diffusion Model* (Tevet
et al., ICLR 2023). This Motius release provides an inference pipeline, a
checkpoint-loading bundle, and the Gaussian diffusion sampler needed to run the
HumanML3D checkpoint with the same public task interface used by other Motius
methods.

## Preview

<p align="center">
  <img src="../../assets/model_zoo/mdm/mdm_humanml3d_001840_roundhouse_kick_smpl_mesh_1024_30fps.gif" width="64%" alt="HumanML3D MDM roundhouse-kick SMPL mesh demo">
</p>

<p align="center">
  <sub>1024px / 30fps GIF demo, HumanML3D test sample 001840: "someone executes a roundhouse kick with their left foot." MP4 source is kept under <code>assets/model_zoo/mdm/</code>.</sub>
</p>

<!-- GitHub README/model-card pages only render inline MP4 players for uploaded
     GitHub attachment URLs. Until those attachment URLs are available, cards use
     verified 30fps GIF demos and keep MP4 sources under assets/. -->

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MDM, classifier-free diffusion for human motion |
| Task | Text-to-Motion |
| Venue | ICLR 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Default guidance scale | `2.5` |
| Checkpoint | [`ZeyuLing/hftrainer-mdm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d) |
| Pipeline | `motius.pipelines.mdm.MDMPipeline` |

The checkpoint artifact contains `model.safetensors`, `mdm_config.json`,
`Mean.npy`, and `Std.npy`. The mean and standard-deviation files are the
HumanML3D training normalization statistics and are part of the checkpoint
contract.

## Usage

Install the Motius package and OpenAI CLIP:

```bash
python -m pip install -e ".[dev]"
python -m pip install git+https://github.com/openai/CLIP.git
```

Run text-to-motion inference:

```python
from motius.pipelines.mdm import MDMPipeline

pipe = MDMPipeline.from_pretrained(
    "ZeyuLing/hftrainer-mdm-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale.

## Evaluation Results

Protocol: HumanML3D official test split with selected captions unless stated
otherwise. For FID and MM-Dist, lower is better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | 1,985 | 0.411 | 0.589 | 0.701 | 1.374 | 3.680 | 8.652 | Measured |
| MotionStreamer Evaluator | - | - | - | - | - | - | - | Pending |
| Motius Joint-Position Evaluator | - | - | - | - | - | - | - | Pending |

Ground-truth sanity row for the HumanML3D Official evaluator:

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official GT | 1,985 | 0.523 | 0.710 | 0.804 | - | 2.939 | 9.207 |

## Motion Representation

MDM generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

For qualitative inspection and cross-representation evaluation, generated
HumanML3D-263 motions are retargeted to the repository SMPL motion format and
rendered as SMPL mesh videos.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mdm.MDMPipeline` |
| Bundle | `motius.models.mdm.MDMBundle` |
| Network | `motius.models.mdm.network.MDM` |
| Diffusion sampler | `motius.models.mdm.network.diffusion` |
| Collation helper | `motius.models.mdm.network.collate` |

The network and Gaussian diffusion sampler are vendored for inference parity
with the released MDM checkpoint. Training-only geometry losses are represented
by explicit stubs, so unsupported training paths fail clearly.

## Citation

```bibtex
@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
}
```
