<h1 align="center">MotionLCM Model Card</h1>

<p align="center">
  <strong>Latent consistency motion generation, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.19759">Paper</a> |
  <a href="https://dai-wenxun.github.io/MotionLCM-page/">Project Page</a> |
  <a href="https://github.com/Dai-Wenxun/MotionLCM">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d">Motius Checkpoint</a>
</p>

MotionLCM is the real-time controllable motion generation method from
*MotionLCM: Real-time Controllable Motion Generation via Latent Consistency
Model* (Dai et al., ECCV 2024). This Motius release exposes the text-to-motion
path: SentenceT5 text features, latent consistency sampling in the MLD latent
space, MLD VAE decoding, and HumanML3D-263 denormalization.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MotionLCM, latent consistency model for human motion |
| Task | Text-to-Motion |
| Venue | ECCV 2024 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | SentenceT5-Large, frozen |
| Default sampler | LCM, 1 inference step |
| Checkpoint | [`ZeyuLing/hftrainer-motionlcm-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-motionlcm-humanml3d) |
| Pipeline | `motius.pipelines.motionlcm.MotionLCMPipeline` |

The checkpoint artifact contains `vae.safetensors`, `denoiser.safetensors`,
`motionlcm_config.json`, `Mean.npy`, and `Std.npy`. The SentenceT5 text encoder
is resolved by model name and is not duplicated inside the checkpoint artifact.

## Usage

```python
from motius.pipelines.motionlcm import MotionLCMPipeline

pipe = MotionLCMPipeline.from_pretrained(
    "ZeyuLing/hftrainer-motionlcm-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
    num_inference_steps=1,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale.

## Evaluation Results

Protocol: HumanML3D official test split, corrected official captions,
native 263-dim motion, one prediction per test id, NFE=1. For FID and MM-Dist,
lower is better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------: | ---: | ---: | ---: | ---: | ------: | --------: | ------ |
| HumanML3D Official | 4,042 | 0.509 | 0.708 | 0.811 | 0.340 | 2.969 | 9.641 | Measured |
| MotionStreamer Evaluator | 4,042 | 0.566 | 0.735 | 0.808 | 44.055 | 19.454 | 24.640 | Measured |
| Motius Joint-Position Evaluator | - | - | - | - | - | - | - | Pending |

## Motion Representation

MotionLCM generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

MotionLCM samples in the MLD latent space and decodes directly back to
HumanML3D-263. Conversion to SMPL or MotionStreamer-272 is only needed for
cross-representation evaluation.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionlcm.MotionLCMPipeline` |
| Bundle | `motius.models.motionlcm.MotionLCMBundle` |
| Shared MLD/LCM runtime | `motius.models.motionlcm.network` |

The released public surface covers the text-to-motion inference path. Raw
upstream checkpoint conversion and controllable MotionLCM variants remain
outside this scoped release.

## Citation

```bibtex
@inproceedings{dai2024motionlcm,
  title={MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model},
  author={Dai, Wenxun and Chen, Ling-Hao and Wang, Jingbo and Liu, Jinpeng and Dai, Bo and Tang, Yansong},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```
