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

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![MotionLCM HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionlcm/motionlcm_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![MotionLCM HumanML3D 004545 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionlcm/motionlcm_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![MotionLCM HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/motionlcm/motionlcm_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MotionLCM, latent consistency model for human motion |
| Tasks | Text-to-Motion |
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

Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 4,042 | 0.509 | 0.708 | 0.811 | 0.340 | 2.969 | 9.641 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.566 | 0.735 | 0.807 | 44.055 | 19.454 | 24.640 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.516 | 0.692 | 0.774 | 283.053 | 36.587 | 56.946 | Measured |

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
