<h1 align="center">MLD Model Card</h1>

<p align="center">
  <strong>Motion Latent Diffusion, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2212.04048">Paper</a> |
  <a href="https://chenfengye.github.io/motion-latent-diffusion/">Project Page</a> |
  <a href="https://github.com/ChenFengYe/motion-latent-diffusion">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d">Motius Checkpoint</a>
</p>

MLD is the text-to-motion baseline from *Executing Your Commands via Motion
Diffusion in Latent Space* (Chen et al., CVPR 2023). This Motius release
provides a native inference pipeline with the MLD motion VAE, latent diffusion
denoiser, DDIM scheduler, and frozen SentenceT5 text wrapper.

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![MLD HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mld/mld_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![MLD HumanML3D 004545 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mld/mld_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![MLD HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mld/mld_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | MLD, latent diffusion for human motion |
| Task | Text-to-Motion |
| Venue | CVPR 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | SentenceT5-Large, frozen |
| Default sampler | DDIM, 50 inference steps |
| Checkpoint | [`ZeyuLing/hftrainer-mld-humanml3d`](https://huggingface.co/ZeyuLing/hftrainer-mld-humanml3d) |
| Pipeline | `motius.pipelines.mld.MLDPipeline` |

The checkpoint artifact contains `vae.safetensors`, `denoiser.safetensors`,
`mld_config.json`, `Mean.npy`, and `Std.npy`. The SentenceT5 text encoder is
resolved by model name and is not duplicated inside the checkpoint artifact.

## Usage

Install the Motius package and the runtime dependencies used by the MLD stack:

```bash
python -m pip install -e ".[dev]"
```

Run text-to-motion inference:

```python
from motius.pipelines.mld import MLDPipeline

pipe = MLDPipeline.from_pretrained(
    "ZeyuLing/hftrainer-mld-humanml3d",
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
    num_inference_steps=50,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale.

## Evaluation Results

Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL/SMPL-H evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 4,042 | 0.518 | 0.716 | 0.816 | 0.297 | 2.950 | 9.628 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.566 | 0.733 | 0.810 | 39.744 | 19.337 | 24.902 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.517 | 0.685 | 0.770 | 258.621 | 36.345 | 57.346 | Measured |


## Motion Representation

MLD generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

MLD samples in latent space and decodes directly back to HumanML3D-263.
Conversion to SMPL or MotionStreamer-272 is only needed for
cross-representation evaluation.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mld.MLDPipeline` |
| Bundle | `motius.models.mld.MLDBundle` |
| Shared MLD/LCM runtime | `motius.models.motionlcm.network` |

The runtime is independent from the original checkout for inference. Raw
upstream checkpoint conversion remains outside this public release surface.

## Citation

```bibtex
@inproceedings{chen2023executing,
  title={Executing Your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
}
```
