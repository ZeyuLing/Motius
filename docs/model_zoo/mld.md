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

Protocol: HumanML3D official test split, corrected official captions,
native 263-dim motion, one prediction per test id. For FID and MM-Dist, lower
is better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | ---: | ---: | ---: | ---: | ------: | --------: |
| HumanML3D-263 | 4,042 | 0.518 | 0.716 | 0.816 | 0.297 | 2.950 | 9.628 |
| MotionStreamer-272 | 4,042 | 0.566 | 0.733 | 0.810 | 39.744 | 19.337 | 24.902 |
| MotionCLIP-135 | 4,042 | 0.383 | 0.538 | 0.632 | 134.648 | 42.468 | 22.947 |

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
