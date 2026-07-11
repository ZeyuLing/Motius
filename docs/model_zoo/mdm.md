# MDM Pipeline

MDM is the text-to-motion baseline from *Human Motion Diffusion Model*
(Tevet et al., ICLR 2023). Motius includes an inference-only MDM integration
with the HumanML3D-263 representation.

## Public API

```python
from motius.pipelines.mdm import MDMPipeline

pipe = MDMPipeline.from_pretrained("/path/to/mdm_artifact", device="cuda")
motions = pipe.infer_t2m(
    ["a person walks forward then sits down"],
    [120],
)
```

The return value is a list of NumPy arrays, one per caption. Each array has
shape `(T, 263)` and is denormalized to HumanML3D physical scale.

## Artifact Layout

`MDMBundle.from_pretrained()` expects a local directory or Hugging Face Hub
snapshot with:

```text
mdm_config.json
model.safetensors  # or model.pt
Mean.npy
Std.npy
```

`Mean.npy` and `Std.npy` are the HumanML3D training normalization statistics
used by the diffusion model. They are part of the model contract and should
travel with every released artifact.

## Runtime Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mdm.MDMPipeline` |
| Bundle | `motius.models.mdm.MDMBundle` |
| Network | `motius.models.mdm.network.MDM` |
| Diffusion sampler | `motius.models.mdm.network.diffusion` |
| Collation helper | `motius.models.mdm.network.collate` |

The network and Gaussian diffusion sampler are vendored for inference parity
with the released MDM checkpoint. Training-only geometry losses are represented
by explicit stubs, so unsupported training paths fail clearly instead of
pulling in private or upstream checkout dependencies.

## Dependencies

MDM inference uses the frozen CLIP ViT-B/32 text encoder. Install OpenAI CLIP
before loading an MDM artifact:

```bash
python -m pip install git+https://github.com/openai/CLIP.git
```

The core Motius import and registry tests do not instantiate CLIP or download
weights.

## Motion Representation

The public MDM pipeline generates HumanML3D-263 features at the model's native
20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

## Length Handling

MDM is trained with HumanML3D sequence lengths between 40 and 196 frames.
`MDMPipeline.clamp_length()` rounds requested lengths down to a multiple of 4
and clamps them to this range.
