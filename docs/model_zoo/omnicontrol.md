# OmniControl Model Card

OmniControl generates HumanML3D motion from text while controlling selected
3D joint positions at selected frames. The Motius integration vendors the
official MIT inference runtime and uses the released HumanML3D checkpoint.

## Native Control Contract

- Input motion representation: physical-scale HumanML3D-263.
- Control evidence: world-space XYZ positions for any subset of the 22 joints
  at any subset of frames.
- Temporal completion: select all joints at the required prefix, boundary, or
  keyframes.
- Root trajectory: select the pelvis across dense or sparse frames.
- Local joint rotations are not a native OmniControl control input.

## Usage

```python
from motius.pipelines.omnicontrol import OmniControlPipeline

pipe = OmniControlPipeline.from_pretrained(
    "/path/to/model_humanml3d.pt",
    device="cuda",
)
outputs = pipe.infer_control(
    captions=["a person walks forward"],
    motions=[ground_truth_hml263],
    control_mode="first_last",
)
```

The runtime caches the CLIP text embedding once per batch. Motion and spatial
normalization statistics are packaged with Motius so execution does not depend
on an external OmniControl checkout or a current working directory.

## Provenance

- Paper: [OmniControl: Control Any Joint at Any Time for Human Motion Generation](https://arxiv.org/abs/2310.08580)
- Official code: [neu-vi/OmniControl](https://github.com/neu-vi/OmniControl)
- Vendored license: `motius/models/omnicontrol/LICENSE`
