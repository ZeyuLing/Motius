# SONIC Attribution

Motius provides an independently implemented ONNX wrapper for the official
SONIC release. The method, model weights, observation contract, and deployment
documentation come from
[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
and [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC).

The redistributed checkpoint remains subject to the NVIDIA Open Model License
included in the Motius Hugging Face artifact. The Motius wrapper does not
import or require an upstream checkout.

For training, Motius vendors the Apache-2.0 `gear_sonic` source at commit
`4141c34280abb67c82e115342a8720f4a83d750d` under
`motius/trainers/sonic/vendor`. The source license and original notices are
preserved there. Isaac Lab is a separately installed simulator dependency.
