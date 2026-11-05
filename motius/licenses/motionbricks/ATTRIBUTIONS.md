# MotionBricks attribution

Source: NVIDIA NVlabs/GR00T-WholeBodyControl, pinned commit
45c03facbf848e0c743091b7d9c17952096ff908, motionbricks component.

The full upstream dual license notice is preserved in LICENSE. Source code is
Apache-2.0; dataset and checkpoint terms remain those of the original release.
The 60 robot mesh assets were verified against the pinned official Git LFS SHA256
identifiers. Motius keeps only network layers in models/motionbricks, and places
motion representations, data, training objectives and navigation under their
native framework modules. Independent source parity certificates state their
precise tested scope.

## Independent official release native integration

Source: NVlabs/motionbricks at 54fc563aeaaf819a753d72cc9fdc6713989cdd43,
standalone model definitions under tools/onnx/models. The source equations and
checkpoint parameter names are retained under Apache-2.0. Motius separates the
five learned roots from representation/rig code and inference policy; no upstream
runtime package or ONNX graph is used during native inference. Existing Motius
encoder/decoder/MLP/positional layers are reused. The original helper Predictor,
VQVAE wrapper, graph builders, application, editor and exporter package are not
vendored. Small g1_model_rig.npz and soma_model_rig.npz geometry constants retain
the upstream license. The checked source snapshot lives outside the repository
and is required only by the independent parity tool.

Root length selection is outside the neural module. Motion statistics and
feature conversions live under motion/representation; G1 kinematics live under
motion/body_models; generation and SOMA pose conversion live under pipelines.
Native SafeTensors retain official weights and all five neural roots.

Software notices: LICENSE-APACHE-2.0.txt, MotionGPT-LICENSE.txt and
vector-quantize-pytorch-LICENSE.txt. Model weights retain the NVIDIA Open Model
Agreement: https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/
Official sample motions have separate NVIDIA Sample Data Evaluation License
terms and are not redistributed in native checkpoint artifacts.

## Headless scene and pose runtime

Pose buffers, alignment, decoding, endpoint correction, SOMA IK, clip sampling,
Smart Primitive geometry, state transitions and target blending are adapted from
the same pinned NVlabs/motionbricks release, retaining Apache-2.0 attribution.
The component map is `docs/architecture/validation/motionbricks-runtime-source-map.json`.
Motius supplies its own PyTorch engine and headless scene orchestration; these
components are split across configuration, motion IO/runtime and pipelines.
They do not import the upstream application or runtime package. TensorRT, ONNX,
Newton, viewers, mesh renderers and editors are not required by native inference.
Warp and USD support are optional via the `motionbricks-runtime` extra.
Dependency notices include warp, usd-core, scipy and Daniel-Holden-MIT.
The separate scene downloader retains motion data licensing and verifies pinned
hashes; these sample motions are not redistributed in the Hugging Face weights.
