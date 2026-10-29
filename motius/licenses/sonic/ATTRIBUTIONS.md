# SONIC Attribution

Motius provides native neural policy/value roots, a deployment Pipeline and
PPO/auxiliary online training, G1 physics/data components, alongside compatibility with the official
ONNX release. The method, model weights, observation contract, and deployment
documentation come from
[NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)
and [nvidia/GEAR-SONIC](https://huggingface.co/nvidia/GEAR-SONIC).

The redistributed checkpoint remains subject to the NVIDIA Open Model License
included in the Motius Hugging Face artifact. The Motius wrapper does not
import or require an upstream checkout.

The native implementation derives from Apache-2.0 `gear_sonic` source at commit
`4141c34280abb67c82e115342a8720f4a83d750d`. The full upstream license and
original notices are preserved in this directory. The source application, foreign trainer runtime, unused robots, rendering and
training utilities have been removed. The selected G1 dependency closure and
file hashes are recorded in `environment_source_map.json`; 53 pinned physics
assets live outside the Python package and are materialized using
`tools/download_sonic_assets.py`. Isaac Lab is installed separately.

Robot settings and rough terrain include adaptations from BeyondMimic and
Isaac Lab. Original software and asset notices are retained in the two
`THIRD-PARTY ... GEAR-SONIC.txt` files alongside the complete upstream license.
