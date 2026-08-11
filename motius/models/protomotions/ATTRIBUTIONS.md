# ProtoMotions Attribution

Motius provides an independently implemented ONNX wrapper for the G1
BONES-SEED deployment tracker released in
[NVlabs/ProtoMotions](https://github.com/NVlabs/ProtoMotions). The artifact
retains the official unified ONNX graph and deployment YAML without changing
their tensor order or control parameters.

ProtoMotions is distributed under Apache-2.0. The Motius artifact includes the
upstream license and does not import an upstream checkout at inference time.

For training, Motius vendors the Apache-2.0 source at commit
`49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c` under
`motius/trainers/protomotions/vendor`. The source license and third-party asset
notices are preserved there. Isaac Lab is a separately installed simulator
dependency.
