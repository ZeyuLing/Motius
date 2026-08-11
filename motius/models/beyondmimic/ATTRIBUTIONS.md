# BeyondMimic Attribution

Motius provides an independently implemented loader for ONNX policies exported
by [HybridRobotics/whole_body_tracking](https://github.com/HybridRobotics/whole_body_tracking).
The input/output and metadata behavior follows the official
[motion tracking controller](https://github.com/HybridRobotics/motion_tracking_controller).
The MIT-licensed training source is vendored at commit
`cd65172032893724b445448818c34165846d847d`; Motius adds local asset, motion,
resume, and export adapters around the official Isaac Lab/RSL-RL loop.

The upstream code is MIT licensed. As of the integration audit,
the project does not publish a named pretrained policy checkpoint. Motius
therefore supports user-exported ONNX policies and does not label any third
party checkpoint as an official BeyondMimic release.
