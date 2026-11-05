# BeyondMimic source attribution

Native G1 tracking terms follow HybridRobotics/whole_body_tracking commit
`cd65172032893724b445448818c34165846d847d` (MIT).
The neural actor/critic and empirical normalization layers and PPO arithmetic
follow ETH Zurich/NVIDIA RSL-RL v2.3.1 (BSD-3-Clause). Simulator configuration
uses Isaac Lab v2.1.0 APIs (BSD-3-Clause). Complete notices accompany this file.

Motius owns the neural ModelBundle, PPO Trainer, online AccelerateRunner
composition, inference Pipeline, motion representation and simulator terms;
it does not bundle or launch an upstream training application.

The original project does not publish a named general-purpose pretrained
policy. Source comparisons with seeded networks certify the tested arithmetic,
not pretrained tracking quality or complete physical training reproduction.
