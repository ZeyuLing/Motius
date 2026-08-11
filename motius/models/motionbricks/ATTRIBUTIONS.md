# MotionBricks Attribution

This package vendors the Apache-2.0 MotionBricks runtime from:

- Repository: https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks
- Paper: MotionBricks: Scalable Real-Time Motions with Modular Latent Generative Model and Smart Primitives
- Authors: Tingwu Wang, Olivier Dionne, Michael De Ruyter, David Minor, Davis Rempe, Kaifeng Zhao, Mathis Petrovich, Ye Yuan, Chenran Li, Zhengyi Luo, Brian Robison, Xavier Blackwell, Bernardo Antoniazzi, Xue Bin Peng, Yuke Zhu, Simon Yuen

Motius keeps the source code under `motius.models.motionbricks.network`,
rewrites imports into the Motius namespace, and wraps the official G1 runtime in
`MotionBricksBundle` / `MotionBricksPipeline`. Pretrained weights are not
vendored; users should fetch the official Git LFS checkpoints separately.
