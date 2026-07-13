# T2M HumanML3D Leaderboard

[Back to the Motius Model Zoo](../../README.md#model-zoo)

This leaderboard is generated from the measured rows in
[`release_manifest.json`](../model_zoo/release_manifest.json). It reports
the three evaluator views required by every public Motius T2M model card.
Compare methods only within the same evaluator table; each evaluator has
its own motion representation, embedding space, and protocol.

R@1, R@2, R@3, and Diversity are higher-is-better. FID and MM-Dist are
lower-is-better.

## [HumanML3D Official](../evaluator_zoo/humanml3d_official.md)

| Method | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ------ | ------- | ------: | --: | --: | --: | --: | ------: | --------: |
| [MDM](../model_zoo/mdm.md) | Default | 3,970 | 0.4618 | 0.6602 | 0.7631 | 0.4000 | 3.2480 | 9.9660 |
| [T2M-GPT](../model_zoo/t2mgpt.md) | Default | 3,944 | 0.4898 | 0.6784 | 0.7758 | 0.2249 | 3.1450 | 9.6245 |
| [MoMask](../model_zoo/momask.md) | Default | 3,970 | 0.5160 | 0.7090 | 0.8040 | 0.0970 | 2.9900 | 9.4600 |
| [MoGenTS](../model_zoo/mogents.md) | Default | 3,970 | 0.5220 | 0.7130 | 0.8060 | 0.0810 | 2.9290 | 9.4060 |
| [MotionGPT](../model_zoo/motiongpt.md) | Default | 3,962 | 0.4341 | 0.5999 | 0.6857 | 0.1557 | 3.9195 | 9.7471 |
| [FlowMDM](../model_zoo/flowmdm.md) | Default | 3,970 | 0.4388 | 0.6357 | 0.7443 | 0.3274 | 3.3868 | 9.9419 |
| [MotionMillion](../model_zoo/motionmillion.md) | 7B train-only | 3,970 | 0.5232 | 0.7206 | 0.8174 | 0.0649 | 2.8972 | 9.3944 |
| [MotionMillion](../model_zoo/motionmillion.md) | 3B train-only | 3,970 | 0.5282 | 0.7234 | 0.8178 | 0.0710 | 2.8816 | 9.3789 |
| [MotionStreamer](../model_zoo/motionstreamer.md) | Default | 3,970 | 0.4085 | 0.5880 | 0.6899 | 0.1688 | 3.6764 | 9.5792 |
| [HY-Motion T2M](../model_zoo/hymotion_t2m.md) | Full | 3,970 | 0.5610 | 0.7610 | 0.8530 | 0.1030 | 2.5320 | 10.0310 |
| [HY-Motion T2M](../model_zoo/hymotion_t2m.md) | Lite | 3,970 | 0.4881 | 0.6737 | 0.7718 | 0.0847 | 3.1795 | 9.5394 |
| [KIMODO](../model_zoo/kimodo.md) | SMPL-X RP | 3,970 | 0.2923 | 0.4521 | 0.5584 | 1.5198 | 4.5731 | 8.8750 |
| [MLD](../model_zoo/mld.md) | Default | 4,042 | 0.5180 | 0.7160 | 0.8160 | 0.2970 | 2.9500 | 9.6280 |
| [MotionLCM](../model_zoo/motionlcm.md) | Default | 4,042 | 0.5090 | 0.7080 | 0.8110 | 0.3400 | 2.9690 | 9.6410 |
| [ViMoGen](../model_zoo/vimogen.md) | 1.3B prompt-rewrite | 3,970 | 0.2832 | 0.4383 | 0.5471 | 8.3707 | 4.8935 | 6.7088 |
| [DART](../model_zoo/dart.md) | Default | 3,970 | 0.4007 | 0.5916 | 0.7001 | 1.8464 | 3.7089 | 9.8674 |

## [MotionStreamer Evaluator](../evaluator_zoo/motionstreamer.md)

| Method | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ------ | ------- | ------: | --: | --: | --: | --: | ------: | --------: |
| [MDM](../model_zoo/mdm.md) | Default | 4,042 | 0.5208 | 0.6937 | 0.7701 | 35.5169 | 19.4246 | 25.3383 |
| [T2M-GPT](../model_zoo/t2mgpt.md) | Default | 4,042 | 0.5516 | 0.7056 | 0.7788 | 25.4913 | 19.0912 | 25.5949 |
| [MoMask](../model_zoo/momask.md) | Default | 4,042 | 0.6404 | 0.7974 | 0.8609 | 21.0729 | 18.1216 | 25.9789 |
| [MoGenTS](../model_zoo/mogents.md) | Default | 4,042 | 0.4993 | 0.6520 | 0.7354 | 20.1861 | 19.5354 | 25.6972 |
| [MotionGPT](../model_zoo/motiongpt.md) | Default | 4,042 | 0.4940 | 0.6352 | 0.6944 | 23.6811 | 19.6781 | 25.5410 |
| [FlowMDM](../model_zoo/flowmdm.md) | Default | 4,042 | 0.4737 | 0.6496 | 0.7312 | 36.3767 | 20.0018 | 25.1783 |
| [MotionMillion](../model_zoo/motionmillion.md) | 7B train-only | 4,042 | 0.7403 | 0.8777 | 0.9236 | 3.0807 | 15.3706 | 27.5748 |
| [MotionMillion](../model_zoo/motionmillion.md) | 3B train-only | 4,042 | 0.7401 | 0.8772 | 0.9229 | 3.0658 | 15.3806 | 27.5604 |
| [MotionStreamer](../model_zoo/motionstreamer.md) | Default | 4,042 | 0.6303 | 0.7865 | 0.8498 | 12.2110 | 16.5810 | 27.4637 |
| [HY-Motion T2M](../model_zoo/hymotion_t2m.md) | Full | 4,042 | 0.7847 | 0.9172 | 0.9509 | 13.8030 | 14.8196 | 27.4339 |
| [HY-Motion T2M](../model_zoo/hymotion_t2m.md) | Lite | 4,042 | 0.7939 | 0.9152 | 0.9524 | 10.4512 | 14.8361 | 27.4712 |
| [KIMODO](../model_zoo/kimodo.md) | SMPL-X RP | 4,042 | 0.3646 | 0.4998 | 0.5818 | 117.0279 | 21.4102 | 25.3629 |
| [MLD](../model_zoo/mld.md) | Default | 4,042 | 0.5660 | 0.7330 | 0.8100 | 39.7437 | 19.3374 | 24.9017 |
| [MotionLCM](../model_zoo/motionlcm.md) | Default | 4,042 | 0.5657 | 0.7346 | 0.8075 | 44.0549 | 19.4543 | 24.6395 |
| [ViMoGen](../model_zoo/vimogen.md) | 1.3B prompt-rewrite | 4,042 | 0.4291 | 0.5687 | 0.6518 | 152.2095 | 21.0737 | 24.1803 |
| [DART](../model_zoo/dart.md) | Default | 4,042 | 0.5476 | 0.7245 | 0.7937 | 127.8302 | 18.5312 | 26.2611 |

## [Motius Joint-Position Evaluator](../evaluator_zoo/motius_joint_position.md)

| Method | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ------ | ------- | ------: | --: | --: | --: | --: | ------: | --------: |
| [MDM](../model_zoo/mdm.md) | Default | 4,034 | 0.4501 | 0.6290 | 0.7262 | 263.3581 | 37.5544 | 56.3969 |
| [T2M-GPT](../model_zoo/t2mgpt.md) | Default | 4,034 | 0.4869 | 0.6520 | 0.7359 | 209.9396 | 36.3177 | 55.5376 |
| [MoMask](../model_zoo/momask.md) | Default | 4,034 | 0.5665 | 0.7540 | 0.8356 | 143.5427 | 33.3118 | 56.6110 |
| [MoGenTS](../model_zoo/mogents.md) | Default | 4,034 | 0.4623 | 0.6238 | 0.7138 | 158.5883 | 36.7141 | 56.5133 |
| [MotionGPT](../model_zoo/motiongpt.md) | Default | 4,034 | 0.4325 | 0.5801 | 0.6617 | 188.1248 | 38.4534 | 56.8849 |
| [FlowMDM](../model_zoo/flowmdm.md) | Default | 4,034 | 0.4390 | 0.6153 | 0.7111 | 227.4945 | 37.4096 | 55.5127 |
| [MotionMillion](../model_zoo/motionmillion.md) | 7B train-only | 4,034 | 0.6277 | 0.7904 | 0.8579 | 33.6025 | 29.9675 | 53.4788 |
| [MotionMillion](../model_zoo/motionmillion.md) | 3B train-only | 4,034 | 0.6228 | 0.7860 | 0.8569 | 34.4144 | 29.9661 | 54.6241 |
| [MotionStreamer](../model_zoo/motionstreamer.md) | Default | 4,034 | 0.4400 | 0.5967 | 0.6811 | 93.4685 | 35.6739 | 53.8000 |
| [HY-Motion T2M](../model_zoo/hymotion_t2m.md) | Full | 4,034 | 0.5722 | 0.7406 | 0.8170 | 28.3024 | 30.5147 | 54.1295 |
| [HY-Motion T2M](../model_zoo/hymotion_t2m.md) | Lite | 4,034 | 0.5940 | 0.7463 | 0.8135 | 32.0689 | 30.6713 | 55.4210 |
| [KIMODO](../model_zoo/kimodo.md) | SMPL-X RP | 4,034 | 0.3033 | 0.4638 | 0.5570 | 899.8286 | 47.8189 | 54.4397 |
| [MLD](../model_zoo/mld.md) | Default | 4,034 | 0.5169 | 0.6850 | 0.7701 | 258.6208 | 36.3447 | 57.3461 |
| [MotionLCM](../model_zoo/motionlcm.md) | Default | 4,034 | 0.5156 | 0.6915 | 0.7736 | 283.0533 | 36.5871 | 56.9460 |
| [ViMoGen](../model_zoo/vimogen.md) | 1.3B prompt-rewrite | 4,034 | 0.3041 | 0.4330 | 0.5198 | 922.4709 | 47.0574 | 55.6162 |
| [DART](../model_zoo/dart.md) | Default | 4,034 | 0.4249 | 0.6064 | 0.7019 | 371.1307 | 38.7637 | 56.9485 |
