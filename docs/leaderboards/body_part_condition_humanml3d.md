# Part-Level Motion Control · HumanML3D

Public leaderboard: [HumanML3D body-part conditions](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard).
Repository mirror: [static HTML](body_part_condition_humanml3d.html).

The benchmark contains the complete 84-setting matrix used by the paper's
compact and full body-part tables.  Canonical artifacts use:

```text
outputs/evaluation/body_part/humanml3d_official_test_4012/{setting}/{method}/
```

Rotation errors are reported in degrees and position errors in centimeters.
Sparse support uses regular 30-frame evidence; dense support observes every
valid frame.

| Type | Target | Density | Axes | Complete methods | Canonical setting root |
|---|---|---|---|---:|---|
| rotation | upper body | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_upper_sparse` |
| rotation | upper body | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_upper_dense` |
| position | upper body | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_upper_sparse_xz` |
| position | upper body | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_upper_sparse_xyz` |
| position | upper body | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_upper_dense_xz` |
| position | upper body | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_upper_dense_xyz` |
| rotation | lower body | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_lower_sparse` |
| rotation | lower body | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_lower_dense` |
| position | lower body | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_lower_sparse_xz` |
| position | lower body | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_lower_sparse_xyz` |
| position | lower body | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_lower_dense_xz` |
| position | lower body | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_lower_dense_xyz` |
| rotation | left wrist | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_wrist_left_sparse` |
| rotation | left wrist | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_wrist_left_dense` |
| position | left wrist | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_left_sparse_xz` |
| position | left wrist | sparse | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_left_sparse_xyz` |
| position | left wrist | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_left_dense_xz` |
| position | left wrist | dense | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_left_dense_xyz` |
| rotation | right wrist | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_wrist_right_sparse` |
| rotation | right wrist | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_wrist_right_dense` |
| position | right wrist | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_right_sparse_xz` |
| position | right wrist | sparse | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_right_sparse_xyz` |
| position | right wrist | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_right_dense_xz` |
| position | right wrist | dense | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_right_dense_xyz` |
| rotation | both wrists | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_wrist_both_sparse` |
| rotation | both wrists | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_wrist_both_dense` |
| position | both wrists | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_both_sparse_xz` |
| position | both wrists | sparse | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_both_sparse_xyz` |
| position | both wrists | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_both_dense_xz` |
| position | both wrists | dense | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_wrist_both_dense_xyz` |
| rotation | left elbow | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_elbow_left_sparse` |
| rotation | left elbow | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_elbow_left_dense` |
| position | left elbow | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_left_sparse_xz` |
| position | left elbow | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_left_sparse_xyz` |
| position | left elbow | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_left_dense_xz` |
| position | left elbow | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_left_dense_xyz` |
| rotation | right elbow | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_elbow_right_sparse` |
| rotation | right elbow | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_elbow_right_dense` |
| position | right elbow | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_right_sparse_xz` |
| position | right elbow | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_right_sparse_xyz` |
| position | right elbow | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_right_dense_xz` |
| position | right elbow | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_right_dense_xyz` |
| rotation | both elbows | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_elbow_both_sparse` |
| rotation | both elbows | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_elbow_both_dense` |
| position | both elbows | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_both_sparse_xz` |
| position | both elbows | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_both_sparse_xyz` |
| position | both elbows | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_both_dense_xz` |
| position | both elbows | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_elbow_both_dense_xyz` |
| rotation | left foot | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_foot_left_sparse` |
| rotation | left foot | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_foot_left_dense` |
| position | left foot | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_left_sparse_xz` |
| position | left foot | sparse | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_left_sparse_xyz` |
| position | left foot | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_left_dense_xz` |
| position | left foot | dense | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_left_dense_xyz` |
| rotation | right foot | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_foot_right_sparse` |
| rotation | right foot | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_foot_right_dense` |
| position | right foot | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_right_sparse_xz` |
| position | right foot | sparse | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_right_sparse_xyz` |
| position | right foot | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_right_dense_xz` |
| position | right foot | dense | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_right_dense_xyz` |
| rotation | both feet | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_foot_both_sparse` |
| rotation | both feet | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_foot_both_dense` |
| position | both feet | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_both_sparse_xz` |
| position | both feet | sparse | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_both_sparse_xyz` |
| position | both feet | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_both_dense_xz` |
| position | both feet | dense | XYZ | 5/5 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_foot_both_dense_xyz` |
| rotation | left knee | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_knee_left_sparse` |
| rotation | left knee | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_knee_left_dense` |
| position | left knee | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_left_sparse_xz` |
| position | left knee | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_left_sparse_xyz` |
| position | left knee | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_left_dense_xz` |
| position | left knee | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_left_dense_xyz` |
| rotation | right knee | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_knee_right_sparse` |
| rotation | right knee | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_knee_right_dense` |
| position | right knee | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_right_sparse_xz` |
| position | right knee | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_right_sparse_xyz` |
| position | right knee | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_right_dense_xz` |
| position | right knee | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_right_dense_xyz` |
| rotation | both knees | sparse | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_knee_both_sparse` |
| rotation | both knees | dense | -- | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/rot_knee_both_dense` |
| position | both knees | sparse | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_both_sparse_xz` |
| position | both knees | sparse | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_both_sparse_xyz` |
| position | both knees | dense | XZ | 2/2 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_both_dense_xz` |
| position | both knees | dense | XYZ | 4/4 | `outputs/evaluation/body_part/humanml3d_official_test_4012/pos_knee_both_dense_xyz` |

## upper body: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5020 | 0.6638 | 0.7499 | 0.2923 | 37.59 | 15.91 | 0.09 | 3642.87 | 54.50 | 4012 |
| MotionCanvas | native | 0.7051 | 0.8575 | 0.9137 | 0.0026 | 27.09 | 0.00 | 0.06 | 471.02 | 53.45 | 4012 |
## upper body: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5510 | 0.7143 | 0.7953 | 0.1437 | 34.17 | 18.94 | 0.25 | 661.14 | 53.83 | 4012 |
| MotionCanvas | native | 0.6952 | 0.8498 | 0.9079 | 0.0021 | 27.43 | 0.00 | 0.06 | 515.18 | 53.36 | 4012 |
## upper body: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4879 | 0.6439 | 0.7252 | 0.0979 | 35.07 | 11.26 | 0.26 | 1217.51 | 55.82 | 4012 |
| MotionCanvas | native | 0.7010 | 0.8540 | 0.9089 | 0.0063 | 27.36 | 2.10 | 0.06 | 569.67 | 53.69 | 4012 |
## upper body: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5321 | 0.6910 | 0.7683 | 0.0862 | 33.61 | 12.12 | 0.25 | 1264.40 | 55.73 | 4012 |
| CondMDI | native | 0.5346 | 0.7049 | 0.7906 | 0.0475 | 32.47 | 9.11 | 0.08 | 1234.77 | 54.41 | 4012 |
| MotionLab | native | 0.6327 | 0.7988 | 0.8664 | 0.0178 | 29.97 | 5.28 | 0.02 | 302.36 | 53.64 | 4012 |
| MotionCanvas | native | 0.6997 | 0.8521 | 0.9093 | 0.0055 | 27.36 | 2.43 | 0.06 | 572.89 | 53.63 | 4012 |
## upper body: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4492 | 0.6068 | 0.6882 | 0.1125 | 36.36 | 11.43 | 0.32 | 382.84 | 54.77 | 4012 |
| MotionCanvas | native | 0.6873 | 0.8418 | 0.9017 | 0.0045 | 27.72 | 0.96 | 0.06 | 455.57 | 53.34 | 4012 |
## upper body: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5293 | 0.6897 | 0.7689 | 0.0865 | 33.36 | 12.69 | 0.27 | 385.53 | 55.06 | 4012 |
| CondMDI | native | 0.4003 | 0.5328 | 0.6042 | 0.3112 | 40.91 | 7.38 | 0.16 | 674.32 | 53.14 | 4012 |
| MotionLab | native | 0.6173 | 0.7850 | 0.8567 | 0.0308 | 30.71 | 6.36 | 0.03 | 332.72 | 53.99 | 4012 |
| MotionCanvas | native | 0.6868 | 0.8408 | 0.8999 | 0.0031 | 27.69 | 1.20 | 0.06 | 460.23 | 53.40 | 4012 |
## lower body: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5930 | 0.7579 | 0.8337 | 0.0916 | 32.47 | 26.55 | 0.13 | 1611.40 | 55.51 | 4012 |
| MotionCanvas | native | 0.7032 | 0.8527 | 0.9081 | 0.0078 | 27.55 | 0.00 | 0.08 | 495.70 | 53.96 | 4012 |
## lower body: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.4165 | 0.5621 | 0.6414 | 0.3773 | 40.55 | 29.76 | 0.06 | 386.48 | 52.83 | 4012 |
| MotionCanvas | native | 0.6878 | 0.8412 | 0.8989 | 0.0102 | 28.14 | 0.00 | 0.06 | 543.34 | 53.81 | 4012 |
## lower body: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5431 | 0.7067 | 0.7840 | 0.0617 | 33.26 | 9.49 | 0.24 | 1028.73 | 55.73 | 4012 |
| MotionCanvas | native | 0.6944 | 0.8433 | 0.9017 | 0.0074 | 27.67 | 2.34 | 0.10 | 549.86 | 53.74 | 4012 |
## lower body: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5699 | 0.7325 | 0.8066 | 0.0486 | 32.24 | 9.83 | 0.23 | 1071.14 | 55.46 | 4012 |
| CondMDI | native | 0.4789 | 0.6478 | 0.7385 | 0.0553 | 34.10 | 9.91 | 0.20 | 1225.12 | 54.77 | 4012 |
| MotionLab | native | 0.5827 | 0.7400 | 0.8115 | 0.0261 | 31.62 | 5.93 | 0.03 | 326.88 | 54.40 | 4012 |
| MotionCanvas | native | 0.6943 | 0.8433 | 0.9024 | 0.0076 | 27.70 | 2.56 | 0.10 | 554.95 | 53.76 | 4012 |
## lower body: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4740 | 0.6277 | 0.7063 | 0.1362 | 36.52 | 10.34 | 0.14 | 273.03 | 55.75 | 4012 |
| MotionCanvas | native | 0.6846 | 0.8373 | 0.8966 | 0.0065 | 28.03 | 1.06 | 0.06 | 461.89 | 53.68 | 4012 |
## lower body: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5716 | 0.7327 | 0.8058 | 0.0438 | 31.99 | 8.29 | 0.11 | 274.27 | 55.02 | 4012 |
| CondMDI | native | 0.4654 | 0.6241 | 0.7085 | 0.1391 | 35.84 | 7.97 | 0.21 | 696.33 | 53.65 | 4012 |
| MotionLab | native | 0.5595 | 0.7188 | 0.7899 | 0.0418 | 32.84 | 5.42 | 0.03 | 335.41 | 54.67 | 4012 |
| MotionCanvas | native | 0.6874 | 0.8403 | 0.9005 | 0.0057 | 27.87 | 1.22 | 0.05 | 470.21 | 53.57 | 4012 |
## left wrist: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.4681 | 0.6256 | 0.7137 | 0.3322 | 39.22 | 79.52 | 0.09 | 1497.56 | 54.99 | 4012 |
| MotionCanvas | native | 0.7026 | 0.8501 | 0.9062 | 0.0075 | 27.37 | 0.00 | 0.05 | 445.24 | 53.77 | 4012 |
## left wrist: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.3976 | 0.5566 | 0.6514 | 0.5145 | 43.19 | 68.96 | 0.15 | 652.31 | 51.78 | 4012 |
| MotionCanvas | native | 0.7004 | 0.8496 | 0.9059 | 0.0070 | 27.39 | 0.00 | 0.05 | 441.19 | 53.72 | 4012 |
## left wrist: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5638 | 0.7342 | 0.8144 | 0.0396 | 32.19 | 19.95 | 0.09 | 412.18 | 55.27 | 4012 |
| MotionCanvas | native | 0.6992 | 0.8460 | 0.9030 | 0.0084 | 27.58 | 13.01 | 0.05 | 542.64 | 53.86 | 4012 |
## left wrist: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.6066 | 0.7734 | 0.8454 | 0.0278 | 30.77 | 20.98 | 0.08 | 446.79 | 55.10 | 4012 |
| CondMDI | native | 0.4683 | 0.6364 | 0.7288 | 0.0560 | 34.38 | 22.05 | 0.08 | 924.87 | 54.97 | 4012 |
| MaskControl | native | 0.6331 | 0.7988 | 0.8664 | 0.0190 | 29.73 | 15.72 | 0.08 | 337.83 | 54.19 | 4012 |
| MotionLab | native | 0.5940 | 0.7539 | 0.8270 | 0.0209 | 31.00 | 14.34 | 0.02 | 307.63 | 54.08 | 4012 |
| MotionCanvas | native | 0.6992 | 0.8459 | 0.9027 | 0.0084 | 27.55 | 17.15 | 0.05 | 549.49 | 53.87 | 4012 |
## left wrist: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5345 | 0.7071 | 0.7922 | 0.0599 | 33.46 | 18.44 | 0.09 | 220.27 | 54.91 | 4012 |
| MotionCanvas | native | 0.6952 | 0.8469 | 0.9041 | 0.0069 | 27.60 | 2.32 | 0.06 | 450.85 | 53.78 | 4012 |
## left wrist: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.6148 | 0.7820 | 0.8542 | 0.0237 | 30.35 | 17.36 | 0.08 | 231.42 | 54.66 | 4012 |
| CondMDI | native | 0.4491 | 0.6113 | 0.7022 | 0.0916 | 36.31 | 18.69 | 0.09 | 584.74 | 55.66 | 4012 |
| MaskControl | native | 0.6265 | 0.7909 | 0.8587 | 0.0208 | 29.90 | 14.77 | 0.09 | 342.40 | 54.11 | 4012 |
| MotionLab | native | 0.6007 | 0.7604 | 0.8332 | 0.0247 | 30.74 | 13.56 | 0.02 | 318.10 | 54.23 | 4012 |
| MotionCanvas | native | 0.6931 | 0.8442 | 0.9023 | 0.0071 | 27.69 | 5.08 | 0.05 | 460.35 | 53.78 | 4012 |
## right wrist: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.4634 | 0.6233 | 0.7117 | 0.3165 | 39.19 | 68.11 | 0.09 | 1496.46 | 55.02 | 4012 |
| MotionCanvas | native | 0.7007 | 0.8498 | 0.9064 | 0.0079 | 27.40 | 0.00 | 0.05 | 445.00 | 53.82 | 4012 |
## right wrist: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.3762 | 0.5269 | 0.6192 | 0.4320 | 42.96 | 68.52 | 0.13 | 540.26 | 54.00 | 4012 |
| MotionCanvas | native | 0.7048 | 0.8522 | 0.9071 | 0.0077 | 27.36 | 0.00 | 0.06 | 443.86 | 53.76 | 4012 |
## right wrist: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5659 | 0.7363 | 0.8166 | 0.0382 | 32.06 | 20.02 | 0.08 | 415.28 | 55.30 | 4012 |
| MotionCanvas | native | 0.6989 | 0.8447 | 0.9018 | 0.0083 | 27.56 | 13.63 | 0.05 | 524.89 | 53.88 | 4012 |
## right wrist: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.6030 | 0.7701 | 0.8432 | 0.0280 | 30.82 | 21.06 | 0.08 | 452.40 | 55.05 | 4012 |
| CondMDI | native | 0.4674 | 0.6374 | 0.7314 | 0.0547 | 34.33 | 22.16 | 0.09 | 933.07 | 54.90 | 4012 |
| MaskControl | native | 0.6287 | 0.7926 | 0.8639 | 0.0181 | 29.74 | 15.84 | 0.08 | 336.59 | 54.07 | 4012 |
| MotionLab | native | 0.5970 | 0.7539 | 0.8251 | 0.0226 | 30.89 | 14.59 | 0.03 | 314.60 | 54.15 | 4012 |
| MotionCanvas | native | 0.7003 | 0.8462 | 0.9028 | 0.0081 | 27.53 | 17.40 | 0.05 | 535.39 | 53.86 | 4012 |
## right wrist: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5392 | 0.7130 | 0.7975 | 0.0582 | 33.31 | 18.25 | 0.09 | 221.47 | 54.75 | 4012 |
| MotionCanvas | native | 0.6931 | 0.8422 | 0.8979 | 0.0068 | 27.66 | 2.53 | 0.05 | 450.65 | 53.77 | 4012 |
## right wrist: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.6149 | 0.7806 | 0.8526 | 0.0246 | 30.38 | 17.34 | 0.08 | 232.76 | 54.67 | 4012 |
| CondMDI | native | 0.4755 | 0.6422 | 0.7317 | 0.0670 | 34.83 | 18.15 | 0.10 | 610.47 | 55.18 | 4012 |
| MaskControl | native | 0.6243 | 0.7843 | 0.8556 | 0.0211 | 29.96 | 15.30 | 0.09 | 343.58 | 54.09 | 4012 |
| MotionLab | native | 0.5999 | 0.7621 | 0.8321 | 0.0247 | 30.71 | 13.96 | 0.03 | 317.29 | 54.17 | 4012 |
| MotionCanvas | native | 0.6915 | 0.8424 | 0.8991 | 0.0067 | 27.71 | 4.91 | 0.05 | 456.43 | 53.71 | 4012 |
## both wrists: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.4396 | 0.5981 | 0.6899 | 0.4473 | 41.00 | 77.26 | 0.09 | 1573.60 | 52.49 | 4012 |
| MotionCanvas | native | 0.7041 | 0.8537 | 0.9096 | 0.0078 | 27.31 | 0.00 | 0.05 | 444.48 | 53.82 | 4012 |
## both wrists: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.4770 | 0.6450 | 0.7387 | 0.3482 | 38.91 | 65.18 | 0.15 | 1290.97 | 53.65 | 4012 |
| MotionCanvas | native | 0.6996 | 0.8490 | 0.9054 | 0.0072 | 27.41 | 0.00 | 0.06 | 455.06 | 53.69 | 4012 |
## both wrists: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5619 | 0.7289 | 0.8051 | 0.0464 | 32.37 | 18.82 | 0.11 | 602.49 | 55.35 | 4012 |
| MotionCanvas | native | 0.7018 | 0.8476 | 0.9040 | 0.0076 | 27.39 | 10.25 | 0.05 | 589.12 | 53.80 | 4012 |
## both wrists: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.6082 | 0.7709 | 0.8402 | 0.0339 | 30.72 | 20.62 | 0.11 | 662.05 | 54.97 | 4012 |
| CondMDI | native | 0.4922 | 0.6617 | 0.7484 | 0.0513 | 33.69 | 21.68 | 0.09 | 1098.61 | 55.00 | 4012 |
| MaskControl | native | 0.6517 | 0.8124 | 0.8775 | 0.0154 | 29.00 | 11.23 | 0.09 | 349.62 | 54.00 | 4012 |
| MotionLab | native | 0.6402 | 0.7994 | 0.8667 | 0.0148 | 29.51 | 10.51 | 0.03 | 311.57 | 53.95 | 4012 |
| MotionCanvas | native | 0.7041 | 0.8514 | 0.9082 | 0.0074 | 27.28 | 13.07 | 0.05 | 592.78 | 53.74 | 4012 |
## both wrists: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5057 | 0.6703 | 0.7531 | 0.0882 | 34.71 | 19.05 | 0.11 | 270.18 | 55.10 | 4012 |
| MotionCanvas | native | 0.6855 | 0.8402 | 0.8977 | 0.0077 | 27.92 | 1.63 | 0.05 | 454.49 | 53.75 | 4012 |
## both wrists: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5935 | 0.7592 | 0.8305 | 0.0427 | 31.15 | 19.52 | 0.10 | 286.62 | 54.76 | 4012 |
| CondMDI | native | 0.4578 | 0.6247 | 0.7137 | 0.0908 | 36.11 | 17.77 | 0.10 | 594.85 | 55.59 | 4012 |
| MaskControl | native | 0.6496 | 0.8095 | 0.8746 | 0.0130 | 29.15 | 9.88 | 0.08 | 331.46 | 53.98 | 4012 |
| MotionLab | native | 0.6414 | 0.8026 | 0.8708 | 0.0166 | 29.39 | 9.77 | 0.03 | 326.90 | 53.90 | 4012 |
| MotionCanvas | native | 0.6871 | 0.8400 | 0.8983 | 0.0062 | 27.85 | 3.78 | 0.05 | 461.08 | 53.72 | 4012 |
## left elbow: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5393 | 0.7072 | 0.7907 | 0.1180 | 34.65 | 63.27 | 0.10 | 2073.20 | 56.54 | 4012 |
| MotionCanvas | native | 0.6985 | 0.8481 | 0.9046 | 0.0077 | 27.43 | 0.00 | 0.05 | 487.26 | 53.85 | 4012 |
## left elbow: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5589 | 0.7231 | 0.8023 | 0.1051 | 33.94 | 52.82 | 0.10 | 416.28 | 56.33 | 4012 |
| MotionCanvas | native | 0.6964 | 0.8466 | 0.9043 | 0.0081 | 27.57 | 0.00 | 0.06 | 448.14 | 53.91 | 4012 |
## left elbow: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4953 | 0.6598 | 0.7419 | 0.0785 | 35.14 | 18.10 | 0.10 | 516.67 | 55.84 | 4012 |
| MotionCanvas | native | 0.6960 | 0.8431 | 0.9007 | 0.0085 | 27.63 | 12.94 | 0.05 | 455.84 | 53.86 | 4012 |
## left elbow: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4990 | 0.6631 | 0.7454 | 0.0773 | 34.94 | 21.02 | 0.10 | 593.96 | 55.94 | 4012 |
| CondMDI | native | 0.4538 | 0.6234 | 0.7182 | 0.0578 | 34.73 | 17.00 | 0.08 | 1042.17 | 54.94 | 4012 |
| MotionLab | native | 0.4654 | 0.6083 | 0.6808 | 0.0983 | 36.87 | 17.25 | 0.02 | 289.81 | 55.81 | 4012 |
| MotionCanvas | native | 0.6964 | 0.8430 | 0.9005 | 0.0085 | 27.62 | 14.85 | 0.05 | 460.88 | 53.88 | 4012 |
## left elbow: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4132 | 0.5691 | 0.6525 | 0.1691 | 39.57 | 18.98 | 0.11 | 142.74 | 55.70 | 4012 |
| MotionCanvas | native | 0.6981 | 0.8470 | 0.9031 | 0.0077 | 27.55 | 5.50 | 0.05 | 469.57 | 53.76 | 4012 |
## left elbow: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4336 | 0.5884 | 0.6736 | 0.1486 | 38.63 | 22.50 | 0.11 | 156.69 | 55.91 | 4012 |
| CondMDI | native | 0.4648 | 0.6334 | 0.7244 | 0.0784 | 35.30 | 13.97 | 0.08 | 601.52 | 55.08 | 4012 |
| MotionLab | native | 0.4188 | 0.5589 | 0.6362 | 0.2002 | 39.68 | 19.46 | 0.02 | 262.91 | 54.85 | 4012 |
| MotionCanvas | native | 0.6985 | 0.8484 | 0.9040 | 0.0077 | 27.58 | 6.52 | 0.06 | 479.18 | 53.76 | 4012 |
## right elbow: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5401 | 0.7088 | 0.7913 | 0.1191 | 34.69 | 61.60 | 0.10 | 2071.88 | 56.47 | 4012 |
| MotionCanvas | native | 0.7038 | 0.8510 | 0.9060 | 0.0080 | 27.41 | 0.00 | 0.05 | 503.69 | 53.90 | 4012 |
## right elbow: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5577 | 0.7255 | 0.8057 | 0.1111 | 34.04 | 49.30 | 0.10 | 414.06 | 56.34 | 4012 |
| MotionCanvas | native | 0.7023 | 0.8518 | 0.9066 | 0.0072 | 27.43 | 0.00 | 0.05 | 445.50 | 53.63 | 4012 |
## right elbow: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4892 | 0.6512 | 0.7357 | 0.0840 | 35.48 | 18.85 | 0.10 | 507.97 | 55.90 | 4012 |
| MotionCanvas | native | 0.6960 | 0.8420 | 0.8999 | 0.0085 | 27.65 | 12.75 | 0.05 | 460.92 | 53.88 | 4012 |
## right elbow: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5004 | 0.6597 | 0.7430 | 0.0808 | 35.08 | 21.26 | 0.10 | 576.36 | 55.95 | 4012 |
| CondMDI | native | 0.4523 | 0.6220 | 0.7169 | 0.0583 | 34.73 | 16.88 | 0.08 | 1029.33 | 54.89 | 4012 |
| MotionLab | native | 0.4861 | 0.6311 | 0.7048 | 0.0811 | 35.73 | 16.61 | 0.02 | 296.70 | 55.48 | 4012 |
| MotionCanvas | native | 0.6956 | 0.8425 | 0.9004 | 0.0085 | 27.64 | 14.64 | 0.05 | 467.86 | 53.86 | 4012 |
## right elbow: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4031 | 0.5576 | 0.6462 | 0.1831 | 40.14 | 19.74 | 0.11 | 139.41 | 55.78 | 4012 |
| MotionCanvas | native | 0.6972 | 0.8442 | 0.9010 | 0.0077 | 27.58 | 5.50 | 0.05 | 476.74 | 53.69 | 4012 |
## right elbow: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4222 | 0.5805 | 0.6684 | 0.1629 | 39.09 | 22.63 | 0.11 | 149.14 | 55.95 | 4012 |
| CondMDI | native | 0.4686 | 0.6332 | 0.7226 | 0.0828 | 35.58 | 13.51 | 0.09 | 601.82 | 55.89 | 4012 |
| MotionLab | native | 0.4523 | 0.6008 | 0.6782 | 0.1433 | 37.71 | 17.85 | 0.02 | 267.45 | 55.02 | 4012 |
| MotionCanvas | native | 0.6949 | 0.8451 | 0.9028 | 0.0075 | 27.61 | 6.54 | 0.05 | 483.32 | 53.70 | 4012 |
## both elbows: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5614 | 0.7272 | 0.8074 | 0.1137 | 34.04 | 61.82 | 0.10 | 2445.15 | 56.46 | 4012 |
| MotionCanvas | native | 0.6991 | 0.8488 | 0.9059 | 0.0086 | 27.48 | 0.00 | 0.05 | 490.75 | 53.86 | 4012 |
## both elbows: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5880 | 0.7603 | 0.8368 | 0.0908 | 32.68 | 50.84 | 0.10 | 437.40 | 56.10 | 4012 |
| MotionCanvas | native | 0.6994 | 0.8482 | 0.9054 | 0.0082 | 27.53 | 0.00 | 0.06 | 458.02 | 53.78 | 4012 |
## both elbows: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5161 | 0.6837 | 0.7666 | 0.0652 | 34.27 | 15.65 | 0.12 | 674.86 | 55.54 | 4012 |
| MotionCanvas | native | 0.6991 | 0.8462 | 0.9040 | 0.0081 | 27.52 | 11.79 | 0.05 | 483.64 | 53.82 | 4012 |
## both elbows: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5194 | 0.6839 | 0.7666 | 0.0649 | 34.05 | 18.33 | 0.12 | 774.77 | 55.73 | 4012 |
| CondMDI | native | 0.4558 | 0.6247 | 0.7169 | 0.0566 | 34.60 | 16.74 | 0.08 | 1256.42 | 54.96 | 4012 |
| MotionLab | native | 0.5756 | 0.7331 | 0.8046 | 0.0315 | 31.90 | 11.82 | 0.03 | 331.16 | 54.81 | 4012 |
| MotionCanvas | native | 0.7006 | 0.8492 | 0.9053 | 0.0079 | 27.47 | 13.35 | 0.05 | 495.86 | 53.81 | 4012 |
## both elbows: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4449 | 0.6054 | 0.6923 | 0.1325 | 37.92 | 16.39 | 0.16 | 153.14 | 55.44 | 4012 |
| MotionCanvas | native | 0.6936 | 0.8438 | 0.9004 | 0.0069 | 27.64 | 4.17 | 0.05 | 471.93 | 53.65 | 4012 |
## both elbows: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4660 | 0.6264 | 0.7103 | 0.1181 | 37.04 | 19.70 | 0.16 | 161.08 | 55.78 | 4012 |
| CondMDI | native | 0.4449 | 0.6062 | 0.6939 | 0.1187 | 36.98 | 13.67 | 0.08 | 606.96 | 55.57 | 4012 |
| MotionLab | native | 0.5895 | 0.7512 | 0.8228 | 0.0356 | 31.48 | 10.69 | 0.03 | 323.43 | 54.45 | 4012 |
| MotionCanvas | native | 0.6966 | 0.8472 | 0.9040 | 0.0067 | 27.65 | 4.94 | 0.06 | 479.91 | 53.66 | 4012 |
## left foot: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5437 | 0.7120 | 0.7935 | 0.1261 | 34.77 | 68.75 | 0.09 | 1502.98 | 56.60 | 4012 |
| MotionCanvas | native | 0.6979 | 0.8439 | 0.9014 | 0.0085 | 27.62 | 0.00 | 0.05 | 433.29 | 53.89 | 4012 |
## left foot: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5295 | 0.6986 | 0.7835 | 0.1390 | 35.41 | 72.53 | 0.10 | 467.55 | 56.72 | 4012 |
| MotionCanvas | native | 0.7011 | 0.8492 | 0.9058 | 0.0080 | 27.40 | 0.00 | 0.05 | 435.60 | 53.75 | 4012 |
## left foot: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5572 | 0.7269 | 0.8067 | 0.0381 | 32.26 | 16.98 | 0.11 | 365.81 | 55.23 | 4012 |
| MotionCanvas | native | 0.6956 | 0.8423 | 0.8992 | 0.0088 | 27.67 | 13.74 | 0.08 | 503.01 | 53.90 | 4012 |
## left foot: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5758 | 0.7439 | 0.8205 | 0.0343 | 31.69 | 16.76 | 0.10 | 369.93 | 55.14 | 4012 |
| CondMDI | native | 0.4501 | 0.6208 | 0.7147 | 0.0563 | 34.81 | 17.57 | 0.14 | 811.90 | 54.82 | 4012 |
| MaskControl | native | 0.6105 | 0.7756 | 0.8474 | 0.0227 | 30.63 | 15.01 | 0.11 | 330.92 | 54.41 | 4012 |
| MotionLab | native | 0.5541 | 0.7113 | 0.7835 | 0.0350 | 32.67 | 15.04 | 0.03 | 319.20 | 54.67 | 4012 |
| MotionCanvas | native | 0.6954 | 0.8421 | 0.8996 | 0.0089 | 27.67 | 14.80 | 0.08 | 508.98 | 53.90 | 4012 |
## left foot: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5259 | 0.6985 | 0.7830 | 0.0567 | 33.68 | 16.80 | 0.09 | 211.29 | 55.00 | 4012 |
| MotionCanvas | native | 0.6967 | 0.8439 | 0.9015 | 0.0082 | 27.68 | 4.36 | 0.08 | 454.85 | 53.92 | 4012 |
## left foot: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5867 | 0.7554 | 0.8307 | 0.0293 | 31.30 | 13.42 | 0.08 | 238.74 | 54.90 | 4012 |
| CondMDI | native | 0.4705 | 0.6428 | 0.7387 | 0.0812 | 34.87 | 14.02 | 0.13 | 580.65 | 55.15 | 4012 |
| MaskControl | native | 0.6080 | 0.7709 | 0.8425 | 0.0231 | 30.61 | 13.66 | 0.08 | 338.98 | 54.43 | 4012 |
| MotionLab | native | 0.5590 | 0.7151 | 0.7888 | 0.0366 | 32.59 | 13.01 | 0.03 | 327.43 | 54.73 | 4012 |
| MotionCanvas | native | 0.6953 | 0.8437 | 0.9014 | 0.0082 | 27.70 | 5.37 | 0.09 | 461.86 | 53.92 | 4012 |
## right foot: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5406 | 0.7110 | 0.7954 | 0.1260 | 34.88 | 107.76 | 0.09 | 1484.61 | 56.58 | 4012 |
| MotionCanvas | native | 0.6971 | 0.8430 | 0.9007 | 0.0086 | 27.64 | 0.00 | 0.05 | 434.02 | 53.88 | 4012 |
## right foot: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5412 | 0.7123 | 0.7940 | 0.1413 | 35.16 | 100.25 | 0.10 | 460.09 | 56.70 | 4012 |
| MotionCanvas | native | 0.6994 | 0.8473 | 0.9038 | 0.0089 | 27.50 | 0.00 | 0.05 | 439.16 | 53.81 | 4012 |
## right foot: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5565 | 0.7253 | 0.8052 | 0.0382 | 32.23 | 16.93 | 0.11 | 361.40 | 55.18 | 4012 |
| MotionCanvas | native | 0.6953 | 0.8416 | 0.8992 | 0.0089 | 27.67 | 13.95 | 0.08 | 499.78 | 53.90 | 4012 |
## right foot: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5765 | 0.7454 | 0.8225 | 0.0341 | 31.58 | 17.38 | 0.11 | 370.84 | 55.22 | 4012 |
| CondMDI | native | 0.4521 | 0.6204 | 0.7150 | 0.0554 | 34.73 | 17.85 | 0.14 | 800.15 | 54.69 | 4012 |
| MaskControl | native | 0.6062 | 0.7690 | 0.8407 | 0.0266 | 30.82 | 15.71 | 0.11 | 333.53 | 54.56 | 4012 |
| MotionLab | native | 0.5363 | 0.6905 | 0.7643 | 0.0401 | 33.45 | 16.57 | 0.03 | 324.31 | 54.93 | 4012 |
| MotionCanvas | native | 0.6954 | 0.8414 | 0.8988 | 0.0090 | 27.68 | 14.95 | 0.08 | 504.60 | 53.94 | 4012 |
## right foot: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5300 | 0.6987 | 0.7829 | 0.0576 | 33.60 | 17.64 | 0.09 | 208.67 | 54.93 | 4012 |
| MotionCanvas | native | 0.6964 | 0.8435 | 0.8995 | 0.0087 | 27.74 | 4.57 | 0.09 | 451.30 | 54.02 | 4012 |
## right foot: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5992 | 0.7666 | 0.8406 | 0.0270 | 30.92 | 13.97 | 0.08 | 242.70 | 54.82 | 4012 |
| CondMDI | native | 0.4801 | 0.6539 | 0.7439 | 0.0695 | 34.13 | 14.54 | 0.18 | 594.67 | 54.54 | 4012 |
| MaskControl | native | 0.6045 | 0.7683 | 0.8398 | 0.0276 | 30.89 | 13.96 | 0.09 | 336.93 | 54.54 | 4012 |
| MotionLab | native | 0.5323 | 0.6870 | 0.7627 | 0.0522 | 33.81 | 14.98 | 0.03 | 324.81 | 55.08 | 4012 |
| MotionCanvas | native | 0.6950 | 0.8433 | 0.8998 | 0.0090 | 27.78 | 5.56 | 0.09 | 458.95 | 54.06 | 4012 |
## both feet: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5479 | 0.7198 | 0.8033 | 0.1275 | 34.59 | 87.67 | 0.08 | 1484.59 | 56.66 | 4012 |
| MotionCanvas | native | 0.6983 | 0.8447 | 0.9016 | 0.0085 | 27.60 | 0.00 | 0.05 | 435.61 | 53.87 | 4012 |
## both feet: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5442 | 0.7158 | 0.7970 | 0.1558 | 35.44 | 84.98 | 0.10 | 483.32 | 56.80 | 4012 |
| MotionCanvas | native | 0.7037 | 0.8514 | 0.9077 | 0.0087 | 27.32 | 0.00 | 0.05 | 443.02 | 53.71 | 4012 |
## both feet: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5499 | 0.7117 | 0.7875 | 0.0523 | 32.75 | 15.60 | 0.17 | 543.43 | 55.49 | 4012 |
| MaskControl | unsupported | 0.6180 | 0.7827 | 0.8519 | 0.0223 | 30.39 | 9.01 | 0.12 | 340.11 | 54.46 | 4012 |
| MotionCanvas | native | 0.6960 | 0.8410 | 0.8989 | 0.0087 | 27.68 | 12.50 | 0.09 | 535.49 | 53.87 | 4012 |
## both feet: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5742 | 0.7340 | 0.8034 | 0.0483 | 32.18 | 16.23 | 0.16 | 545.13 | 55.32 | 4012 |
| CondMDI | native | 0.4574 | 0.6262 | 0.7195 | 0.0583 | 34.64 | 17.20 | 0.18 | 944.58 | 54.73 | 4012 |
| MaskControl | native | 0.6176 | 0.7798 | 0.8495 | 0.0249 | 30.54 | 10.62 | 0.13 | 348.26 | 54.57 | 4012 |
| MotionLab | native | 0.5681 | 0.7257 | 0.7968 | 0.0304 | 32.15 | 12.62 | 0.03 | 330.82 | 54.70 | 4012 |
| MotionCanvas | native | 0.6966 | 0.8428 | 0.8997 | 0.0088 | 27.68 | 13.28 | 0.09 | 540.18 | 53.88 | 4012 |
## both feet: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4758 | 0.6275 | 0.7038 | 0.1369 | 36.53 | 17.39 | 0.11 | 257.23 | 55.71 | 4012 |
| MaskControl | unsupported | 0.6206 | 0.7847 | 0.8553 | 0.0190 | 30.10 | 8.05 | 0.07 | 334.67 | 54.26 | 4012 |
| MotionCanvas | native | 0.6918 | 0.8417 | 0.9000 | 0.0076 | 27.78 | 3.11 | 0.07 | 453.43 | 53.89 | 4012 |
## both feet: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.5606 | 0.7177 | 0.7873 | 0.0615 | 32.75 | 14.04 | 0.09 | 272.55 | 55.35 | 4012 |
| CondMDI | native | 0.4933 | 0.6664 | 0.7556 | 0.0745 | 33.98 | 13.71 | 0.19 | 613.45 | 54.36 | 4012 |
| MaskControl | native | 0.6112 | 0.7737 | 0.8453 | 0.0221 | 30.41 | 9.57 | 0.07 | 339.52 | 54.29 | 4012 |
| MotionLab | native | 0.5586 | 0.7151 | 0.7872 | 0.0440 | 32.85 | 10.89 | 0.03 | 338.79 | 54.97 | 4012 |
| MotionCanvas | native | 0.6906 | 0.8419 | 0.8993 | 0.0080 | 27.80 | 4.03 | 0.07 | 465.53 | 53.93 | 4012 |
## left knee: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5406 | 0.7124 | 0.7969 | 0.1216 | 34.77 | 43.13 | 0.27 | 1973.66 | 56.44 | 4012 |
| MotionCanvas | native | 0.6978 | 0.8459 | 0.9032 | 0.0084 | 27.62 | 0.00 | 0.08 | 529.97 | 53.86 | 4012 |
## left knee: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5374 | 0.7085 | 0.7899 | 0.1450 | 35.25 | 39.32 | 0.19 | 459.76 | 56.52 | 4012 |
| MotionCanvas | native | 0.6956 | 0.8477 | 0.9041 | 0.0074 | 27.66 | 0.00 | 0.06 | 464.88 | 53.84 | 4012 |
## left knee: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4818 | 0.6413 | 0.7229 | 0.0959 | 36.08 | 13.74 | 0.12 | 451.60 | 56.20 | 4012 |
| MotionCanvas | native | 0.6950 | 0.8424 | 0.8995 | 0.0087 | 27.66 | 9.61 | 0.06 | 455.10 | 53.88 | 4012 |
## left knee: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4763 | 0.6361 | 0.7186 | 0.0970 | 36.20 | 15.04 | 0.12 | 480.06 | 56.26 | 4012 |
| CondMDI | native | 0.4493 | 0.6200 | 0.7139 | 0.0582 | 34.93 | 12.10 | 0.10 | 924.89 | 54.79 | 4012 |
| MotionLab | native | 0.5501 | 0.7070 | 0.7802 | 0.0374 | 32.83 | 10.74 | 0.02 | 315.72 | 54.59 | 4012 |
| MotionCanvas | native | 0.6952 | 0.8426 | 0.8998 | 0.0090 | 27.67 | 10.07 | 0.07 | 460.49 | 53.88 | 4012 |
## left knee: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.3943 | 0.5488 | 0.6363 | 0.1965 | 40.87 | 14.64 | 0.12 | 139.01 | 55.89 | 4012 |
| MotionCanvas | native | 0.6927 | 0.8397 | 0.8982 | 0.0086 | 27.75 | 4.45 | 0.08 | 481.39 | 53.88 | 4012 |
## left knee: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4117 | 0.5658 | 0.6535 | 0.1782 | 39.97 | 16.35 | 0.12 | 159.56 | 56.26 | 4012 |
| CondMDI | native | 0.4611 | 0.6301 | 0.7229 | 0.0579 | 34.49 | 9.18 | 0.12 | 551.86 | 54.57 | 4012 |
| MotionLab | native | 0.5588 | 0.7123 | 0.7856 | 0.0396 | 32.63 | 9.46 | 0.02 | 314.65 | 54.74 | 4012 |
| MotionCanvas | native | 0.6936 | 0.8403 | 0.8989 | 0.0088 | 27.76 | 4.37 | 0.08 | 473.94 | 53.84 | 4012 |
## right knee: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5383 | 0.7075 | 0.7919 | 0.1193 | 34.82 | 49.97 | 0.27 | 1992.99 | 56.47 | 4012 |
| MotionCanvas | native | 0.7000 | 0.8473 | 0.9036 | 0.0085 | 27.56 | 0.00 | 0.07 | 515.26 | 53.90 | 4012 |
## right knee: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5343 | 0.7092 | 0.7954 | 0.1474 | 35.33 | 48.24 | 0.18 | 456.85 | 56.45 | 4012 |
| MotionCanvas | native | 0.6975 | 0.8462 | 0.9023 | 0.0084 | 27.74 | 0.00 | 0.06 | 458.34 | 53.91 | 4012 |
## right knee: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4851 | 0.6473 | 0.7308 | 0.0884 | 35.70 | 13.58 | 0.11 | 448.96 | 56.02 | 4012 |
| MotionCanvas | native | 0.6954 | 0.8423 | 0.8992 | 0.0088 | 27.68 | 9.60 | 0.07 | 462.40 | 53.90 | 4012 |
## right knee: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4855 | 0.6462 | 0.7288 | 0.0885 | 35.71 | 15.01 | 0.12 | 481.13 | 56.11 | 4012 |
| CondMDI | native | 0.4482 | 0.6174 | 0.7136 | 0.0579 | 34.98 | 12.26 | 0.10 | 936.98 | 54.85 | 4012 |
| MotionLab | native | 0.5601 | 0.7138 | 0.7865 | 0.0345 | 32.43 | 10.78 | 0.03 | 314.23 | 54.68 | 4012 |
| MotionCanvas | native | 0.6949 | 0.8419 | 0.8990 | 0.0092 | 27.71 | 10.07 | 0.07 | 469.10 | 53.92 | 4012 |
## right knee: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4028 | 0.5556 | 0.6416 | 0.1792 | 40.11 | 14.23 | 0.12 | 142.86 | 55.80 | 4012 |
| MotionCanvas | native | 0.6954 | 0.8419 | 0.8975 | 0.0082 | 27.71 | 4.06 | 0.08 | 467.80 | 53.85 | 4012 |
## right knee: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4070 | 0.5630 | 0.6497 | 0.1703 | 39.82 | 15.83 | 0.12 | 151.71 | 55.79 | 4012 |
| CondMDI | native | 0.4633 | 0.6363 | 0.7280 | 0.0632 | 34.60 | 9.51 | 0.15 | 565.27 | 54.83 | 4012 |
| MotionLab | native | 0.5567 | 0.7131 | 0.7854 | 0.0408 | 32.74 | 9.47 | 0.03 | 315.72 | 54.70 | 4012 |
| MotionCanvas | native | 0.6949 | 0.8417 | 0.8989 | 0.0084 | 27.76 | 3.94 | 0.08 | 465.02 | 53.85 | 4012 |
## both knees: rotation / sparse / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5406 | 0.7119 | 0.7983 | 0.1219 | 34.67 | 46.36 | 0.33 | 2332.32 | 56.29 | 4012 |
| MotionCanvas | native | 0.7002 | 0.8486 | 0.9052 | 0.0085 | 27.55 | 0.00 | 0.08 | 514.48 | 53.90 | 4012 |
## both knees: rotation / dense / --

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| KIMODO | extra position evidence | 0.5377 | 0.7048 | 0.7858 | 0.1604 | 35.68 | 43.97 | 0.19 | 510.49 | 56.35 | 4012 |
| MotionCanvas | native | 0.6944 | 0.8477 | 0.9044 | 0.0070 | 27.71 | 0.00 | 0.06 | 486.14 | 53.74 | 4012 |
## both knees: position / sparse / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4939 | 0.6541 | 0.7348 | 0.0845 | 35.42 | 12.33 | 0.16 | 582.26 | 56.11 | 4012 |
| MotionCanvas | native | 0.6968 | 0.8440 | 0.9019 | 0.0087 | 27.60 | 8.83 | 0.07 | 482.07 | 53.83 | 4012 |
## both knees: position / sparse / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4950 | 0.6554 | 0.7379 | 0.0811 | 35.32 | 13.78 | 0.17 | 626.13 | 56.19 | 4012 |
| CondMDI | native | 0.4488 | 0.6185 | 0.7121 | 0.0553 | 34.89 | 11.99 | 0.12 | 1104.68 | 54.77 | 4012 |
| MotionLab | native | 0.5829 | 0.7373 | 0.8070 | 0.0277 | 31.62 | 8.50 | 0.03 | 326.03 | 54.47 | 4012 |
| MotionCanvas | native | 0.6973 | 0.8444 | 0.9028 | 0.0088 | 27.62 | 9.02 | 0.08 | 490.34 | 53.86 | 4012 |
## both knees: position / dense / XZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4150 | 0.5678 | 0.6543 | 0.1664 | 39.58 | 13.08 | 0.19 | 149.87 | 55.74 | 4012 |
| MotionCanvas | native | 0.6946 | 0.8451 | 0.9020 | 0.0063 | 27.69 | 2.80 | 0.07 | 465.74 | 53.68 | 4012 |
## both knees: position / dense / XYZ

| Method | Protocol | R@1 | R@2 | R@3 | FID | MM-Dist | Ctrl. | Foot | Jitter | Diversity | N |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OmniControl | native | 0.4367 | 0.5907 | 0.6756 | 0.1446 | 38.38 | 14.92 | 0.19 | 174.05 | 56.04 | 4012 |
| CondMDI | native | 0.4702 | 0.6409 | 0.7333 | 0.0622 | 34.29 | 8.69 | 0.16 | 544.11 | 54.46 | 4012 |
| MotionLab | native | 0.5802 | 0.7358 | 0.8067 | 0.0354 | 31.91 | 6.54 | 0.03 | 329.85 | 54.51 | 4012 |
| MotionCanvas | native | 0.6946 | 0.8451 | 0.9022 | 0.0063 | 27.71 | 2.74 | 0.08 | 460.70 | 53.60 | 4012 |

Rebuild after adding canonical artifacts or metrics:

```bash
python3 scripts/eval/build_body_part_condition_benchmark.py
```
