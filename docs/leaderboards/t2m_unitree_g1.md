# Text-to-Motion · Unitree G1

<p align="center">
  <a href="https://huggingface.co/spaces/ZeyuLing/t2m-unitree-g1-leaderboard">↗ Live Leaderboard</a> ·
  <a href="README.md#text-to-motion">📊 All T2M Settings</a> ·
  <a href="../evaluator_zoo/g1_tmr.md">📐 TMR-G1 Evaluator</a> ·
  <a href="../tasks/README.md">🧭 Task Registry</a>
</p>

This is the robotic Text-to-Motion setting for methods that generate Unitree
G1 motion directly. It complements the
[HumanML3D setting](https://huggingface.co/spaces/ZeyuLing/t2m-humanml3d-leaderboard);
both implement the same Text-to-Motion task.

## Fixed Protocol

| Field | Contract |
| ----- | -------- |
| Task | Text-to-Motion |
| Embodiment | Unitree G1, 29 actuated joints |
| Protocol ID | `unitree-g1-paper-eval-1024-v1` |
| Test population | Fixed 1,024-case Unitree G1 evaluation split |
| Caption selection | One persisted caption per test sample |
| Evaluation representation | Canonical `g1_38` at 30 fps |
| Evaluator | [Motius TMR-G1](../evaluator_zoo/g1_tmr.md) |
| Semantic metrics | R@1 · R@2 · R@3 · normalized FID · MM-Dist · Diversity |
| Physical metrics | Root drift · foot slide · floating · jitter · penetration |

Native G1 outputs must use a declared bridge into `g1_38`. KIMODO and ARDY
retain their native generation tensors; MuJoCo `qpos-36` outputs use
`convert_motion(qpos, "g1_qpos", "g1_38")`. Human-body generations retargeted
to G1 are a separate conversion track and must not be mixed with direct
G1-native generation.

## Method Coverage

| Method | Variant | Native output | Artifact | Evaluation status |
| ------ | ------- | ------------- | -------- | ----------------- |
| [KIMODO](../model_zoo/kimodo.md) | G1-RP | Unitree G1 native arrays | [📦 Checkpoint](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp) | Measured, 1,024 cases |
| [KIMODO](../model_zoo/kimodo.md) | G1-SEED | Unitree G1 native arrays | [📦 Checkpoint](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed) | Pending full split |
| HY-Motion G1 | Iteration 339,000 | Unitree G1 | Release artifact pending | Measured, 1,024 cases |
| [ARDY](../model_zoo/ardy.md) | G1-RP Horizon-52 | ARDY G1-414 / qpos-36 | [📦 Checkpoint](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) | Pending full split |

## Leaderboard

| Method | Samples | R@1 ↑ | R@2 ↑ | R@3 ↑ | FID ↓ | MM-Dist ↓ | Diversity |
| ------ | ------: | ----: | ----: | ----: | ----: | --------: | --------: |
| GT calibration | 1,024 | 0.8489 | 0.9500 | 0.9796 | 0.000 | 16.346 | 36.768 |
| **HY-Motion G1 · iter 339k** | 1,024 | **0.7096** | **0.8390** | **0.8945** | **42.039** | **20.485** | **36.171** |
| KIMODO G1-RP | 1,024 | 0.5157 | 0.6749 | 0.7523 | 85.651 | 25.129 | 35.803 |
| KIMODO G1-SEED | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| ARDY G1-RP Horizon-52 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

The measured rows above come from one persisted metric report: the same 1,024
ordered captions, TMR-G1 epoch-139 checkpoint, 32-sample retrieval groups,
20 deterministic repeats, and normalized-latent FID.

## Latest Isolated Rerun

A newer HY-Motion G1 iteration-20k result measured R@1/2/3
`0.9078 / 0.9711 / 0.9873`, FID `16.575`, MM-Dist `18.269`, and Diversity
`36.414`. Its companion GT calibration was `0.9502 / 0.9942 / 0.9990`,
which differs materially from the comparison snapshot above. It is preserved
as an isolated rerun and is not cross-ranked until the baselines are replayed
under that exact evaluator snapshot.

The canonical storage root is
`outputs/evaluation/text_to_motion/text_to_motion_unitree_g1/unitree-g1-paper-eval-1024-v1`.
