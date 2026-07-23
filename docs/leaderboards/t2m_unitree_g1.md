# Text-to-Motion · Unitree G1

<p align="center">
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
| Test population | Fixed HY-Motion G1 evaluation split |
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
| [KIMODO](../model_zoo/kimodo.md) | G1-RP | Unitree G1 native arrays | [📦 Checkpoint](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-rp) | Pending full split |
| [KIMODO](../model_zoo/kimodo.md) | G1-SEED | Unitree G1 native arrays | [📦 Checkpoint](https://huggingface.co/ZeyuLing/hftrainer-kimodo-g1-seed) | Pending full split |
| HY-Motion G1 | Current G1 release | Unitree G1 | Release artifact pending | Pending full split |
| [ARDY](../model_zoo/ardy.md) | G1-RP Horizon-52 | ARDY G1-414 / qpos-36 | [📦 Checkpoint](https://huggingface.co/nvidia/ARDY-G1-RP-25FPS-Horizon52) | Pending full split |

## Leaderboard

| Method | Samples | R@1 ↑ | R@2 ↑ | R@3 ↑ | FID ↓ | MM-Dist ↓ | Diversity |
| ------ | ------: | ----: | ----: | ----: | ----: | --------: | --------: |
| GT | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| KIMODO G1-RP | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| KIMODO G1-SEED | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| HY-Motion G1 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| ARDY G1-RP Horizon-52 | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Rows become rankable only after the full fixed split, persisted selected
captions, normalized prediction files, and exact evaluator checkpoint are
published together.
