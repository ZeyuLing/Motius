# Changelog

## Unreleased

- Move migrated methods into native neural bundles, trainers, pipelines and shared motion components; remove the copied model and robot-training application trees.
- Integrate native AnimateCanvas, video capture, motion generation, physical tracking and retrieval paths, with compatibility aliases for older MotionCanvas artifacts.
- Retain the optimized FlowHMR frontend and Blender workflow, including standard and lightweight inference, full-frame batching and memory controls; document measured hardware and validation limits.
- Add source-comparison records, architectural audits, public robot-asset downloaders, training recipes and resume checks. Numerical comparisons are scoped to the tested models and cases; Windows and consumer-GPU end-to-end testing remains pending.

- Rename MotionCanvas to AnimateCanvas across model, pipeline, trainer, configs,
  model cards and local demo assets. Continue loading previously released
  checkpoint metadata through compatibility aliases.
- Redesign the homepage News section and README release list. Keep News limited
  to release announcements; record implementation updates here.

## 0.2.0 — 2026-10-05

- Replace retired HYMotionV2M with the official FlowHMR 491-D, 0.46B release.
- Add native base flow-matching training, official data preprocessing, full-state
  resume and strict base/latest inference checkpoint loading.
- Bundle the official YOLOX/VGGT-Omega/SAM frontend and preserve component licenses.
- Add Hugging Face inference artifacts, standardized Model Card and source-parity
  checks covering losses, gradients, sampling, camera and video preprocessing.
- Export native SMPL-H parameters and shared world-space monocular capture results.

## 0.1.0

- Existing Motius framework and method integrations.
