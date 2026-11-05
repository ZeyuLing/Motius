# GenTrack Training and Reproduction

GenTrack post-trains a native G1-38 flow-matching generator against physical
execution feedback while updating a tracker on newly generated references.
The public implementation preserves the experiment code's two independent
clocks:

1. At round \(r\), the generator is scored by the frozen tracker from round
   \(r-1\).
2. Same-noise samples from the immutable generator \(G_0\) provide the
   counterfactual baseline and semantic anchor.
3. Flow-GRPO updates the live generator; structurally valid samples enter the
   tracker replay pool.
4. The tracker receives a fixed public/generated mixture, trains for its
   declared PPO budget, and is exported for the next round only after the
   checkpoint and budget attestations pass.

The implementation lives in:

- `motius/models/gentrack`: G1 neural bundle. Data processing, physical rewards and optimization
  objectives live in their framework components.
- `motius/trainers/gentrack`: online GRPO, reward-weighted SFT, DPO,
  generator-only, and frozen-pool trainers.
- `configs/gentrack`: the main recipe and matched ablations.
- `tools/gentrack`: co-evolution, unified evaluation, replay conversion,
  qualification, and runtime-attestation CLIs.

The historical internal class prefix `PhysFlow` remains in a few source
filenames so existing checkpoints and configs stay loadable. Public registry
names use `GenTrack`.

## Installation

The generator-only unit tests need the normal development environment. The
physical judge and tracker update need the corresponding extras and an
installed simulator:

```bash
python -m pip install -e ".[dev,gentrack,motion-tracking-train-protomotions]"
```

Install the selected, hash-verified ProtoMotions robot assets with
`python tools/download_protomotions_assets.py`. They are stored under
`checkpoints/robots/protomotions`, outside the Python network package.
Tracker training uses the native Motius Trainer and AccelerateRunner with an
installed Isaac Lab / Isaac Sim runtime; the physical judge uses MuJoCo.
The legacy 135D-to-G1 retarget helper additionally needs
`pip install -e ".[gentrack-retarget]"`; native G1-38 training does not.

## Required Inputs

GenTrack does not redistribute third-party model weights or restricted motion
data. Set these paths explicitly:

```bash
export MOTIUS_DATA_ROOT=/path/to/preprocessed/g1_38
export MOTIUS_TRAIN_MANIFEST=train.json
export MOTIUS_MOTION_STATS=/path/to/g1_38/stats
export MOTIUS_GENTRACK_G0_CHECKPOINT=/path/to/hymotion_g1_g0
export MOTIUS_GENTRACK_PROTOMOTIONS_ARTIFACT=/path/to/g1-bones-deploy
export MOTIUS_GENTRACK_TRAINEE_CHECKPOINT=/path/to/protomotions_t0/last.ckpt
export MOTIUS_GENTRACK_PROTOMOTIONS_ONNX=/path/to/t0/unified_pipeline.onnx
export MOTIUS_GENTRACK_PROTO_PYTHON=/path/to/isaaclab/python
export MOTIUS_GENTRACK_PROTO_CONFIGURATION=/path/to/t0/resolved_configs.yaml
export MOTIUS_GENTRACK_PROTOMOTIONS_ASSETS=/path/to/robots/protomotions
```

The HYMotion manifest follows `ManifestTextMotionDataset`: every motion is
G1-38, text features are precomputed, clips are capped at 360 frames, and the
normalization statistics match the immutable \(G_0\) checkpoint.

The ProtoMotions backend is the default and needs no private launcher. The
optional SONIC backends use SONIC's simulator-owned runtime; point
`MOTIUS_GENTRACK_SONIC_RUNNER` at its official evaluation launcher and, for
the persistent service mode, set `MOTIUS_GENTRACK_SONIC_MATERIALIZER`.
The released-controller MuJoCo ablation instead uses
`MOTIUS_GENTRACK_SONIC_MUJOCO_RUNNER`. These explicit integration hooks keep
scheduler, image, and restricted checkpoint locations out of source control.
The optional Humanoid-GPT judge similarly accepts
`MOTIUS_GENTRACK_HGPT_PYTHON` and `MOTIUS_GENTRACK_HGPT_WORKER`; its upstream
JAX rollout runtime is not imported into the generator process.

## Training

Objective: clipped Flow-GRPO with an immutable-\(G_0\) reference KL term,
periodic GT anchoring, and lagged execution reward. Precision: FP32
(`mixed_precision="no"`) in the released recipe so simulator-derived
advantages and flow log-ratios share one numerical contract. Launch
[`configs/gentrack/train_gentrack_g1.py`](../../configs/gentrack/train_gentrack_g1.py)
for the online method or
[`configs/gentrack/train_gentrack_offline_g1.py`](../../configs/gentrack/train_gentrack_offline_g1.py)
for the frozen-pool control. Checkpoints are written below
`outputs/training/gentrack_g1`; `--auto-resume` restores the latest complete
state, while `--load-from PATH --load-scope full` resumes an explicit
checkpoint. Use `--load-scope model` only for a declared weight warm start.

## Generator Update

For a single frozen-judge round, write a judge specification:

```json
{
  "judges": [
    {
      "name": "quality",
      "onnx": "/path/to/t0/unified_pipeline.onnx",
      "weight": 1.0
    }
  ]
}
```

Then launch:

```bash
export PHYSFLOW_JUDGE_SPEC=/path/to/judge_spec.json
python tools/train.py configs/gentrack/train_gentrack_g1.py \
  --work-dir outputs/training/gentrack_g1/r0
```

`PHYSFLOW_JUDGE_SPEC` is retained as the environment name because it is stored
in released runtime receipts. The config uses the public
`GenTrackG1Bundle`/`GenTrackFlowGRPOTrainer` names and the paper-reported
immutable-reference KL weight of `0.02`.

## Multi-Round Co-Evolution

The dependency-light orchestrator records every checkpoint identity, source
mixture, requested optimizer budget, and judge transition:

```bash
python tools/gentrack/coevolve.py \
  --judge-mode lagged \
  --num-rounds 12 \
  --gen-iters 120 \
  --trainee-epochs 120 \
  --gen-init-ckpt "$MOTIUS_GENTRACK_G0_CHECKPOINT" \
  --trainee-init-ckpt "$MOTIUS_GENTRACK_TRAINEE_CHECKPOINT" \
  --initial-judge-onnx "$MOTIUS_GENTRACK_PROTOMOTIONS_ONNX" \
  --tracker-python "$MOTIUS_GENTRACK_PROTO_PYTHON" \
  --trainee-configuration "$MOTIUS_GENTRACK_PROTO_CONFIGURATION" \
  --root outputs/training/gentrack_coevolve
```

By default, the tracker consumes the online replay pool accumulated by the
generator. Large-cluster runs may replace the complete generated bank each
round. That mode is opt-in with `--refresh-generator-bank` and requires a
site-specific `--refresh-generator-command`; the command receives the
checkpoint, annotation, prompt-bank, count, sharding, GPU, and output paths
through the environment and must atomically publish `COVERAGE_READY.json`.
Keeping this launcher hook explicit avoids embedding private scheduler or
storage paths in the public repository.

## Controls and Ablations

The released configs preserve the matched update budget:

- `train_gentrack_offline_g1.py`: frozen selected-pool generator control.
- `ablation_rwsft_g1.py`: reward-weighted SFT instead of Flow-GRPO.
- `ablation_dpo_g1.py`: matched Flow-DPO preference update.
- `ablation_no_execution_g1.py`: disables execution advantage.
- `ablation_success_only_g1.py`: retains only the binary success signal.

All variants keep \(G_0\), the GT anchor cadence, optimizer budget, and data
contract fixed unless the named ablation explicitly changes them.

## Evaluation and Attestation

The unified evaluator operates on persisted reference and execution
trajectories:

```bash
python tools/gentrack/evaluate.py --help
python tools/gentrack/validate_qualification.py --help
python tools/gentrack/collect_runtime_receipt.py --help
```

Evaluation never substitutes kinematic generation for physical execution.
Continuous tracking errors include failed trajectories; success is the
protocol termination outcome. Runtime receipts hash generator weights,
tracker checkpoints, judge specs, manifests, and result summaries.

## Validation

```bash
python -m compileall -q motius tools/gentrack configs/gentrack
pytest -q \
  tests/test_gentrack_flow_grpo.py \
  tests/test_gentrack_immutable_anchor.py \
  tests/test_gentrack_evaluator.py
```

The first two tests exercise the algorithm without simulator startup. The
evaluator test validates trajectory protocol handling and aggregate metrics.
Physical end-to-end replay additionally requires the user-provided tracker
artifacts and simulator environment listed above.

## Native tracker validation

The native tracker uses `motius.trainers.protomotions.train`, the common
AccelerateRunner, and Isaac Lab. Each round emits a full native checkpoint and
a `last.ckpt` interchange file with current neural weights, all four Adam
states, advantage EMA, epoch and transition count. The co-evolution launcher
restores those states with `--resume-original`. ONNX export runs through
`tools/export_protomotions_onnx.py`; its observation, policy and PD composition
lives in the inference Pipeline.

Recorded checks cover [original-state PPO/AMP resume](../architecture/validation/protomotions-resume-source.json),
[real CLI checkpoint handoff](../architecture/protomotions-checkpoint-bridge.json),
[NPZ/CSV/PyRoki-output conversion and packaging](../architecture/gentrack-motion-conversion.json),
and a [128-frame original/native MuJoCo judge comparison](../architecture/gentrack-native-judge.json).
These checks establish the stated component equivalence. A complete GenTrack
co-evolution experiment and full-dataset metrics require separate runs.
