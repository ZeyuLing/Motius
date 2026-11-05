# Native ProtoMotions G1 training

The public `ProtoMotionsTrainer` is the framework `ProtoMotionsPPOTrainer`. Training uses the shared AccelerateRunner, native rollout/GAE/replay data components and the native G1 Isaac Lab environment. No original application package or import alias is required.

Install Isaac Sim 5.1 and Isaac Lab 2.3.2 in a dedicated Python 3.11 environment. Keep the simulator's compatible Torch/NumPy versions. The validated runtime is recorded in `docs/architecture/isaac51-runtime.json`.

Download the released G1 checkpoint and its resolved training/inference PT and YAML configuration files. Materialize only the source-pinned G1 assets:

```bash
python tools/download_protomotions_assets.py
python -m motius.trainers.protomotions.train \
  --checkpoint /path/to/protomotions/last.ckpt \
  --motion-file /path/to/g1/reference.motion \
  --asset-root checkpoints/robots/protomotions \
  --num-envs 4096 --headless --run-name g1_bones_seed
```

The checkpoint example must point to the ProtoMotions G1 release, accompanied by `resolved_configs.yaml` and `resolved_configs.pt`; deployment export uses `resolved_configs_inference.yaml`. Explicit `--configuration` and `--environment-configuration` paths are also accepted. `--max-iterations`, `--batch-size` and `--discriminator-batch-size` override the corresponding training budgets. Existing native Accelerate checkpoints can be restored with `--auto-resume`. By default an original checkpoint initializes the neural state with fresh optimizers. `--resume-original` also restores the four original Adam states, advantage EMA, epoch and transition count; original replay data is initialized by the same setup protocol. Full native checkpoints use `--auto-resume`.

For distributed training, launch the same module through Accelerate with the desired process count. Each process receives its own simulator environment and local reference sampler; normalization moments follow the original distributed merge order. Current-weight Pipeline artifacts are written under the run's `exported` directory.

Validation records:

- `protomotions-physical-motion-source.json`: independent source FK and reference interpolation.
- `protomotions-physical-tracking-source.json`: original evaluator and native Pipeline in four real G1 environments, 60 steps. Actions, policy observations, poses, rewards and RNG are bitwise equal. PhysX contact-force diagnostics are compared separately against measured source-self-repeat variation.
- `protomotions-physical-training-source.json`: original AMP.fit versus native AccelerateRunner, two 32x4 physical rollouts and complete PPO/AMP updates, 1,670 exact comparisons.
- `protomotions-distributed-runner-source.json`: two-GPU Fabric versus Accelerate full updates with distinct rank-local tensor environments.

The physical comparisons use a short synthetic FK motion reference and controlled FP32/eager execution. They do not establish full-dataset tracking quality, long-run training convergence, or default compiled-runtime equivalence.
