"""Exercise native DART AdamW/EMA through the shared two-GPU Runner."""

from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from mmengine import Config
    from motius.runner import AccelerateRunner
    from verify_motionstreamer_source import hashes

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    config = Config.fromfile(str(ROOT / "configs/dart/prepared_motion.py"))
    config.train_cfg.max_iters = 3
    config.train_dataloader.batch_size = 2
    config.train_dataloader.num_workers = 0
    config.default_hooks.checkpoint.interval = 3
    runner = AccelerateRunner.from_cfg(config)
    trainer = runner.trainer
    network = runner.accelerator.unwrap_model(
        getattr(runner.bundle, trainer.active_name)
    )
    frozen_name = "vae_model" if trainer.stage == "denoiser" else "denoiser_model"
    before = hashes(network)
    before_ema = hashes(getattr(runner.bundle, trainer.ema_name))
    frozen = hashes(getattr(runner.bundle, frozen_name))

    class Check:
        def after_run(self):
            assert runner.global_step == 3
            states = dict(
                online=hashes(
                    runner.accelerator.unwrap_model(
                        getattr(runner.bundle, trainer.active_name)
                    )
                ),
                ema=hashes(getattr(runner.bundle, trainer.ema_name)),
                frozen=hashes(getattr(runner.bundle, frozen_name)),
            )
            assert states["online"] != before and states["ema"] != before_ema
            assert states["frozen"] == frozen
            replicas = [None] * runner.accelerator.num_processes
            torch.distributed.all_gather_object(replicas, states)
            assert all(value == replicas[0] for value in replicas)
            if runner.accelerator.is_main_process:
                saved = Path(runner.work_dir) / "checkpoint-iter_3/model.pt"
                checkpoint = torch.load(
                    saved, map_location="cpu", weights_only=False, mmap=True
                )
                assert {
                    trainer.active_name,
                    trainer.ema_name,
                    frozen_name,
                } <= checkpoint.keys()
                args.output_dir.mkdir(parents=True, exist_ok=True)
                report = dict(
                    complete=True,
                    scope="Original released DART weights, native primitive Trainer/full configured body losses, original AdamW/clip/LR/EMA and shared two-GPU Runner on explicit prepared NPZ fixtures. Three updates, every online/EMA/frozen replica hash synchronized, frozen root unchanged, complete checkpoint. No original full corpus/multi-primitive curriculum/AMP or implicit save-time VAE calibration claim.",
                    stage=trainer.stage,
                    world_size=runner.accelerator.num_processes,
                    updates=3,
                    online_tensors=len(states["online"]),
                    ema_tensors=len(states["ema"]),
                    all_replica_hashes_equal=True,
                )
                (args.output_dir / "source_parity.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            runner.accelerator.wait_for_everyone()

    hook = Check()
    hook.runner = runner
    runner.hooks.append(hook)
    runner.train()


if __name__ == "__main__":
    main()
