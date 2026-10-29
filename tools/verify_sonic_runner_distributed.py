"""Exercise native SONIC online PPO through the shared Runner on two GPUs."""

from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from accelerate import Accelerator
    from motius.configuration.sonic import original_training_options
    from motius.runner.sonic_tracking import build_sonic_runner
    from verify_sonic_online_source import TensorEnvironment, SCHEMA
    from verify_motionstreamer_source import hashes

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    accelerator = Accelerator(mixed_precision="no")
    torch.manual_seed(1729 + accelerator.process_index)
    checkpoint = args.artifact.resolve() / "sonic_release/last.pt"
    options = original_training_options(checkpoint.parent / "config.yaml")
    options.update(steps=6, iterations=3, save_interval=3)
    options["trainer"].update(learning_epochs=2, num_mini_batches=2)
    runner, collector = build_sonic_runner(
        TensorEnvironment(),
        args.output_dir / "native",
        checkpoint=checkpoint,
        schema=SCHEMA,
        options=options,
        accelerator=accelerator,
    )

    class Verify:
        def after_run(self):
            assert runner.global_step == 3
            state = {
                name: hashes(accelerator.unwrap_model(getattr(runner.bundle, name)))
                for name in ["policy", "critic"]
            }
            replicas = [None] * accelerator.num_processes
            torch.distributed.all_gather_object(replicas, state)
            assert all(
                value == replicas[0] for value in replicas
            ), "SONIC replicas diverged"
            assert len(runner.trainer.last_updates) == 4
            assert collector.environment.tick == 18
            if accelerator.is_main_process:
                checkpoints = list(
                    (args.output_dir / "native").glob("checkpoint-iter_3*")
                )
                assert checkpoints, "Shared Runner checkpoint missing"
                report = dict(
                    complete=True,
                    world_size=accelerator.num_processes,
                    rollout_iterations=3,
                    optimizer_updates=12,
                    neural_tensors=sum(map(len, state.values())),
                    all_replica_state_sha256_equal=True,
                    scope="Released SONIC neural roots and native online collector/PPO Trainer "
                    "through the shared AccelerateRunner, two GPUs, three rollouts and "
                    "twelve AdamW updates; all policy/value parameters and buffers agree "
                    "across replicas. Checkpoint saved. Controlled tensor environment; "
                    "not independent original distributed or physical simulation evidence.",
                )
                (args.output_dir / "source_parity.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            accelerator.wait_for_everyone()

    hook = Verify()
    hook.runner = runner
    runner.hooks.append(hook)
    runner.train()


if __name__ == "__main__":
    main()
