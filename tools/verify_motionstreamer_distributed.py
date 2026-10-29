"""Run the native AR recipe and check every trained parameter across DDP replicas."""

from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from mmengine import Config
    from motius.runner.accelerate_runner import AccelerateRunner
    from verify_motionstreamer_source import hashes
    import motius.hooks
    import motius.models.motionstreamer
    import motius.trainers.motionstreamer
    import motius.datasets.motionstreamer

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    os.environ.update(
        MOTIUS_MS_STAGE="ar",
        MOTIUS_MS_ARTIFACT=str(args.artifact.resolve()),
        MOTIUS_MS_HML_ROOT=str(args.fixture.resolve() / "humanml3d_272"),
        MOTIUS_MS_LATENTS=str(args.fixture.resolve() / "latents"),
        MOTIUS_MS_TOTAL_STEPS="17",
        MOTIUS_MS_WARMUP_STEPS="2",
        MOTIUS_MS_BATCH_SIZE="2",
        MOTIUS_MS_WORKERS="0",
    )
    config = Config.fromfile(
        str(
            Path(__file__).resolve().parents[1]
            / "configs/motionstreamer/train_motionstreamer.py"
        )
    )
    config.work_dir = str(args.output_dir.resolve() / "native")
    config.train_cfg.max_iters = 3
    config.default_hooks.checkpoint.save_last = False
    config.default_hooks.logger.iter_interval = 1
    runner = AccelerateRunner.from_cfg(config)

    class ReplicaCheck:
        def after_run(self):
            assert runner.global_step == 3
            assert not torch.backends.cuda.matmul.allow_tf32
            assert not torch.backends.cudnn.allow_tf32
            assert not torch.backends.cuda.flash_sdp_enabled()
            assert not torch.backends.cuda.mem_efficient_sdp_enabled()
            assert torch.backends.cuda.math_sdp_enabled()
            model = runner.accelerator.unwrap_model(runner.bundle.ar)
            state = hashes(model)
            collected = [None] * runner.accelerator.num_processes
            torch.distributed.all_gather_object(collected, state)
            assert all(
                candidate == collected[0] for candidate in collected
            ), "DDP neural replicas diverged"
            scheduler = next(iter(runner.lr_schedulers.values()))
            schedule_state = scheduler.state_dict()
            assert schedule_state["iteration"] == 3
            if runner.accelerator.is_main_process:
                report = dict(
                    complete=True,
                    scope="Native shared Runner, real SentenceT5 and original token reader, three AR updates on two H200 GPUs. Static-graph DDP supports the original detached attention-scale parameters and diffusion head execution in the Trainer. Every trained neural tensor SHA256 identical across replicas and exactly one LR step per optimizer update. Synthetic filesystem fixture; not independent original distributed training or full-dataset convergence.",
                    cuda_precision=dict(allow_tf32=False, float32_matmul_precision=torch.get_float32_matmul_precision()),
                    sdpa_backends=dict(flash=False, mem_efficient=False, math=True),
                    world_size=runner.accelerator.num_processes,
                    updates=3,
                    neural_tensors=len(state),
                    all_replica_neural_tensors_sha256_equal=True,
                    scheduler_iterations=schedule_state["iteration"],
                )
                args.output_dir.mkdir(parents=True, exist_ok=True)
                (args.output_dir / "source_parity.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )
            runner.accelerator.wait_for_everyone()

    hook = ReplicaCheck()
    hook.runner = runner
    runner.hooks.append(hook)
    runner.train()


if __name__ == "__main__":
    main()
