"""Native joint TM2D through the shared two-GPU training Runner."""

from pathlib import Path
import argparse, json, os, sys

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
    cfg = Config.fromfile(str(ROOT / "configs/tm2d/prepared_tokens.py"))
    cfg.train_cfg.max_iters = 3
    cfg.train_dataloader.batch_size = 2
    cfg.default_hooks.checkpoint.interval = 3
    runner = AccelerateRunner.from_cfg(cfg)
    before = hashes(runner.accelerator.unwrap_model(runner.bundle.joint_network))
    frozen = {
        name: hashes(getattr(runner.bundle, name))
        for name in ["vq_encoder", "quantizer", "vq_decoder"]
    }

    class Check:
        def after_run(self):
            assert runner.global_step == 3
            state = hashes(runner.accelerator.unwrap_model(runner.bundle.joint_network))
            assert state != before
            assert frozen == {
                name: hashes(getattr(runner.bundle, name)) for name in frozen
            }
            replicas = [None] * runner.accelerator.num_processes
            torch.distributed.all_gather_object(replicas, state)
            assert all(value == replicas[0] for value in replicas)
            if runner.accelerator.is_main_process:
                saved = torch.load(
                    Path(runner.work_dir) / "checkpoint-iter_3/model.pt",
                    map_location="cpu",
                    weights_only=False,
                    mmap=True,
                )
                assert {"joint_network", *frozen} <= saved.keys()
                args.output_dir.mkdir(parents=True, exist_ok=True)
                report = dict(
                    complete=True,
                    scope="Released full TM2D networks, native shared-audio-decoder joint Trainer, prepared padded tokens/music fixture and shared two-GPU Runner. Three Adam updates; every joint neural hash synchronized, all frozen VQ roots unchanged and checkpointed. Full original corpus/epoch protocol excluded.",
                    world_size=runner.accelerator.num_processes,
                    updates=3,
                    neural_tensors=len(state),
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
