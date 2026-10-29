"""Actual released PRISM networks through the shared two-GPU Runner."""

from pathlib import Path
import argparse, json, os, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module(force=True)
class PRISMPreparedAuditDataset(Dataset):
    def __len__(self):
        return 32

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(317 + index)
        return dict(
            motion=torch.randn(17, 138, generator=generator) * 0.01,
            num_frames=17 if index % 2 == 0 else 13,
            caption="A person walks forward." if index % 2 == 0 else "A person turns.",
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    from mmengine import Config
    from motius.runner import AccelerateRunner
    from verify_motionstreamer_source import hashes
    import motius.models.prism
    import motius.trainers.prism
    import motius.hooks

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    cfg = Config(
        dict(
            custom_imports=dict(
                imports=[
                    "motius.models.prism",
                    "motius.trainers.prism",
                    "motius.hooks",
                ],
                allow_failed_imports=False,
            ),
            model=dict(
                type="PRISMBundle",
                checkpoint_path=str(args.artifact),
                device="cpu",
                training=True,
            ),
            trainer=dict(type="PrismTrainer", max_text_length=16),
            optimizer=dict(type="AdamW", lr=1e-5, weight_decay=0.01),
            train_dataloader=dict(
                batch_size=2,
                num_workers=0,
                shuffle=True,
                dataset=dict(type="PRISMPreparedAuditDataset"),
            ),
            train_cfg=dict(by_epoch=False, max_iters=3, max_grad_norm=1.0),
            accelerator=dict(
                mixed_precision="no",
                gradient_accumulation_steps=1,
                step_scheduler_with_optimizer=False,
                ddp_kwargs=dict(find_unused_parameters=True),
            ),
            cuda_precision=dict(allow_tf32=False, float32_matmul_precision="highest"),
            sdpa_backends=dict(flash=False, mem_efficient=False, math=True),
            default_hooks=dict(
                checkpoint=dict(
                    type="CheckpointHook", interval=3, save_last=True, max_keep_ckpts=3
                )
            ),
            work_dir=str(args.output_dir / "native"),
            seed=37,
        )
    )
    runner = AccelerateRunner.from_cfg(cfg)
    before = hashes(runner.accelerator.unwrap_model(runner.bundle.transformer))
    frozen = {
        name: hashes(getattr(runner.bundle, name)) for name in ["vae", "text_encoder"]
    }

    class Check:
        def after_run(self):
            assert runner.global_step == 3
            state = hashes(runner.accelerator.unwrap_model(runner.bundle.transformer))
            assert state != before
            assert frozen == {
                name: hashes(getattr(runner.bundle, name)) for name in frozen
            }
            replicas = [None] * runner.accelerator.num_processes
            torch.distributed.all_gather_object(replicas, state)
            assert all(value == replicas[0] for value in replicas)
            if runner.accelerator.is_main_process:
                report = dict(
                    complete=True,
                    scope="Actual released PRISM-KT transformer, frozen VAE/UMT5, native Trainer with raw captions and shared two-GPU Runner on explicit prepared physical-motion fixtures. Three AdamW updates; all transformer hashes synchronized, frozen roots unchanged and complete trainable/optimizer checkpoint. Full original corpus/AMP/progress protocol excluded.",
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
