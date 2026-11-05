"""Train the released MotionCanvas network with native conditions and losses."""

from pathlib import Path
import argparse, json, os, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module(force=True)
class MotionCanvasPreparedAuditDataset(Dataset):
    def __init__(self, artifact):
        import numpy as np

        artifact = Path(artifact)
        self.mean = torch.from_numpy(np.load(artifact / "Mean.npy")).float()
        self.std = torch.from_numpy(np.load(artifact / "Std.npy")).float()

    def __len__(self):
        return 32

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(391 + index)
        motion = torch.randn(12, 198, generator=generator) * 0.03 * self.std + self.mean
        length = 12 if index % 2 else 9
        motion[length:] = 0
        mask = torch.zeros_like(motion)
        mask[2:length:3] = 1
        return dict(
            src_motion=motion.clone(),
            tgt_motion=motion,
            src_mask=mask,
            src_length=length,
            tgt_length=length,
            edit_mode=bool(index % 2),
            data_src="Game" if index % 3 else "Other",
            text_vec_raw=torch.randn(1, 768, generator=generator) * 0.03,
            text_ctxt_raw=torch.randn(4, 4096, generator=generator) * 0.03,
            text_ctxt_raw_length=torch.tensor(4 if index % 2 else 2),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from torch.utils.data import Dataset
    from mmengine import Config
    from motius.models.motioncanvas import MotionCanvasBundle
    from motius.registry import DATASETS
    from motius.runner import AccelerateRunner
    from verify_motionstreamer_source import hashes
    import motius.trainers.motioncanvas, motius.hooks

    assert os.uname().nodename == "zeyuling-g2-depth11-eval60k-4g-1020-worker-0"
    artifact = args.artifact.resolve()
    model = json.loads((artifact / "motioncanvas_config.json").read_text())["config"]
    model.pop("format", None)
    model["text_encoder"] = MotionCanvasBundle._resolve_text_encoder_paths(
        model["text_encoder"], artifact
    )
    model.update(
        type="MotionCanvasBundle",
        motion_weights_path=str(artifact / "motion_transformer.safetensors"),
        mean_path=str(artifact / "Mean.npy"),
        std_path=str(artifact / "Std.npy"),
        bone_offsets_path=str(artifact / "bone_offsets_22.pt"),
    )
    config = Config(
        dict(
            model=model,
            trainer=dict(type="MotionCanvasTrainer"),
            optimizer=dict(type="AdamW", lr=1e-5, weight_decay=0.0),
            train_dataloader=dict(
                batch_size=2,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                dataset=dict(
                    type="MotionCanvasPreparedAuditDataset", artifact=str(artifact)
                ),
            ),
            train_cfg=dict(by_epoch=False, max_iters=3, max_grad_norm=2.0),
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
                    type="CheckpointHook", interval=3, save_last=True, max_keep_ckpts=1
                )
            ),
            work_dir=str(args.output_dir / "native"),
            seed=37,
        )
    )
    runner = AccelerateRunner.from_cfg(config)
    before = hashes(runner.accelerator.unwrap_model(runner.bundle.motion_transformer))

    class Check:
        def after_run(self):
            assert runner.global_step == 3
            network = runner.accelerator.unwrap_model(runner.bundle.motion_transformer)
            from verify_motionstreamer_source import digest

            state = dict(
                neural=hashes(network),
                orphan={
                    name: digest(value)
                    for name, value in runner.bundle.named_parameters()
                    if not name.startswith("motion_transformer.")
                },
            )
            replicas = [None] * runner.accelerator.num_processes
            torch.distributed.all_gather_object(replicas, state)
            assert all(value == replicas[0] for value in replicas), "Replicas diverged"
            assert state["neural"] != before
            if runner.accelerator.is_main_process:
                report = dict(
                    complete=True,
                    scope="Actual released MotionCanvas 0.46B weights, native ordinary Trainer/full configured geometric flow losses and shared Runner: three two-GPU updates on distinct prepared physical-motion/text-feature fixtures, all neural and orphan-condition replica hashes synchronized and checkpoint written. No raw-text encoder, raw corpus, sparse-rollout trainer or original distributed training protocol claim.",
                    world_size=runner.accelerator.num_processes,
                    updates=3,
                    neural_tensors=len(state["neural"]),
                    orphan_condition_tensors=len(state["orphan"]),
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
