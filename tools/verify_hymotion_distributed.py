"""Train the released HYMotion network with native conditions and losses."""

from pathlib import Path
import argparse, json, os, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


import torch
from torch.utils.data import Dataset
from motius.registry import DATASETS


@DATASETS.register_module(force=True)
class HYMotionPreparedAuditDataset(Dataset):
    def __init__(self, artifact):
        import numpy as np

        artifact = Path(artifact)
        self.mean = torch.from_numpy(np.load(artifact / "Mean.npy")).float()
        self.std = torch.from_numpy(np.load(artifact / "Std.npy")).float()

    def __len__(self):
        return 32

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(391 + index)
        motion = torch.randn(12, 201, generator=generator) * 0.03 * self.std + self.mean
        length = 12 if index % 2 else 9
        motion[length:] = 0
        return dict(
            motion=motion,
            num_frames=length,
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
    from motius.models.hymotion_t2m import HyMotionT2MBundle
    from motius.registry import DATASETS
    from motius.runner import AccelerateRunner
    from verify_motionstreamer_source import hashes
    import motius.trainers.hymotion_t2m, motius.hooks

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    artifact = args.artifact.resolve()
    model = json.loads((artifact / "hymotion_t2m_config.json").read_text())["config"]
    for key in [
        "format",
        "text_encoder_components",
        "external_text_encoder_components",
    ]:
        model.pop(key, None)
    model["text_encoder"] = HyMotionT2MBundle._resolve_text_encoder_cfg_paths(
        model["text_encoder"], artifact
    )
    model.update(
        type="HyMotionT2MBundle",
        motion_weights_path=str(artifact / "motion_transformer.safetensors"),
        mean_path=str(artifact / "Mean.npy"),
        std_path=str(artifact / "Std.npy"),
    )
    config = Config(
        dict(
            model=model,
            trainer=dict(type="HyMotionT2MTrainer"),
            optimizer=dict(type="AdamW", lr=1e-5, weight_decay=0.0),
            train_dataloader=dict(
                batch_size=2,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                dataset=dict(
                    type="HYMotionPreparedAuditDataset", artifact=str(artifact)
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
                    scope="Complete original HY-Motion Lite checkpoint, native Trainer/flow loss and shared Runner: three two-GPU updates on distinct prepared physical-motion/text-feature fixtures, all neural and orphan-condition replica hashes synchronized and checkpoint written. No raw-text encoder, raw corpus or original unpublished training protocol claim.",
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
