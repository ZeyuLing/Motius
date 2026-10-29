"""Released MDM-family networks through the shared two-GPU native Runner."""

from pathlib import Path
import argparse
import importlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=[
            "mdm",
            "condmdi",
            "omnicontrol",
            "projflow",
            "motionclr",
            "mld",
            "motionlcm",
        ],
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=["vae", "diffusion"], default="diffusion")
    parser.add_argument("--teacher", type=Path)
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.output_dir = args.output_dir.resolve()
    import torch
    from torch.utils.data import Dataset
    from mmengine import Config
    from motius.registry import DATASETS
    from motius.runner import AccelerateRunner
    from verify_motionstreamer_source import hashes
    import motius.hooks

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    importlib.import_module("motius.models." + args.method)
    importlib.import_module("motius.trainers." + args.method)
    names = {
        "motionlcm": ("MotionLCMBundle", "MotionLCMTrainer"),
        "mld": ("MLDBundle", "MLDTrainer"),
        "projflow": ("ProjFlowBundle", "ProjFlowTrainer"),
        "motionclr": ("MotionCLRBundle", "MotionCLRTrainer"),
        "mdm": ("MDMBundle", "MDMTrainer"),
        "condmdi": ("CondMDIBundle", "CondMDITrainer"),
        "omnicontrol": ("OmniControlBundle", "OmniControlTrainer"),
    }
    bundle_name, trainer_name = names[args.method]
    meta_path = args.artifact / (args.method + "_config.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    root = "network" if args.method == "motionclr" else "net"
    model = dict(
        type=bundle_name,
        config=meta.get("config"),
        weights_path=str(args.artifact / "model.safetensors"),
        clip_weights_path=str(args.artifact / "clip.safetensors"),
        device="cpu",
    )
    if args.method in {"mld", "motionlcm"}:
        root = "vae" if args.stage == "vae" else "denoiser"
        model = dict(
            type=bundle_name,
            config=meta["config"],
            vae_weights_path=str(args.artifact / "vae.safetensors"),
            denoiser_weights_path=str(args.artifact / "denoiser.safetensors"),
            mean_path=str(args.artifact / "Mean.npy"),
            std_path=str(args.artifact / "Std.npy"),
            text_encoder_name=str(args.artifact / meta["text_encoder_name"]),
            device="cpu",
        )
    elif args.method == "motionclr":
        from motius.models.motionclr.bundle import (
            _network_config_from_options,
            _parse_official_options,
        )

        model = dict(
            type=bundle_name,
            network_config=_network_config_from_options(
                _parse_official_options(args.artifact / "opt.txt")
            ),
            checkpoint_path=str(args.artifact / "latest.tar"),
            mean_path=str(args.artifact / "mean.npy"),
            std_path=str(args.artifact / "std.npy"),
            device="cpu",
            torch_dtype="float32",
        )
    elif args.method == "projflow":
        model.pop("config")
        model.update(
            model_id=meta["model_id"],
            mean_path=str(args.artifact / "joints_mean.npy"),
            std_path=str(args.artifact / "joints_std.npy"),
        )
    elif args.method == "condmdi":
        model.update(
            mean_abs_path=str(args.artifact / "Mean_abs_3d.npy"),
            std_abs_path=str(args.artifact / "Std_abs_3d.npy"),
        )
    else:
        model.update(
            mean_path=str(args.artifact / "Mean.npy"),
            std_path=str(args.artifact / "Std.npy"),
        )
        if args.method == "omnicontrol":
            model.update(
                raw_mean_path=str(args.artifact / "Mean_raw.npy"),
                raw_std_path=str(args.artifact / "Std_raw.npy"),
            )

    @DATASETS.register_module()
    class GaussianMotionAuditDataset(Dataset):
        def __len__(self):
            return 32

        def __getitem__(self, index):
            generator = torch.Generator().manual_seed(71 + index)
            motion = torch.randn(263, 1, 48, generator=generator) * 0.03
            length = 48 if index % 2 else 40
            motion[..., length:] = 0
            if args.method == "projflow":
                motion = torch.randn(48, 22, 3, generator=generator) * 0.03
                motion[length:] = 0
            elif args.method in {"motionclr", "mld", "motionlcm"}:
                motion = torch.randn(48, 263, generator=generator) * 0.03
                motion[length:] = 0
            output = dict(
                motion=motion,
                motion_lens=torch.tensor(length),
                text=["A person walks.", "A person raises both arms."][index % 2],
            )
            if args.method == "condmdi":
                mask = torch.zeros_like(motion, dtype=torch.bool)
                mask[..., ::8] = True
                mask[..., length:] = False
                output.update(obs_mask=mask, obs_x0=motion)
            elif args.method == "omnicontrol":
                hint = torch.zeros(48, 66)
                hint[:length:8, :3] = 0.1
                output["hint"] = hint
            return output

    trainer = dict(type=trainer_name)
    if args.method == "mld":
        trainer["stage"] = args.stage
    elif args.method == "motionlcm":
        if args.teacher is None:
            raise ValueError("Pass explicit MotionLCM teacher weights")
        trainer["teacher_weights_path"] = str(args.teacher)
    cfg = Config(
        dict(
            work_dir=str(args.output_dir / "native"),
            model=model,
            trainer=trainer,
            optimizer=dict(type="AdamW", lr=1e-4, weight_decay=0.0),
            train_dataloader=dict(
                batch_size=2,
                num_workers=0,
                shuffle=True,
                drop_last=True,
                dataset=dict(type="GaussianMotionAuditDataset"),
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
                    type="CheckpointHook", interval=3, max_keep_ckpts=1, save_last=True
                )
            ),
            seed=37,
        )
    )
    runner = AccelerateRunner.from_cfg(cfg)
    net = runner.accelerator.unwrap_model(getattr(runner.bundle, root))
    before = hashes(net)
    frozen_model = (
        runner.bundle.text_encoder
        if args.method in {"mld", "motionlcm"}
        else net.clip_model
    )
    frozen = hashes(frozen_model)
    frozen_motion_root = "denoiser" if args.stage == "vae" else "vae"
    frozen_motion = (
        hashes(getattr(runner.bundle, frozen_motion_root))
        if args.method in {"mld", "motionlcm"}
        else None
    )
    if args.method == "motionlcm":
        frozen_teacher = hashes(runner.bundle.teacher_denoiser)
        initial_target = hashes(runner.bundle.target_denoiser)

    class Check:
        def after_run(self):
            assert runner.global_step == 3
            net = runner.accelerator.unwrap_model(getattr(runner.bundle, root))
            state = hashes(net)
            replicas = [None] * runner.accelerator.num_processes
            torch.distributed.all_gather_object(replicas, state)
            assert all(
                value == replicas[0] for value in replicas
            ), "Neural replicas diverged"
            assert hashes(frozen_model) == frozen, "Frozen CLIP changed"
            if args.method in {"mld", "motionlcm"}:
                assert (
                    hashes(getattr(runner.bundle, frozen_motion_root)) == frozen_motion
                )
            assert state != before, "No learned state changed"
            if args.method == "motionlcm":
                assert hashes(runner.bundle.teacher_denoiser) == frozen_teacher
                target = hashes(runner.bundle.target_denoiser)
                assert target != initial_target, "EMA target did not change"
                targets = [None] * runner.accelerator.num_processes
                torch.distributed.all_gather_object(targets, target)
                assert all(value == targets[0] for value in targets)
            assert all(
                not p.requires_grad and p.grad is None
                for p in frozen_model.parameters()
            )
            if runner.accelerator.is_main_process:
                path = args.output_dir / "native/checkpoint-iter_3/model.pt"
                assert path.is_file()
                checkpoint = torch.load(
                    path, map_location="cpu", weights_only=False, mmap=True
                )
                assert root in checkpoint
                if args.method in {"mld", "motionlcm"}:
                    assert {"vae", "denoiser", "text_encoder"} <= checkpoint.keys()
                    if args.method == "motionlcm":
                        assert {
                            "teacher_denoiser",
                            "target_denoiser",
                        } <= checkpoint.keys()
                else:
                    assert any(
                        name.startswith("clip_model.") for name in checkpoint[root]
                    )
                report = dict(
                    complete=True,
                    scope=args.method
                    + " released networks, native prepared-motion Trainer and actual shared AccelerateRunner: three two-GPU updates, distinct rank-local inputs, all neural replica hashes synchronized, frozen CLIP unchanged and full network checkpoint saved. Synthetic normalized-motion/observation fixture; not original distributed optimizer/AMP/corpus protocol.",
                    world_size=runner.accelerator.num_processes,
                    updates=3,
                    stage=args.stage if args.method == "mld" else None,
                    neural_tensors=len(state),
                    all_replica_neural_hashes_equal=True,
                    frozen_clip_unchanged=True,
                )
                args.output_dir.mkdir(parents=True, exist_ok=True)
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
