"""Released ProjFlow network, flow objective, gradients and control sampler vs source."""

from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch, numpy as np
    from safetensors.torch import load_file
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    meta = json.loads((args.artifact / "projflow_config.json").read_text())
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        if "float" not in np.__dict__:
            np.float = float
        import clip

        clip.load = lambda *a, **kw: (
            clip.model.build_model(
                dict(load_file(str(args.artifact / "clip.safetensors")))
            ).float(),
            None,
        )
        from models.ACMDM import ACMDM_models

        model = ACMDM_models[meta["model_id"]](input_dim=3, cond_mode="text").cuda()
        missing, unexpected = model.load_state_dict(
            load_file(str(args.artifact / "model.safetensors")), strict=False
        )
        assert not unexpected and all(
            name.startswith("clip_model.") for name in missing
        ), (missing, unexpected)
    else:
        from motius import Pipeline
        from motius.trainers import ProjFlowTrainer

        pipe = Pipeline.from_pretrained(
            str(args.artifact), bundle_kwargs={"device": "cuda"}
        )
        model = pipe.bundle.net
        trainer = ProjFlowTrainer(pipe.bundle)
    result = dict(neural=hashes(model), training=[], inference=[])
    for iteration in range(2):
        torch.manual_seed(51 + iteration)
        motion = torch.randn(2, 48, 22, 3, device="cuda") * 0.03
        lengths = torch.tensor([48, 36], device="cuda")
        captions = [
            "A person walks forward.",
            "A person " + "raises both arms slowly " * 30,
        ]
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(711 + iteration)
        loss = (
            model.forward_loss(motion.permute(0, 3, 1, 2), captions, lengths)
            if args.worker == "upstream"
            else trainer.train_step(
                dict(motion=motion, motion_lens=lengths, text=captions)
            )["loss"]
        )
        loss.backward()
        result["training"].append(
            tensors(
                dict(
                    loss=loss,
                    gradients=hashes(model, gradients=True),
                    cpu_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state(),
                )
            )
        )
    model.eval()
    mean = torch.from_numpy(np.load(args.artifact / "joints_mean.npy")).cuda()
    std = torch.from_numpy(np.load(args.artifact / "joints_std.npy")).cuda()
    physical = (
        torch.sin(
            torch.arange(2 * 48 * 22 * 3, device="cuda").reshape(2, 48, 22, 3) * 0.017
        )
        * 0.03
    )
    physical[..., 1] += 0.9
    arrays = [physical[0].cpu().numpy(), physical[1, :36].cpu().numpy()]
    for case in range(2):
        scale = 2.5 if case == 0 else 1.0
        seed = 821 + case
        if args.worker == "upstream":
            control = (physical - mean) / std
            mask = torch.zeros_like(physical)
            for index, length in enumerate([48, 36]):
                for frame in [0, length - 1]:
                    mask[index, frame] = 1.0
            control = torch.where(
                mask.bool(), control, torch.zeros_like(control)
            ).permute(0, 3, 1, 2)
            mask = mask.permute(0, 3, 1, 2)
            valid = (
                torch.arange(48, device="cuda")[None]
                < torch.tensor([48, 36], device="cuda")[:, None]
            )
            attention = (
                valid.unsqueeze(-1)
                .repeat(1, 1, model.patches_per_frame)
                .flatten(1)
                .unsqueeze(1)
                .unsqueeze(1)
            )
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                conds = model.encode_text(captions)
                noise = torch.randn(2, 3, 48, 22, device="cuda")
                if scale != 1.0:
                    noise = torch.cat([noise, noise])
                    conds = torch.cat([conds, torch.zeros_like(conds)])
                    attention = attention.repeat(2, 1, 1, 1)
                    control = control.repeat(2, 1, 1, 1)
                    mask = mask.repeat(2, 1, 1, 1)
                sample = model.gen_diffusion.sample_projflow(
                    num_steps=100, use_projflow=case == 0
                )(
                    noise,
                    model.forward_with_CFG,
                    conds=conds,
                    attention_mask=attention,
                    cfg=scale,
                    A=mask,
                    y=control,
                )[
                    -1
                ]
                if scale != 1.0:
                    sample, _ = sample.chunk(2)
                world = sample.permute(0, 2, 3, 1) * std + mean
                output = [world[i, :length] for i, length in enumerate([48, 36])]
        else:
            output = pipe.infer_control(
                captions,
                arrays,
                lengths=[48, 36],
                control_mode="first_last",
                guidance_scale=scale,
                num_steps=100,
                seed=seed,
                use_projflow=case == 0,
            )
        result["inference"].append(tensors(output))
        print(
            args.worker,
            "original/native ProjFlow sampling case",
            case,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    parser.add_argument("--source-python", type=Path)
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        tests=[],
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        scope="Released ACMDM Raw Flow S prior, original network/transport objective, all gradient hashes and CPU/CUDA RNG; two ragged 48-frame minibatches with raw long/short text. Two full 100-step original samplers versus native public control Pipeline, with and without ProjFlow projection and CFG. Original corpus ingestion, EMA/optimizer and full training protocol excluded. Controlled FP32/math-attention/deterministic-convolution runtime.",
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in [] if args.reuse_results else ["upstream", "native"]:
        subprocess.run(
            [
                (
                    str(args.source_python)
                    if kind == "upstream" and args.source_python
                    else sys.executable
                ),
                str(Path(__file__).resolve()),
                "--upstream",
                str(args.upstream),
                "--artifact",
                str(args.artifact),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
            check=True,
        )
    try:
        compare_full(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("ProjFlow original objectives, gradients, RNG and control sampling exact")


if __name__ == "__main__":
    main()
