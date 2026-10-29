"""Actual MotionCLR release: original networks, training objective and sampling."""

from pathlib import Path
from types import SimpleNamespace as NS
import argparse, ast, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch, numpy as np, clip
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
    options = {}
    for line in (args.artifact / "opt.txt").read_text().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        try:
            value = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            value = value.strip()
        options[key.strip()] = value
    options.update(dim_pose=263, is_train=True)
    opt = NS(**options)
    clip_state = dict(load_file(str(args.artifact / "clip.safetensors")))
    clip.load = lambda *a, **kw: (
        clip.model.build_model(clip_state.copy()).float(),
        None,
    )
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from models import build_models
        from utils.model_load import load_model_weights

        import yaml
        def options_tree(value):
            return NS(**{key: options_tree(item) for key,item in value.items()}) if isinstance(value,dict) else value
        edit_config=options_tree(yaml.safe_load((args.upstream/'options/noedit.yaml').read_text()))
        model = build_models(opt,edit_config=edit_config).cuda()
        load_model_weights(
            model, str(args.artifact / "latest.tar"), use_ema=True, device="cuda:0"
        )
        from trainers.ddpm_trainer import DDPMTrainer
        from diffusers import DDPMScheduler

        trainer = object.__new__(DDPMTrainer)
        trainer.opt = opt
        trainer.model = model
        trainer.device = torch.device("cuda")
        trainer.diffusion_steps = 1000
        trainer.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule=opt.beta_schedule,
            variance_type="fixed_small",
            prediction_type=opt.prediction_type,
            clip_sample=False,
        )
        trainer.mse_criterion = torch.nn.MSELoss(reduction="none")
    else:
        from motius.models.motionclr import MotionCLRBundle
        from motius.trainers import MotionCLRTrainer
        from motius.pipelines.motionclr import MotionCLRPipeline

        bundle = MotionCLRBundle.from_pretrained(
            str(args.artifact), device="cuda", torch_dtype="float32"
        )
        pipe = MotionCLRPipeline(bundle)
        model = bundle.network
        trainer = MotionCLRTrainer(bundle)
    result = dict(neural=hashes(model), training=[], inference=[])
    for case in range(2):
        frames = 48 if case == 0 else 52
        torch.manual_seed(71 + case)
        motion = torch.randn(2, frames, 263, device="cuda") * 0.03
        lengths = torch.tensor([frames, frames - 11], device="cuda")
        captions = [
            "A person walks forward.",
            "A person " + "raises both arms slowly " * 30,
        ]
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(912 + case)
        if args.worker == "upstream":
            trainer.forward((captions, motion, lengths))
            trainer.backward_G()
            loss = trainer.loss
        else:
            loss = trainer.train_step(
                dict(motion=motion, motion_lens=lengths, text=captions)
            )["loss"]
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
    mean = np.load(args.artifact / "mean.npy")
    std = np.load(args.artifact / "std.npy")
    for case, scheduler in enumerate(["dpmsolver", "ddim"]):
        lengths = [48, 41]
        seed = 813 + case
        steps = 10 if case == 0 else 50
        torch.manual_seed(seed)
        if args.worker == "upstream":
            from models.gaussian_diffusion import DiffusePipeline

            os.chdir(args.upstream)
            sampler = DiffusePipeline(
                opt,
                model,
                scheduler,
                steps,
                torch.device("cuda"),
                torch_dtype=torch.float32,
            )
            with torch.no_grad():
                samples = sampler.generate_batch(
                    captions, torch.tensor(lengths, device="cuda")
                )
            normalized = [samples[i, :length] for i, length in enumerate(lengths)]
            # Official dataset.inv_transform uses the supplied NumPy float64 statistics.
            physical = [
                (value.cpu().numpy() * std + mean).astype(np.float32)
                for value in normalized
            ]
        else:
            pipe = MotionCLRPipeline(
                bundle, diffuser_name=scheduler, num_inference_steps=steps
            )
            normalized = pipe.infer_t2m(
                captions, lengths, seed=seed, return_normalized=True
            )
            physical = pipe.infer_t2m(captions, lengths, seed=seed)
        result["inference"].append(
            tensors(dict(normalized=normalized, physical=physical))
        )
        print(args.worker, "MotionCLR full", scheduler, "sampling saved", flush=True)
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
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
        scope="Original MotionCLR source and SHA-pinned 6.7GB public EMA checkpoint: all neural hashes, original DDPMTrainer noise/timesteps, masked reconstruction losses, all gradients and CPU/CUDA RNG; full default 10-step DPM and 50-step DDIM samples with raw long/short text, normalized features and original float64-statistic denormalization at the common float32 output boundary. Controlled FP32/deterministic-convolution/math attention. FP16, edits, raw-data/EMA/optimizer/full training protocol excluded.",
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in [] if args.reuse_results else ["upstream", "native"]:
        subprocess.run(
            [
                sys.executable,
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
    print("Original MotionCLR objectives, gradients, RNG and complete sampling exact")


if __name__ == "__main__":
    main()
