"""Official CLIP-conditioned MLD against the author's original checkpoint/code."""

from pathlib import Path
from types import SimpleNamespace as NS
import argparse, ast, inspect, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch, numpy as np
    from diffusers import DDIMScheduler
    from verify_motionstreamer_source import digest, hashes
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    metadata = json.loads((args.stats / "mld_config.json").read_text())["config"]
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from mld.models.architectures.mld_vae import MldVae
        from mld.models.architectures.mld_denoiser import MldDenoiser
        from mld.models.architectures.mld_clip import MldTextEncoder

        vae = MldVae(ablation=NS(MLP_DIST=False, PE_TYPE="mld"), **metadata["vae"])
        denoiser = MldDenoiser(
            ablation=NS(SKIP_CONNECT=True, VAE_TYPE="mld", DIFF_PE_TYPE="mld"),
            **metadata["denoiser"],
        )
        text = MldTextEncoder(str(args.text))
        state = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False, mmap=True
        )["state_dict"]
        for name, model in [("vae", vae), ("denoiser", denoiser)]:
            model.load_state_dict(
                {
                    key[len(name) + 1 :]: value
                    for key, value in state.items()
                    if key.startswith(name + ".")
                },
                strict=True,
            )
        tree = ast.parse((args.upstream / "mld/models/modeltype/mld.py").read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "MLD"
        )
        function = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "_diffusion_reverse"
        )
        namespace = dict(torch=torch, inspect=inspect)
        exec(
            compile(
                ast.Module(body=[function], type_ignores=[]),
                str(args.upstream / "mld/models/modeltype/mld.py"),
                "exec",
            ),
            namespace,
        )
    else:
        from motius.models.mld import MLDBundle
        from motius.pipelines.mld import MLDPipeline

        bundle = MLDBundle(
            model_ckpt=str(args.checkpoint),
            text_encoder_name=str(args.text),
            mean_path=str(args.stats / "Mean.npy"),
            std_path=str(args.stats / "Std.npy"),
            device="cuda",
        )
        pipe = MLDPipeline(bundle)
        vae, denoiser, text = bundle.vae, bundle.denoiser, bundle.text_encoder
    for model in [vae, denoiser, text]:
        model.cuda().eval()
    # Only text modules used by get_text_features belong in the native bundle.
    text_state = text.text_model.state_dict()
    text_hashes = {
        key: digest(value)
        for key, value in text_state.items()
        if not key.startswith(("vision_model.", "visual_projection."))
        and key != "logit_scale"
    }
    neural = hashes(denoiser)
    neural.pop("mem_pos.pe", None)
    result = dict(
        neural=dict(vae=hashes(vae), denoiser=neural, text=text_hashes), cases=[]
    )
    if args.training:
        training_worker(
            args, vae, denoiser, text, bundle if args.worker == "native" else None
        )
        return
    for case, scale in enumerate([7.5, 1.0]):
        captions = [
            "A person walks forward.",
            "A person " + "raises both arms slowly " * 30,
        ]
        lengths = [48, 40]
        texts = [""] * 2 + captions if scale > 1 else captions
        torch.manual_seed(190 + case)
        if args.worker == "upstream":
            with torch.no_grad():
                embeddings = text(texts)
            cfg = NS(model=NS(scheduler=NS(num_inference_timesteps=50, eta=0.0)))
            scheduler = DDIMScheduler(
                **{
                    key: value
                    for key, value in metadata["scheduler"].items()
                    if key != "eta"
                }
            )
            runtime = NS(
                latent_dim=[1, 256],
                vae_type="mld",
                cfg=cfg,
                guidance_scale=scale,
                do_classifier_free_guidance=scale > 1,
                scheduler=scheduler,
                denoiser=denoiser,
            )
            with torch.no_grad():
                latent = namespace["_diffusion_reverse"](runtime, embeddings, lengths)
                features = vae.decode(latent, lengths)
            mean = torch.from_numpy(np.load(args.stats / "Mean.npy")).cuda()
            std = torch.from_numpy(np.load(args.stats / "Std.npy")).cuda()
            physical = [
                (features[i, :length] * std + mean).cpu().numpy().astype(np.float32)
                for i, length in enumerate(lengths)
            ]
        else:
            from motius.motion.conditioning.motionlcm_text import encode_text

            with torch.no_grad():
                embeddings = encode_text(text, texts)
            torch.manual_seed(190 + case)
            physical = pipe.infer_t2m(
                captions, lengths, guidance_scale=scale, clamp=False
            )
        result["cases"].append(tensors(dict(text=embeddings, motion=physical)))
        print(
            args.worker,
            "official CLIP MLD full 50-step inference",
            case,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def training_worker(args, vae, denoiser, text, bundle):
    import torch
    import numpy as np
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors
    from diffusers import DDPMScheduler

    def selected(path, class_name, names, namespace):
        tree = ast.parse(path.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        methods = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name in names
        ]
        assert {node.name for node in methods} == set(names)
        exec(
            compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"),
            namespace,
        )

    if args.worker == "upstream":
        if "float" not in np.__dict__:
            np.float = float
        if "bool" not in np.__dict__:
            np.bool = bool
        from mld.models.losses.mld import MLDLosses
        from mld.data.humanml.scripts.motion_process import recover_from_ric

        scope = dict(torch=torch, np=np)
        selected(
            args.upstream / "mld/models/modeltype/mld.py",
            "MLD",
            ["train_vae_forward", "train_diffusion_forward", "_diffusion_process"],
            scope,
        )
        selected(
            args.upstream / "mld/data/HumanML3D.py",
            "HumanML3DDataModule",
            ["feats2joints"],
            dict_scope := dict(torch=torch, recover_from_ric=recover_from_ric),
        )
        geometry = NS(
            hparams=NS(
                mean=np.load(args.stats / "Mean.npy"),
                std=np.load(args.stats / "Std.npy"),
            ),
            njoints=22,
        )
        geometry_forward = lambda features: dict_scope["feats2joints"](
            geometry, features
        )
    else:
        from motius.trainers import MLDTrainer
    values = dict(training=[])
    for stage in ["vae", "diffusion"]:
        if args.worker == "upstream":
            cfg = NS(
                TRAIN=NS(
                    STAGE=stage, ABLATION=NS(VAE_TYPE="mld", PREDICT_EPSILON=True)
                ),
                LOSS=NS(
                    DIST_SYNC_ON_STEP=False,
                    LAMBDA_PRIOR=0.0,
                    LAMBDA_KL=1e-4,
                    LAMBDA_REC=1.0,
                    LAMBDA_GEN=1.0,
                    LAMBDA_LATENT=1e-5,
                    LAMBDA_JOINT=1.0,
                ),
            )
            objective = MLDLosses(True, "xyz", cfg).cuda()
            runtime = NS(
                vae=vae,
                denoiser=denoiser,
                text_encoder=text,
                vae_type="mld",
                condition="text",
                is_vae=True,
                predict_epsilon=True,
                guidance_uncodp=0.1,
                cfg=cfg,
                feats2joints=geometry_forward,
                noise_scheduler=DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                ),
            )
            runtime._diffusion_process = lambda *a, **kw: scope["_diffusion_process"](
                runtime, *a, **kw
            )
            vae.requires_grad_(stage == "vae")
            denoiser.requires_grad_(stage == "diffusion")
        else:
            trainer = MLDTrainer(bundle, stage=stage)
        vae.train()
        denoiser.train()
        text.eval()
        for case in range(3):
            torch.manual_seed(72 + case)
            features = torch.randn(2, 48, 263, device="cuda") * 0.03
            features[1, 36:] = 0
            lengths = [48, 36]
            if case == 2:
                lengths = [40, 36]
                features[:, 40:] = 0
            captions = [
                "A person walks forward.",
                "A person " + "raises both arms slowly " * 30,
            ]
            vae.zero_grad(set_to_none=True)
            denoiser.zero_grad(set_to_none=True)
            torch.manual_seed(813 + case)
            np.random.seed(813 + case)
            if args.worker == "upstream":
                fields = scope["train_" + stage + "_forward"](
                    runtime,
                    dict(
                        motion=features[:, : max(lengths)],
                        length=lengths,
                        text=captions,
                    ),
                )
                # Recent TorchMetrics wraps instance.update and discards its
                # return value. Execute the unchanged original loss method.
                loss = MLDLosses.update(objective, fields)
            else:
                trainer.train()
                loss = trainer.train_step(
                    dict(motion=features, motion_lens=lengths, text=captions)
                )["loss"]
            loss.backward()
            gradients = hashes(denoiser, gradients=True)
            # The source registers this unused positional tensor as a frozen
            # Parameter. The selected encoder graph never reads it.
            if "mem_pos.pe" in gradients:
                assert gradients.pop("mem_pos.pe") is None
            values["training"].append(
                tensors(
                    dict(
                        stage=stage,
                        loss=loss,
                        vae_gradients=hashes(vae, gradients=True),
                        denoiser_gradients=gradients,
                        cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(),
                        numpy_rng=[
                            np.random.get_state()[1].astype(np.int64),
                            np.random.get_state()[2],
                        ],
                    )
                )
            )
            print(
                args.worker,
                "original/native MLD",
                stage,
                "objective",
                case,
                "saved",
                flush=True,
            )
    torch.save(values, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "checkpoint", "text", "stats", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    parser.add_argument("--training", action="store_true")
    parser.add_argument("--reuse-results", action="store_true")
    args = parser.parse_args()
    for name in ["upstream", "checkpoint", "text", "stats", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
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
        scope="Official CLIP-conditioned MLD, author Google Drive checkpoint SHA757bea86d97f9e7ab434ebc1a074c4f9d457b0b24fb0d72d50ae3a6e08574914, original VAE/denoiser/text class and unchanged reverse-diffusion method. All used neural states and full 50-step guidance 7.5/1.0 text-to-motion versus native public Pipeline, ragged lengths and long raw captions. Native text-only CLIP excludes unused vision tower/projection/logit scale; unused encoder-path mem_pos buffer excluded. Controlled FP32/math attention. Shared supplied HumanML statistics; training and legacy SentenceT5 checkpoint are separate scopes.",
    )
    if args.training:
        report["scope"] = (
            "Author original MLD checkpoint/code, VAE feature/joint/KL objectives and latent-diffusion noise objective, all active gradient hashes and CPU/CUDA/NumPy RNG over three ragged minibatches per stage, including storage padding beyond the local valid-frame maximum; native stage switch preserves intrinsic flags. Real CLIP/VAE/denoiser; original raw corpus/optimizer/distributed protocol excluded. Controlled FP32/math attention/deterministic convolution."
        )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in [] if args.reuse_results else ["upstream", "native"]:
        cmd = [sys.executable, str(Path(__file__).resolve())]
        for name in ["upstream", "checkpoint", "text", "stats", "output_dir"]:
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            cmd + ["--worker", kind] + (["--training"] if args.training else []),
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
    print(
        "Official CLIP MLD "
        + (
            "training objectives/gradients/RNG"
            if args.training
            else "neural states and complete public inference"
        )
        + " exact"
    )


if __name__ == "__main__":
    main()
