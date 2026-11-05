"""Compare released DART neural weights and source sampling/primitive objectives."""

from __future__ import annotations
import argparse, ast, json, os, subprocess, sys, types
from pathlib import Path
from types import SimpleNamespace
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def source_definitions(path, namespace, names=None):
    tree = ast.parse(path.read_text())
    body = [
        n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.ClassDef))
        and (names is None or n.name in names)
    ]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def worker(args):
    import copy, enum, math, typing
    import numpy as np
    import torch

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    from motius.configuration.dart import read_arguments
    from motius.motion.representation import dart_rotation as transforms
    from motius.motion.representation.dart import get_smplx_param_from_6d

    denoiser_file = (
        args.artifact / "mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt"
    )
    denoiser_args = read_arguments(denoiser_file.parent / "args.yaml").denoiser_args
    vae_file = args.artifact / denoiser_args.mvae_path
    vae_args = read_arguments(vae_file.parent / "args.yaml")
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from model.mld_denoiser import DenoiserTransformer
        from model.mld_vae import AutoMldVae

        denoiser = DenoiserTransformer(**vars(denoiser_args.model_args))
        vae = AutoMldVae(**vars(vae_args.model_args))
        denoiser.load_state_dict(
            torch.load(denoiser_file, map_location="cpu", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
        state = torch.load(vae_file, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        vae.load_state_dict(state, strict=True)
        vae.latent_mean = state["latent_mean"]
        vae.latent_std = state["latent_std"]
        # Compile original diffusion definitions unchanged, excluding unrelated
        # dataset/body-model imports executed by its repository application.
        scope = dict(
            torch=torch,
            th=torch,
            F=torch.nn.functional,
            np=np,
            enum=enum,
            math=math,
            deepcopy=copy.deepcopy,
        )
        source_definitions(
            args.upstream / "diffusion/nn.py", scope, {"mean_flat", "sum_flat"}
        )
        source_definitions(args.upstream / "diffusion/losses.py", scope)
        source_definitions(args.upstream / "diffusion/gaussian_diffusion.py", scope)
        source_definitions(args.upstream / "diffusion/respace.py", scope)
        gd = SimpleNamespace(**scope)
        scope["gd"] = gd
        source_definitions(
            args.upstream / "mld/train_mld.py", scope, {"create_gaussian_diffusion"}
        )
        make_diffusion = scope["create_gaussian_diffusion"]
        denoiser.to(args.device).eval()
        vae.to(args.device).eval()
        decode = lambda latent, history, future: vae.decode(
            latent, history, future, scale_latent=True
        )
        encode = lambda future, history: vae.encode(future, history, scale_latent=True)
    else:
        from motius.models.dart import DARTBundle
        from motius.pipelines.dart import DARTPipeline
        from motius.diffusion.dart import create_gaussian_diffusion

        bundle = DARTBundle.from_pretrained(args.artifact, device=args.device)
        pipeline = DARTPipeline(bundle)
        denoiser, vae = bundle.denoiser_model, bundle.vae_model
        make_diffusion = create_gaussian_diffusion
        decode = lambda latent, history, future: bundle.decode_motion(
            latent, history, future, scale_latent=True
        )
        encode = lambda future, history: bundle.encode_motion(
            future, history, scale_latent=True
        )
    output = {}
    for case, (batch, respacing, guidance) in enumerate(
        [(1, "", 5.0), (2, "ddim5", 2.5), (2, "", 1.0)]
    ):
        diffusion_args = SimpleNamespace(**vars(denoiser_args.diffusion_args))
        diffusion_args.respacing = respacing
        diffusion = make_diffusion(diffusion_args)
        torch.manual_seed(43 + case)
        history = torch.randn(batch, 2, 276, device=args.device)
        text = torch.randn(batch, 512, device=args.device)
        noise = torch.randn(batch, 1, 256, device=args.device)
        y = {
            "text_embedding": text,
            "history_motion_normalized": history,
            "scale": torch.ones_like(noise) * guidance,
        }

        def guided(x, timesteps, y):
            y["uncond"] = False
            conditional = denoiser(x, timesteps, y)
            y["uncond"] = True
            unconditional = denoiser(x, timesteps, y)
            return unconditional + y["scale"] * (conditional - unconditional)

        fn = diffusion.p_sample_loop if not respacing else diffusion.ddim_sample_loop
        with torch.no_grad():
            latent = fn(
                guided,
                noise.shape,
                noise=noise,
                clip_denoised=False,
                model_kwargs={"y": y},
                device=args.device,
                progress=False,
            )
            prediction = decode(latent.permute(1, 0, 2), history, 8)
            future = torch.randn(batch, 8, 276, device=args.device)
            encoded, dist = encode(future, history)
        output[f"sampling{case}"] = dict(
            latent=latent.cpu(),
            prediction=prediction.cpu(),
            encoded=encoded.cpu(),
            mu=dist.loc.cpu(),
            sigma=dist.scale.cpu(),
        )
    # Source objectives use the exact DART276 layout and real stored stats.
    import pickle

    with open(
        args.artifact / "data/hml3d_smplh/seq_data_zero_male/mean_std_h2_f8.pkl", "rb"
    ) as stream:
        mean, std = pickle.load(stream)
    mean, std = mean.to(args.device), std.to(args.device)
    from motius.motion.representation.dart import PrimitiveUtility

    if args.body_models is not None:
        os.environ["MOTIUS_BODY_MODEL_DIR"] = str(args.body_models)
        if args.worker == "upstream":
            import config_files.data_paths as source_paths

            source_paths.body_model_dir = args.body_models
            from utils.smpl_utils import PrimitiveUtility, get_smplx_param_from_6d
            from pytorch3d import transforms
        representation = PrimitiveUtility(device=args.device, body_type="smplh")
    else:
        representation = object.__new__(PrimitiveUtility)
        representation.motion_repr = dict(
            transl=3,
            poses_6d=132,
            transl_delta=3,
            global_orient_delta_6d=6,
            joints=66,
            joints_delta=66,
        )
    processor = SimpleNamespace(
        history_length=2,
        future_length=8,
        num_primitive=1,
        primitive_utility=representation,
        denormalize=lambda x: x * std + mean,
        tensor_to_dict=representation.tensor_to_dict,
    )
    for stage in ["denoiser", "vae"]:
        denoiser.zero_grad(set_to_none=True)
        vae.zero_grad(set_to_none=True)
        denoiser.requires_grad_(stage == "denoiser").train(stage == "denoiser")
        vae.requires_grad_(stage == "vae").train(stage == "vae")
        torch.manual_seed(313)
        motion = torch.randn(2, 10, 276, device=args.device)
        text = torch.randn(2, 512, device=args.device)
        condition = {
            "gender": (
                ["female", "male"] if args.body_models is not None else ["male", "male"]
            ),
            "betas": torch.zeros(2, 10, 10, device=args.device),
        }
        weights = dict(
            weight_smpl_joints_rec=10.0 if args.body_models is not None else 0.0,
            weight_joints_consistency=10.0 if args.body_models is not None else 0.0,
        )
        network = denoiser if stage == "denoiser" else vae
        if args.optimizer_check:
            source_train_args = read_arguments(
                (denoiser_file if stage == "denoiser" else vae_file).parent
                / "args.yaml"
            ).train_args
            optimizer = torch.optim.AdamW(
                network.parameters(), lr=source_train_args.learning_rate
            )
            average = copy.deepcopy(network).requires_grad_(False).eval()
        np.random.seed(19)
        if args.worker == "native":
            from motius.trainers.dart import DARTTrainer

            trainer = DARTTrainer(bundle, processor, stage=stage, loss_weights=weights)
            if args.optimizer_check:
                from accelerate import Accelerator

                trainer.accelerator = Accelerator(mixed_precision="no")
                trainer.set_optimizers(
                    {"default": trainer.accelerator.prepare(optimizer)}
                )
                trainer.runner = SimpleNamespace(global_step=17)
            torch.manual_seed(117)
            terms = trainer.train_step(
                dict(motion=motion, text_embedding=text, condition=condition)
            )
        else:
            defaults = dict(
                weight_latent_rec=1.0,
                weight_feature_rec=1.0,
                weight_joints_delta=10000.0,
                weight_transl_delta=10000.0,
                weight_orient_delta=10000.0,
                weight_kl=1e-6,
                weight_rec=1.0,
                **weights,
            )
            if stage == "vae":
                defaults.update(
                    weight_joints_delta=100.0,
                    weight_transl_delta=100.0,
                    weight_orient_delta=100.0,
                )
            scope = dict(
                torch=torch,
                transforms=transforms,
                get_smplx_param_from_6d=get_smplx_param_from_6d,
            )
            tree = ast.parse(
                (
                    args.upstream
                    / f'mld/train_{"mld" if stage=="denoiser" else "mvae"}.py'
                ).read_text()
            )
            cls = next(
                n
                for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "Trainer"
            )
            method = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "calc_loss"
            )
            if stage == "vae" and args.body_models is None:
                # Body consistency contributes zero in this focused comparison;
                # skip its asset-only evaluation while preserving all active terms.
                begin = next(
                    i
                    for i, n in enumerate(method.body)
                    if isinstance(n, ast.Assign)
                    and ast.unparse(n.targets[0]) == "gt_motion_tensor"
                )
                end = next(
                    i
                    for i, n in enumerate(method.body)
                    if isinstance(n, ast.Expr)
                    and isinstance(n.value, ast.Constant)
                    and n.value.value == "temporal delta loss"
                )
                method.body[begin:end] = ast.parse(
                    "dataset=self.train_dataset\nterms['smpl_joints_rec']=torch.zeros(1,device=self.device)\nterms['joints_consistency']=torch.zeros(1,device=self.device)"
                ).body
            exec(
                compile(
                    ast.fix_missing_locations(
                        ast.Module(body=[method], type_ignores=[])
                    ),
                    str(args.upstream),
                    "exec",
                ),
                scope,
            )
            owner = SimpleNamespace(
                args=SimpleNamespace(train_args=SimpleNamespace(**defaults)),
                train_dataset=processor,
                device=torch.device(args.device),
                rec_criterion=torch.nn.HuberLoss(delta=1.0, reduction="mean"),
            )
            torch.manual_seed(117)
            history = motion[:, :2]
            future = motion[:, -8:]
            if stage == "denoiser":
                with torch.no_grad():
                    latent, _ = vae.encode(future, history, scale_latent=True)
                start = latent.permute(1, 0, 2)
                # The native trainer uses this same original categorical draw.
                np.random.seed(19)
                timestep = (
                    torch.from_numpy(np.random.choice(10, size=2, p=np.ones(10) / 10))
                    .long()
                    .to(args.device)
                )
                noisy = diffusion.q_sample(start, timestep, torch.randn_like(start))
                prediction = denoiser(
                    noisy,
                    diffusion._scale_timesteps(timestep),
                    {"text_embedding": text, "history_motion_normalized": history},
                ).permute(1, 0, 2)
                decoded = vae.decode(prediction, history, 8, scale_latent=True)
                terms = scope["calc_loss"](
                    owner,
                    motion,
                    {"y": condition},
                    history,
                    future,
                    decoded,
                    latent,
                    prediction,
                    None,
                )
            else:
                latent, dist = vae.encode(future, history)
                decoded = vae.decode(latent, history, 8)
                terms = scope["calc_loss"](
                    owner,
                    motion,
                    {"y": condition},
                    history,
                    future,
                    decoded,
                    latent,
                    dist,
                )
        if args.optimizer_check and args.worker == "upstream":
            # Execute the author's LR, backward, clipping, AdamW and EMA
            # statements unchanged; only provide the original local bindings.
            train = next(
                n
                for n in cls.body
                if isinstance(n, ast.FunctionDef) and n.name == "train"
            )
            outer = next(n for n in ast.walk(train) if isinstance(n, ast.While))
            lr_statement = next(
                n
                for n in outer.body
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "train_args.anneal_lr"
            )
            primitive_loop = next(
                n
                for n in outer.body
                if isinstance(n, ast.For) and ast.unparse(n.target) == "primitive_idx"
            )
            begin = next(
                i
                for i, n in enumerate(primitive_loop.body)
                if ast.unparse(n) == "optimizer.zero_grad()"
            )
            end = next(
                i
                for i, n in enumerate(primitive_loop.body[begin:], begin)
                if isinstance(n, ast.If)
                and ast.unparse(n.test) == "train_args.ema_decay > 0"
            )
            owner.step = 18
            owner.denoiser_model = owner.model = network
            owner.denoiser_model_avg = owner.model_avg = average
            bindings = dict(
                torch=torch,
                nn=torch.nn,
                self=owner,
                optimizer=optimizer,
                train_args=source_train_args,
                total_steps=sum(
                    getattr(source_train_args, f"stage{i}_steps") for i in [1, 2, 3]
                ),
                model=network,
                denoiser_model=network,
                loss=terms["loss"],
            )
            exec(
                compile(
                    ast.Module(
                        body=[lr_statement, *primitive_loop.body[begin : end + 1]],
                        type_ignores=[],
                    ),
                    str(args.upstream),
                    "exec",
                ),
                bindings,
            )
        elif not args.optimizer_check:
            terms["loss"].backward()
        output[stage] = dict(
            losses={
                k: v.detach().cpu()
                for k, v in terms.items()
                if not (args.optimizer_check and k == "loss")
            },
            gradients={
                k: p.grad.cpu()
                for k, p in network.named_parameters()
                if p.grad is not None
            },
        )
        if args.optimizer_check:
            from verify_motionstreamer_source import hashes

            if args.worker == "native":
                average = getattr(bundle, trainer.ema_name)
            output[stage].update(
                online={
                    k: torch.tensor(list(bytes.fromhex(v)), dtype=torch.uint8)
                    for k, v in hashes(network).items()
                },
                average={
                    k: torch.tensor(list(bytes.fromhex(v)), dtype=torch.uint8)
                    for k, v in hashes(average).items()
                },
                learning_rate=torch.tensor(
                    optimizer.param_groups[0]["lr"], dtype=torch.float64
                ),
                adam={
                    str(k): {name: value.cpu() for name, value in state.items()}
                    for k, state in optimizer.state_dict()["state"].items()
                },
            )
            if stage == "vae":
                torch.manual_seed(839)
                if args.worker == "native":
                    trainer.calibrate_latent(dict(motion=motion))
                else:
                    calibration = next(
                        n
                        for n in cls.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "get_latent_scale"
                    )
                    # A prepared primitive is the public data boundary; source
                    # calibration still executes its complete unchanged body.
                    owner.batch_size = len(motion)
                    prepared = dict(
                        motion_tensor_normalized=motion.permute(0, 2, 1).unsqueeze(2),
                        texts=["", ""],
                        text_embedding=text,
                        gender=condition["gender"],
                        betas=condition["betas"],
                        history_motion=motion[:, :2].permute(0, 2, 1).unsqueeze(2),
                        history_mask=torch.ones_like(
                            motion.permute(0, 2, 1).unsqueeze(2), dtype=torch.bool
                        ),
                        history_length=2,
                        future_length=8,
                    )
                    owner.train_dataset.get_batch = lambda size: [prepared]
                    calibration_scope = dict(torch=torch)
                    get_batch = next(
                        n
                        for n in cls.body
                        if isinstance(n, ast.FunctionDef)
                        and n.name == "get_primitive_batch"
                    )
                    exec(
                        compile(
                            ast.Module(body=[calibration, get_batch], type_ignores=[]),
                            str(args.upstream),
                            "exec",
                        ),
                        calibration_scope,
                    )
                    owner.get_primitive_batch = types.MethodType(
                        calibration_scope["get_primitive_batch"], owner
                    )
                    calibration_scope["get_latent_scale"](owner, average)
                output[stage]["calibration"] = dict(
                    mean=average.latent_mean.cpu(), std=average.latent_std.cpu()
                )
    torch.save(output, args.output_dir / f"{args.worker}.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--body-models", type=Path)
    parser.add_argument("--optimizer-check", action="store_true")
    parser.add_argument("--worker", choices=["upstream", "native"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for key in ["upstream", "artifact", "output_dir"]:
        setattr(args, key, getattr(args, key).resolve())
    if args.body_models is not None:
        args.body_models = args.body_models.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    root = Path(__file__).resolve().parents[1]
    path = args.output_dir / "source_parity.json"
    report = dict(
        complete=False,
        comparison=(
            "Original DART primitive objectives, AdamW/clipping/LR/EMA update at completed step17, every updated neural hash/Adam state, and original VAE latent calibration on an explicit prepared batch"
            if args.optimizer_check
            else "Original DART released neural weights, complete primitive DDPM/DDIM and active denoiser/VAE objectives"
        ),
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        reference_extraction=(
            "Original neural modules, diffusion, complete unchanged loss bodies, actual original SMPL-H geometry/PyTorch3D, both genders. Both body objectives active at weight10 (published VAE defaults; additional denoiser diagnostic coverage)."
            if args.body_models is not None
            else "Original neural modules and unchanged diffusion definitions. VAE body losses disabled in this focused test; full SMPL-H rollout is a separate acceptance gate."
        ),
        artifact_files={
            str(p.relative_to(args.artifact)): digest(p)
            for p in args.artifact.rglob("*")
            if p.is_file()
            and ".cache" not in p.parts
            and p.suffix in [".pt", ".pkl", ".yaml"]
        },
        tests=[],
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        subprocess.run(
            [sys.executable, __file__, *sys.argv[1:], "--worker", kind],
            cwd=root,
            env={
                **os.environ,
                "PYTHONPATH": str(root),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            check=True,
        )
    import torch

    try:
        compare(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    folders = [
        "motius/models/dart",
        "motius/pipelines/dart",
        "motius/trainers/dart",
        "motius/diffusion/dart",
        "motius/configuration/dart.py",
        "motius/motion/conditioning/dart.py",
        "motius/motion/conditioning/dart_seed.py",
        "motius/motion/representation/dart.py",
        "motius/motion/representation/dart_rotation.py",
    ]
    report.update(
        complete=True,
        torch_version=torch.__version__,
        native_files={
            str(p.relative_to(root)): digest(p)
            for folder in folders
            for p in (
                list((root / folder).rglob("*.py"))
                if (root / folder).is_dir()
                else [root / folder]
            )
        },
    )
    path.write_text(json.dumps(report, indent=2) + "\n")
    print("All DART primitive comparisons exact")


if __name__ == "__main__":
    main()
