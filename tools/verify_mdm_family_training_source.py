"""Released MDM-family objectives, all gradients and RNG against original code."""

from pathlib import Path
from types import SimpleNamespace as NS
import argparse
import ast
import importlib
import json
import os
import shutil
import subprocess
import sys
import __future__

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        namespace,
    )


def worker(args):
    import torch
    import numpy as np
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
    config = json.loads((args.artifact / (args.method + "_config.json")).read_text())[
        "config"
    ]
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        # Restore NumPy's historical alias used by the pinned original code.
        # np.float was exactly the builtin float, not a different numeric type.
        if "float" not in np.__dict__:
            np.float = float
        if "bool" not in np.__dict__:
            np.bool = bool
        import clip

        clip.load = lambda *a, **kw: (
            clip.model.build_model(
                dict(load_file(str(args.artifact / "clip.safetensors")))
            ).float(),
            None,
        )
        module = importlib.import_module(
            {
                "mdm": "model.mdm",
                "condmdi": "model.mdm_unet",
                "omnicontrol": "model.cmdm",
            }[args.method]
        )

        class UnusedVisualizer:
            """Unentered non-neural SMPL visualization dependency; fail on use."""

            def __init__(self, *a, **kw):
                self.smpl_model = self

            def _apply(self, function):
                return self

            def train(self, *a, **kw):
                return self

            def __call__(self, *a, **kw):
                raise AssertionError("Excluded body visualizer entered the objective")

        module.Rotation2xyz = UnusedVisualizer
        from diffusion import gaussian_diffusion as gd
        from diffusion.respace import SpacedDiffusion, space_timesteps

        scope = dict(
            gd=gd, SpacedDiffusion=SpacedDiffusion, space_timesteps=space_timesteps
        )
        if args.method == "mdm":
            from data_loaders.humanml_utils import HML_EE_JOINT_NAMES

            scope["HML_EE_JOINT_NAMES"] = HML_EE_JOINT_NAMES
            source_functions(
                args.upstream / "utils/parser_util.py", ["get_cond_mode"], scope
            )
        elif args.method == "condmdi":
            from diffusion.respace import DiffusionConfig

            scope["DiffusionConfig"] = DiffusionConfig
        source_functions(
            args.upstream / "utils/model_util.py",
            ["get_model_args", "create_gaussian_diffusion"],
            scope,
        )
        values = dict(config)
        for name, value in dict(
            unconstrained=False, drop_redundant=False, traj_only=False, use_ddim=False
        ).items():
            values.setdefault(name, value)
        source_args = NS(**values)
        options = scope["get_model_args"](source_args, NS(dataset=NS()))
        cls = getattr(
            module,
            {"mdm": "MDM", "condmdi": "MDM_UNET", "omnicontrol": "CMDM"}[args.method],
        )
        torch.manual_seed(81)
        model = cls(**options)
        state = load_file(str(args.artifact / "model.safetensors"))
        missing, unexpected = model.load_state_dict(state, strict=False)
        assert not unexpected, unexpected
        assert all(
            name.startswith("clip_model.") or "sequence_pos_encoder.pe" in name
            for name in missing
        ), missing
        model.cuda()
        if args.method == "omnicontrol":
            # The unchanged original runtime reads these corpus statistics by
            # relative path. Preserve that data layout in an isolated directory.
            root = args.output_dir / "source-data"
            for subdir, names in [
                ("HumanML3D", ["Mean.npy", "Std.npy"]),
                ("humanml_spatial_norm", ["Mean_raw.npy", "Std_raw.npy"]),
            ]:
                destination = root / "dataset" / subdir
                destination.mkdir(parents=True, exist_ok=True)
                for name in names:
                    origin = args.upstream / "dataset" / subdir / name
                    if origin.is_file():
                        np.testing.assert_array_equal(
                            np.load(origin), np.load(args.artifact / name)
                        )
                    shutil.copy2(
                        origin if origin.is_file() else args.artifact / name,
                        destination / name,
                    )
            os.chdir(root)
        diffusion = scope["create_gaussian_diffusion"](source_args)
        from diffusion.resample import UniformSampler

        sampler = UniformSampler(diffusion)
    else:
        from motius import Pipeline
        from motius.trainers import MDMTrainer, CondMDITrainer, OmniControlTrainer

        pipe = Pipeline.from_pretrained(
            str(args.artifact), bundle_kwargs={"device": "cuda"}
        )
        bundle = pipe.bundle
        model = bundle.net
        trainer = {
            "mdm": MDMTrainer,
            "condmdi": CondMDITrainer,
            "omnicontrol": OmniControlTrainer,
        }[args.method](bundle)

    assert next(
        model.parameters()
    ).is_cuda, "Neural comparison must run on the allocated GPU"
    if args.inference:
        inference_worker(
            args,
            model,
            diffusion if args.worker == "upstream" else pipe.diffusion,
            config,
            pipe if args.worker == "native" else None,
        )
        return
    probes = []

    def before_forward(module, inputs):
        probes.append(
            tensors(
                dict(
                    noisy=inputs[0], timesteps=inputs[1], rng=torch.cuda.get_rng_state()
                )
            )
        )

    model.register_forward_pre_hook(before_forward)
    results = dict(neural=hashes(model), training=[])
    for iteration in range(2):
        torch.manual_seed(120 + iteration)
        motion = torch.randn(2, 263, 1, 48, device="cuda") * 0.03
        lengths = torch.tensor([48, 36], device="cuda")
        mask = (torch.arange(48, device="cuda")[None] < lengths[:, None])[:, None, None]
        captions = (
            ["A person walks forward.", ""]
            if iteration == 0
            else [
                "A person " + "walks forward slowly " * 30,
                "A person raises both arms.",
            ]
        )
        extra = {}
        if args.method == "condmdi":
            observed = torch.zeros_like(motion, dtype=torch.bool)
            observed[:, :, :, ::8] = True
            observed &= mask
            extra.update(obs_x0=motion, obs_mask=observed)
        if args.method == "omnicontrol":
            hint = torch.zeros(2, 48, 66, device="cuda")
            hint[:, ::8, :3] = 0.1
            hint[1, 36:] = 0
            extra["hint"] = hint
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(921 + iteration)
        np.random.seed(921 + iteration)
        if args.worker == "upstream":
            timesteps, weights = sampler.sample(2, motion.device)
            conditions = dict(text=captions, lengths=lengths, mask=mask)
            kwargs = {"y": conditions}
            if args.method == "condmdi":
                kwargs.update(extra)
            elif args.method == "omnicontrol":
                conditions.update(extra)
            terms = diffusion.training_losses(
                model, motion, timesteps, model_kwargs=kwargs
            )
            loss = (terms["loss"] * weights).mean()
            metrics = {name: value.detach().mean() for name, value in terms.items()}
        else:
            output = trainer.train_step(
                dict(motion=motion, motion_lens=lengths, text=captions, **extra)
            )
            loss = output["loss"]
            metrics = {
                name.removeprefix(args.method + "_"): value
                for name, value in output.items()
                if name != "loss"
            }
        loss.backward()
        results["training"].append(
            tensors(
                dict(
                    loss=loss,
                    terms=metrics,
                    gradients=hashes(model, gradients=True),
                    cuda_rng=torch.cuda.get_rng_state(),
                    numpy_rng=[
                        np.random.get_state()[1].astype(np.int64),
                        np.random.get_state()[2],
                    ],
                )
            )
        )
    torch.save(probes, args.output_dir / (args.worker + "_probes.pt"))
    torch.save(results, args.output_dir / (args.worker + ".pt"))
    print(
        args.worker,
        args.method,
        "original/prepared native objective and all gradients saved",
        flush=True,
    )


def inference_worker(args, model, diffusion, config, pipe):
    import torch
    import numpy as np
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    model.eval()
    assert diffusion.num_timesteps == 1000
    original_step = diffusion.p_sample
    steps = []

    def record_step(*a, **kw):
        result = original_step(*a, **kw)
        t = a[2] if len(a) > 2 else kw["t"]
        if int(t[0]) in {999, 499, 9, 1, 0}:
            steps.append(tensors(dict(t=t, **result)))
        return result

    diffusion.p_sample = record_step
    result = dict(neural=hashes(model), inference=[])
    captions = [
        "A person walks forward.",
        "A person " + "raises both arms slowly " * 20,
    ]
    lengths = [196, 160] if args.method == "omnicontrol" else [48, 40]
    frames = max(lengths)
    valid = (
        torch.arange(frames, device="cuda")[None]
        < torch.tensor(lengths, device="cuda")[:, None]
    )
    physical = (
        torch.sin(
            torch.arange(2 * frames * 263, device="cuda").reshape(2, frames, 263)
            * 0.017
        )
        * 0.03
    )
    physical[..., 3] = 0.9
    physical[1, lengths[1] :] = 0
    arrays = [physical[i, :length].cpu().numpy() for i, length in enumerate(lengths)]
    if args.worker == "upstream":
        from model.cfg_sampler import ClassifierFreeSampleModel
        from data_loaders.humanml.scripts.motion_process import recover_from_ric

        if args.method == "condmdi":
            from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos

            _, root, angle = recover_root_rot_pos(physical, return_rot_ang=True)
            absolute = physical.clone()
            absolute[..., 0] = angle
            absolute[..., 1:3] = root[..., [0, 2]]
            mean = torch.from_numpy(np.load(args.artifact / "Mean_abs_3d.npy")).cuda()
            std = torch.from_numpy(np.load(args.artifact / "Std_abs_3d.npy")).cuda()
        else:
            mean = torch.from_numpy(np.load(args.artifact / "Mean.npy")).cuda()
            std = torch.from_numpy(np.load(args.artifact / "Std.npy")).cuda()
    for case in range(2):
        scale = 2.5 if case == 0 else 1.0
        seed = 512 + case
        steps.clear()
        if args.worker == "upstream":
            y = dict(
                text=captions,
                lengths=torch.tensor(lengths, device="cuda"),
                mask=valid[:, None, None],
            )
            kwargs = dict(y=y)
            if args.method == "mdm":
                y["scale"] = torch.full((2,), scale, device="cuda")
                with torch.no_grad():
                    y["text_embed"] = model.encode_text(captions)
            elif args.method == "condmdi":
                observed = torch.zeros(2, 263, 1, 48, dtype=torch.bool, device="cuda")
                if case:
                    observed[0, :, :, 0] = True
                    observed[0, :, :, 47] = True
                    observed[1, :, :, 0] = True
                    observed[1, :, :, 39] = True
                normalized = (
                    ((absolute - mean) / std).permute(0, 2, 1).unsqueeze(2).contiguous()
                )
                kwargs.update(obs_x0=normalized, obs_mask=observed)
                y.update(
                    diffusion_steps=1000,
                    text_scale=torch.full((2,), scale, device="cuda"),
                    reconstruction_guidance=False,
                )
                if case:
                    y.update(
                        imputate=True,
                        stop_imputation_at=0,
                        replacement_distribution="conditional",
                        inpainted_motion=normalized,
                        inpainting_mask=observed,
                    )
            else:
                hint = torch.zeros(2, frames, 66, device="cuda")
                if case:
                    positions = recover_from_ric(physical, 22)
                    raw_mean = (
                        torch.from_numpy(np.load(args.artifact / "Mean_raw.npy"))
                        .cuda()
                        .reshape(22, 3)
                    )
                    raw_std = (
                        torch.from_numpy(np.load(args.artifact / "Std_raw.npy"))
                        .cuda()
                        .reshape(22, 3)
                    )
                    normalized = (positions - raw_mean) / raw_std
                    for i, length in enumerate(lengths):
                        for frame in [0, length - 1]:
                            hint[i, frame, :3] = normalized[i, frame, 0]
                y.update(scale=torch.full((2,), scale, device="cuda"))
                if case:
                    y["hint"] = hint
            wrapped = ClassifierFreeSampleModel(model) if scale != 1 else model
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(seed)
                sample = diffusion.p_sample_loop(
                    wrapped,
                    (2, 263, 1, frames),
                    clip_denoised=False,
                    model_kwargs=kwargs,
                    progress=False,
                )
            output = (
                (sample[:, :, 0].permute(0, 2, 1).contiguous() * std + mean)
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            output = [output[i, :length] for i, length in enumerate(lengths)]
        else:
            options = dict(guidance_param=scale, progress=False)
            if args.method == "mdm":
                with torch.random.fork_rng(devices=[0]):
                    torch.manual_seed(seed)
                    output = pipe.infer_t2m(captions, lengths, **options)
            elif args.method == "condmdi":
                options.update(
                    seed=seed, return_absolute=True, return_joints=args.final_joints
                )
                output = (
                    pipe.infer_control(
                        captions, arrays, lengths, control_mode="first_last", **options
                    )
                    if case
                    else pipe.infer_t2m(captions, lengths, **options)
                )
            else:
                options.update(seed=seed)
                output = (
                    pipe.infer_control(
                        captions,
                        arrays,
                        lengths,
                        control_mode="first_last",
                        joint_indices=[0],
                        **options,
                    )
                    if case
                    else pipe.infer_t2m(captions, lengths, **options)
                )
        if args.final_joints:
            if args.worker == "upstream":
                output = [
                    recover_from_ric(
                        torch.from_numpy(value),
                        22,
                        **({"abs_3d": True} if args.method == "condmdi" else {}),
                    )
                    for value in output
                ]
            elif args.method != "condmdi":
                from motius.motion.representation.humanml import (
                    recover_from_ric as native_recover,
                )

                output = [
                    native_recover(torch.from_numpy(value), 22) for value in output
                ]
        result["inference"].append(tensors(dict(output=output, steps=list(steps))))
        print(
            args.worker,
            args.method,
            "full 1000-step inference case",
            case,
            "saved",
            flush=True,
        )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method", choices=["mdm", "condmdi", "omnicontrol"], required=True
    )
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    parser.add_argument("--reuse-results", action="store_true")
    parser.add_argument("--reuse-source", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--final-joints", action="store_true")
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        tests=[],
        method=args.method,
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        scope="Published HumanML3D neural weights and original repository network, "
        "diffusion factories, uniform NumPy timestep sampling, denoising objectives, "
        "all gradient hashes and RNG versus native prepared-motion Trainer. "
        "Two ragged-length minibatches, raw short/long/empty captions, actual "
        "keyframe observations or spatial guidance where applicable. Controlled "
        "FP32/math-attention/deterministic-convolution runtime. Unused SMPL "
        "visualizer constructor excluded and guarded against entry. Geometry "
        "losses, raw dataset/mask-selection, optimizer/AMP/distributed/full "
        "inference protocols excluded.",
    )
    if args.inference:
        report["scope"] = (
            "Published HumanML3D weights and original repository network, CFG, full released 1000-step ancestral diffusion, raw CLIP captions, ragged lengths, text-only and method-native control conditions versus the public native inference Pipeline. Samples and selected denoising steps compared bitwise in separate processes. CondMDI uses its original absolute-root output representation. Original raw corpus ingestion, SMPL visualization, training and modified-current export are outside this inference certificate."
        )
    if args.final_joints:
        assert args.inference
        report[
            "scope"
        ] += " Final original CPU world-joint recovery is included; CondMDI exercises public return_joints=True without relative-root conversion."
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if args.reuse_source:
        assert (args.output_dir / "upstream.pt").is_file()
    for kind in (
        []
        if args.reuse_results
        else (["native"] if args.reuse_source else ["upstream", "native"])
    ):
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--method",
                args.method,
                "--upstream",
                str(args.upstream),
                "--artifact",
                str(args.artifact),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
                *(["--inference"] if args.inference else []),
                *(["--final-joints"] if args.final_joints else []),
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
    original = torch.load(args.output_dir / "upstream.pt", weights_only=True)
    report["neural_state_hashes_compared"] = len(original["neural"])
    if "training" in original:
        report["gradient_hashes_compared"] = [
            sum(value is not None for value in item["gradients"].values())
            for item in original["training"]
        ]
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        args.method,
        (
            "full original-source inference exact"
            if args.inference
            else "original denoising objectives, all gradients and RNG exact"
        ),
    )


if __name__ == "__main__":
    main()
