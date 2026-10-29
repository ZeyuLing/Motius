"""Compare original MotionLCM distillation, gradients, Adam and target EMA."""

from pathlib import Path
from types import SimpleNamespace as NS
import argparse, ast, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    import numpy as np
    import torch.nn.functional as F
    from typing import Generator
    from diffusers import DDPMScheduler
    from safetensors.torch import load_file
    from verify_motionstreamer_source import hashes, digest
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    config = json.loads((args.artifact / "motionlcm_config.json").read_text())["config"]
    teacher_state = torch.load(args.teacher, map_location="cpu", weights_only=False)[
        "state_dict"
    ]
    teacher_state = {
        k[len("denoiser.") :]: v
        for k, v in teacher_state.items()
        if k.startswith("denoiser.")
    }
    teacher_config = {**config["denoiser"], "time_cond_proj_dim": None}
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from mld.models.architectures.mld_vae import MldVae
        from mld.models.architectures.mld_denoiser import MldDenoiser
        from mld.models.architectures.mld_clip import MldTextEncoder

        vae = MldVae(**config["vae"]).cuda()
        vae.load_state_dict(
            load_file(str(args.artifact / "vae.safetensors")), strict=True
        )
        text_encoder = (
            MldTextEncoder(str(args.artifact / "text_encoder/sentence-t5-large"))
            .cuda()
            .eval()
        )
        teacher_unet = MldDenoiser(**teacher_config).cuda()
        teacher_unet.load_state_dict(teacher_state, strict=True)
        unet = MldDenoiser(**config["denoiser"]).cuda()
        target_unet = MldDenoiser(**config["denoiser"]).cuda()
        for network in [unet, target_unet]:
            missing = network.load_state_dict(teacher_state, strict=False)
            assert (
                missing.missing_keys == ["time_embedding.cond_proj.weight"]
                and not missing.unexpected_keys
            )
        for network in [vae, text_encoder, teacher_unet, target_unet]:
            network.requires_grad_(False)
        source = args.upstream / "train_motionlcm.py"
        tree = ast.parse(source.read_text())
        names = {
            "guidance_scale_embedding",
            "append_dims",
            "scalings_for_boundary_conditions",
            "predicted_origin",
            "extract_into_tensor",
            "DDIMSolver",
            "update_ema",
        }
        namespace = dict(torch=torch, np=np, F=F, Generator=Generator)
        selected = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
        ]
        assert {n.name for n in selected} == names
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"),
            namespace,
        )
        main = next(
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main"
        )
        loop = next(
            n
            for n in ast.walk(main)
            if isinstance(n, ast.For)
            and isinstance(n.target, ast.Tuple)
            and [getattr(x, "id", None) for x in n.target.elts] == ["step", "batch"]
        )
        begin = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.With)
            and isinstance(n.body[0], ast.Assign)
            and ast.unparse(n.body[0].targets[0]) == "(latents, _)"
        )
        end = next(
            i
            for i, n in enumerate(loop.body)
            if isinstance(n, ast.Expr) and ast.unparse(n.value) == "loss.backward()"
        )
        # Keep every original numerical statement; only return the resulting loss.
        function = ast.FunctionDef(
            name="source_objective",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=x) for x in ["feats_ref", "lengths", "text"]],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=loop.body[begin:end]
            + [ast.Return(value=ast.Name(id="loss", ctx=ast.Load()))],
            decorator_list=[],
        )
        function = ast.fix_missing_locations(function)
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
        )
        cfg = NS(
            TRAIN=NS(
                num_ddim_timesteps=50,
                w_min=5.0,
                w_max=15.0,
                unet_time_cond_proj_dim=256,
                loss_type="huber",
                huber_c=0.001,
            )
        )
        solver = namespace["DDIMSolver"](
            noise_scheduler.alphas_cumprod.numpy(), ddim_timesteps=50
        ).to("cuda")
        namespace.update(
            vae=vae,
            text_encoder=text_encoder,
            teacher_unet=teacher_unet,
            unet=unet,
            target_unet=target_unet,
            cfg=cfg,
            noise_scheduler=noise_scheduler,
            solver=solver,
            alpha_schedule=noise_scheduler.alphas_cumprod.sqrt().cuda(),
            sigma_schedule=(1 - noise_scheduler.alphas_cumprod).sqrt().cuda(),
            uncond_prompt_embeds=text_encoder([""] * 256),
        )
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
            namespace,
        )
        objective = namespace["source_objective"]
    else:
        from motius.models.motionlcm import MotionLCMBundle
        from motius.trainers.motionlcm import MotionLCMTrainer

        bundle = MotionLCMBundle.from_pretrained(str(args.artifact), device="cuda")
        trainer = MotionLCMTrainer(bundle, teacher_weights_path=str(args.teacher))
        trainer.train()
        vae, text_encoder, teacher_unet, unet, target_unet = (
            bundle.vae,
            bundle.text_encoder,
            bundle.teacher_denoiser,
            bundle.denoiser,
            bundle.target_denoiser,
        )
    for module in [vae, teacher_unet, unet, target_unet]:
        module.train()
    result = dict(
        initial={
            name: hashes(module)
            for name, module in [
                ("vae", vae),
                ("teacher", teacher_unet),
                ("online", unet),
                ("target", target_unet),
            ]
        },
        updates=[],
    )
    optimizer = torch.optim.AdamW(
        unet.parameters(), lr=2e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0
    )

    def fingerprint(value):
        if isinstance(value, torch.Tensor):
            return digest(value)
        if isinstance(value, dict):
            return {str(k): fingerprint(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [fingerprint(v) for v in value]
        return value

    for case in range(3):
        torch.manual_seed(105 + case)
        motion = torch.randn(2, 48, 263, device="cuda") * 0.03
        motion[1, 36:] = 0
        captions = [
            "A person walks forward.",
            "A person " + "raises both arms slowly " * 30,
        ]
        torch.manual_seed(908 + case)
        if args.worker == "upstream":
            loss = objective(motion, [48, 36], captions)
        else:
            loss = trainer.train_step(
                dict(motion=motion, motion_lens=[48, 36], text=captions)
            )["loss"]
        loss.backward()
        values = dict(
            loss=loss,
            gradients=hashes(unet, gradients=True),
            cpu_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state(),
        )
        values["grad_norm"] = torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if args.worker == "upstream":
            namespace["update_ema"](target_unet.parameters(), unet.parameters(), 0.95)
        else:
            trainer.update_target()
        values.update(
            online=hashes(unet),
            target=hashes(target_unet),
            teacher=hashes(teacher_unet),
            optimizer=fingerprint(optimizer.state_dict()),
        )
        result["updates"].append(tensors(values))
        print(args.worker, "distillation/Adam/EMA update", case, "saved", flush=True)
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "teacher", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "teacher", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        scope="Original MotionLCM train_motionlcm.py unchanged numerical minibatch body, real author V1 teacher and released VAE/SentenceT5, initial teacher/online/target states, three Huber objectives/all gradients/clip norms/AdamW states/target EMA states/CPU+CUDA RNG. Controlled FP32/math attention and prepared ragged-motion fixture. Original corpus/scheduler/validation/distributed protocol excluded.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        cmd = [sys.executable, str(Path(__file__).resolve())]
        for name in ["upstream", "artifact", "teacher", "output_dir"]:
            cmd += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            cmd + ["--worker", kind],
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
    print("MotionLCM original distillation/gradient/Adam/target EMA exact", flush=True)


if __name__ == "__main__":
    main()
