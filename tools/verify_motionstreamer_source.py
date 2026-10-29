"""Compare published MotionStreamer neural execution, generation and training objectives."""

from pathlib import Path
import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def hashes(module, *, gradients=False):
    import torch

    source = module.named_parameters() if gradients else module.state_dict().items()
    return (
        {
            name: None if value.grad is None else digest(value.grad)
            for name, value in source
        }
        if gradients
        else {name: digest(value) for name, value in source}
    )


def digest(tensor):
    import torch

    return hashlib.sha256(
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    ).hexdigest()


class PreparedText:
    """Deterministic text-feature input fixture; no text network is substituted."""

    def encode(self, text):
        import numpy as np

        captions = text if isinstance(text, list) else [text]
        result = np.stack(
            [
                np.sin(
                    np.arange(768, dtype=np.float32) * 0.013
                    + (0 if not caption else 0.17)
                )
                for caption in captions
            ]
        )
        return result if isinstance(text, list) else result[0]


def worker(args):
    import torch
    from safetensors.torch import load_file
    from verify_prompthmr_video_parity import tensors

    assert torch.cuda.is_available()
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from models.llama_model import LLaMAHF, LLaMAHFConfig
        from models.tae import Causal_HumanTAE

        ar = LLaMAHF(
            LLaMAHFConfig.from_name("Normal_size"), device=torch.device("cuda")
        )
        tae = Causal_HumanTAE(
            hidden_size=1024,
            down_t=2,
            stride_t=2,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            latent_dim=16,
            clip_range=[-30, 20],
        ).cuda()
        ar_state = load_file(str(args.artifact / "ar.safetensors"))
        tae_state = load_file(str(args.artifact / "tae.safetensors"))
        if args.original_weights is not None:
            ar_state = torch.load(
                args.original_weights / "Experiments/t2m_model/latest.pth",
                map_location="cpu",
                weights_only=False,
            )["trans"]
            tae_state = torch.load(
                args.original_weights / "Causal_TAE/net_last.pth",
                map_location="cpu",
                weights_only=False,
            )["net"]
        ar_state = {
            name.removeprefix("module."): value for name, value in ar_state.items()
        }
        tae_state = {
            name.removeprefix("module."): value for name, value in tae_state.items()
        }
        ar.load_state_dict(ar_state, strict=True)
        tae.load_state_dict(tae_state, strict=True)
        sample = lambda text, **kw: ar.sample_for_eval_CFG(
            text, tokenize_model=PreparedText(), device=torch.device("cuda"), **kw
        )
    else:
        from motius.models.motionstreamer import MotionStreamerBundle
        from motius.pipelines.motionstreamer.sampling import sample_for_eval_CFG

        bundle = MotionStreamerBundle.from_pretrained(
            args.artifact, load_text_model=False, device="cuda"
        )
        ar, tae = bundle.ar, bundle.tae
        sample = lambda text, **kw: sample_for_eval_CFG(
            ar, text, tokenize_model=PreparedText(), device=torch.device("cuda"), **kw
        )
    ar.cuda().eval()
    tae.cuda().eval()
    result = dict(weights=dict(ar=hashes(ar), tae=hashes(tae)), generation=[])
    for guidance in [1.0, 4.0]:
        torch.manual_seed(81)
        torch.cuda.manual_seed_all(81)
        with torch.no_grad():
            latent = sample(["walk forward"], length=12, cfg=guidance)
            motion = tae.forward_decoder(latent)
        result["generation"].append(
            tensors(dict(latent=latent, motion=motion, rng=torch.cuda.get_rng_state()))
        )
    latents = torch.sin(
        torch.arange(2 * 4 * 16, device="cuda").reshape(2, 4, 16) * 0.037
    )
    features = torch.cos(torch.arange(2 * 768, device="cuda").reshape(2, 768) * 0.013)
    lengths = torch.tensor([4, 3], device="cuda")
    result["training"] = {}
    for stage in ["ar", "streaming"]:
        ar.train().zero_grad(set_to_none=True)
        torch.manual_seed(417)
        torch.cuda.manual_seed_all(417)
        if args.worker == "upstream":
            file = args.upstream / (
                "train_t2m.py" if stage == "ar" else "train_motionstreamer.py"
            )
            function = "forward_loss_withmask_2_forward" + (
                "_streaming" if stage == "streaming" else ""
            )
            tree = ast.parse(file.read_text())
            names = {"lengths_to_mask", "cosine_decay", "replace_with_pred", function}
            selected = ast.Module(
                body=[
                    node
                    for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name in names
                ],
                type_ignores=[],
            )
            context = dict(
                torch=torch, comp_device=torch.device("cuda"), diffmlps_batch_mul=4
            )
            exec(compile(selected, str(file), "exec"), context)
            kw = (
                dict(A_token_length=torch.tensor([1, 0], device="cuda"))
                if stage == "streaming"
                else {}
            )
            loss = context[function](latents, ar, lengths, features, 3, 17, **kw)
        else:
            from motius.trainers.motionstreamer import MotionStreamerTrainer

            trainer = MotionStreamerTrainer(bundle, stage=stage, total_steps=17)
            loss = trainer.train_step(
                dict(
                    latents=latents,
                    text_features=features,
                    lengths=lengths,
                    prefix_lengths=torch.tensor([1, 0], device="cuda"),
                    step=3,
                )
            )["loss"]
        loss.backward()
        result["training"][stage] = tensors(
            dict(
                loss=loss,
                gradients=hashes(ar, gradients=True),
                cuda_rng=torch.cuda.get_rng_state(),
                cpu_rng=torch.get_rng_state(),
            )
        )
    tae.train().zero_grad(set_to_none=True)
    torch.manual_seed(118)
    torch.cuda.manual_seed_all(118)
    target = torch.sin(
        torch.arange(2 * 16 * 272, device="cuda").reshape(2, 16, 272) * 0.0037
    )
    if args.worker == "upstream":
        from utils.losses import ReConsLoss

        criterion = ReConsLoss(motion_dim=272)
        prediction, mean, log_variance = tae(target)
        reconstruction = criterion(prediction, target)
        divergence = criterion.forward_KL(mean, log_variance)
        root = criterion.forward_root(prediction, target)
        loss = reconstruction + divergence + 7.0 * root
    else:
        from motius.trainers.motionstreamer import MotionStreamerTrainer

        trainer = MotionStreamerTrainer(bundle, stage="tae")
        loss = trainer.train_step(dict(motion=target))["loss"]
    loss.backward()
    result["training"]["tae"] = tensors(
        dict(
            loss=loss,
            gradients=hashes(tae, gradients=True),
            cuda_rng=torch.cuda.get_rng_state(),
        )
    )
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--original-weights", type=Path)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    if args.original_weights is not None:
        args.original_weights = args.original_weights.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    report = dict(
        complete=False,
        scope="Pristine original MotionStreamer AR/TAE classes and two-forward AR/streaming and TAE reconstruction objectives versus native neural modules, Pipeline sampling and Trainer. Original released AR/TAE checkpoints when --original-weights is provided, otherwise published Motius weights; both actual architectures loaded strictly; prepared T5 features. Three tokens with all original 50 diffusion steps, CFG=1/4, decoded motion, complete neural-state/gradient SHA256 and RNG. Raw text encoder, source continuation bug and full dataset training excluded.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        tests=[],
    )
    if args.original_weights is not None:
        report["original_checkpoint_provenance"] = json.loads(
            (args.original_weights / "provenance.json").read_text()
        )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
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
                *(
                    ["--original-weights", str(args.original_weights)]
                    if args.original_weights is not None
                    else []
                ),
            ],
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            },
            check=True,
        )
    import torch
    from verify_prompthmr_video_parity import compare_full

    try:
        compare_full(
            torch.load(args.output_dir / "upstream.pt", weights_only=True),
            torch.load(args.output_dir / "native.pt", weights_only=True),
            "",
            report["tests"],
        )
        report["complete"] = True
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        "Original/native MotionStreamer inference, objectives and gradient hashes exact"
    )


if __name__ == "__main__":
    main()
