"""Offline current-weight exports through public native motion pipelines."""

from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    path = args.artifact if args.worker == "export" else args.output_dir / "exported"
    from motius import Pipeline as AutoPipeline

    if args.method == "motionclr" and args.worker == "export":
        from motius.pipelines.motionclr import MotionCLRPipeline

        pipe = MotionCLRPipeline.from_pretrained(
            str(path), bundle_kwargs={"device": "cuda", "torch_dtype": "float32"}
        )
    else:
        pipe = AutoPipeline.from_pretrained(str(path), bundle_kwargs={"device": "cuda"})
    bundle = pipe.bundle
    if args.training_checkpoint is not None and args.worker == "export":
        if args.method not in {
            "mld",
            "motionlcm",
            "edge",
            "motioncanvas",
            "hymotion_t2m",
            "dart",
            "tm2d",
        }:
            raise ValueError("This training checkpoint bridge covers MLD/MotionLCM")
        state = torch.load(
            args.training_checkpoint, map_location="cpu", weights_only=False, mmap=True
        )
        if args.method in {"motioncanvas", "hymotion_t2m"}:
            bundle.load_state_dict_selective(state, strict=True)
        elif args.method == "tm2d":
            from motius.models.tm2d.network import JointTokenNetwork

            bundle.joint_network = JointTokenNetwork(
                bundle.audio_transformer, bundle.text_transformer
            )
            for name in ["joint_network", "vq_encoder", "quantizer", "vq_decoder"]:
                getattr(bundle, name).load_state_dict(state[name], strict=True)
            bundle.text_transformer.decoder.load_state_dict(
                bundle.audio_transformer.decoder.state_dict(), strict=True
            )
            bundle.eval()
        elif args.method == "dart":
            import copy

            for name in ["denoiser_model", "vae_model"]:
                getattr(bundle, name).load_state_dict(state[name], strict=True)
                average = copy.deepcopy(getattr(bundle, name))
                average.load_state_dict(state[name + "_ema"], strict=True)
                setattr(bundle, name + "_ema", average)
                # The original DART rollout selects the saved averaged model.
                setattr(bundle, name, average)
            bundle.eval()
        elif args.method == "edge":
            bundle.network.load_state_dict(state["network"], strict=True)
            import copy

            bundle.ema_network = copy.deepcopy(bundle.network)
            bundle.ema_network.load_state_dict(state["ema_network"], strict=True)
            if args.edge_use_ema:
                bundle.network = bundle.ema_network
        else:
            for name in ["vae", "denoiser", "text_encoder"]:
                getattr(bundle, name).load_state_dict(state[name], strict=True)
    if args.method in {"motioncanvas", "hymotion_t2m"}:
        bundle.eval()
        captions = ["A person walks forward.", "A person turns and raises both arms."]
        if args.worker == "export":
            from motius.motion.conditioning.hymotion_text import encode_hymotion_text

            encode_hymotion_text(bundle, captions)
            with torch.no_grad():
                bundle.null_vtxt_feat[0, 0, 0].add_(0.125)
                bundle.special_game_vtxt_feat[0, 0, 0].add_(0.0625)
                list(bundle.motion_transformer.parameters())[-1].add_(0.03125)
                bundle._text_encoder.sentence_emb_text_encoder.text_model.final_layer_norm.bias[
                    0
                ].add_(
                    0.0625
                )
                bundle._text_encoder.llm_text_encoder.embed_tokens.weight[7, 0].add_(
                    0.125
                )
        if args.method == "motioncanvas":
            output = pipe.infer_m2m(
                captions=captions,
                num_frames=[24, 20],
                seed=81,
                data_src=["Game", "Other"],
            )
        else:
            output = pipe.infer_text_to_motion(
                captions,
                num_frames=[24, 20],
                seed_input=[81, 923],
                use_special_game_feat=True,
            )
        condition = {
            name: getattr(bundle, name)
            for name in [
                "null_vtxt_feat",
                "null_ctxt_input",
                "special_game_vtxt_feat",
                "special_game_ctxt_feat",
            ]
        }
        values = tensors(
            dict(
                neural=hashes(bundle.motion_transformer),
                condition=condition,
                text_neural=hashes(bundle._text_encoder),
                motion=output,
                mean=bundle.mean,
                std=bundle.std,
            )
        )
    elif args.method == "prism":
        import numpy as np

        if args.worker == "export":
            token = pipe.backend.tokenizer("A person walks forward.").input_ids[0]
            with torch.no_grad():
                list(bundle.transformer.parameters())[-1].add_(0.03125)
                next(bundle.vae.parameters()).view(-1)[0].add_(0.000125)
                bundle.text_encoder.shared.weight[token, 0].add_(0.125)
        prefix = args.output_dir / "prefix.npz"
        np.savez(
            prefix,
            global_orient=np.zeros((9, 3), np.float32),
            body_pose=np.zeros((9, 63), np.float32),
            transl=np.tile(np.array([[0.0, 0.0, 0.9]], np.float32), (9, 1)),
        )
        output = dict(
            text=pipe.infer_text_to_motion(
                ["A person walks forward.", "A person turns."],
                [17, 13],
                seed=81,
                num_inference_steps=50,
            ),
            prefix=pipe.infer_temporal_motion_completion(
                "A person waves.",
                str(prefix),
                num_frames=21,
                condition_num_frames=5,
                seed=91,
                num_inference_steps=50,
            ),
            sequential=pipe.infer_sequential_text_to_motion(
                ["walk forward", "turn left"],
                [17, 13],
                pad_to_frames=33,
                ar_condition_frames=5,
                seed=93,
                num_inference_steps=50,
            ),
        )
        values = tensors(
            dict(
                neural={
                    name: hashes(getattr(bundle, name))
                    for name in ["transformer", "vae", "text_encoder"]
                },
                motion=output,
            )
        )
    elif args.method == "vermo":
        import numpy as np

        if args.worker == "export":
            with torch.no_grad():
                bundle.lm.get_input_embeddings().weight[
                    pipe.processor.output_bos_id, 0
                ].add_(0.125)
                bundle.lm.model.norm.weight[0].add_(0.03125)
                next(bundle.motion_tokenizer.parameters()).view(-1)[0].add_(0.000125)
        motions = [
            np.random.default_rng(81 + i).normal(size=(length, 138)).astype(np.float32)
            * 0.01
            for i, length in enumerate([16, 25])
        ]
        output = pipe.infer_motion_to_text(
            motions, input_representation="vermo138", max_new_tokens=16
        )
        values = tensors(
            dict(
                neural={
                    name: hashes(getattr(bundle, name))
                    for name in ["lm", "motion_tokenizer"]
                },
                captions=output,
            )
        )
    elif args.method == "tm2d":
        if args.worker == "export":
            with torch.no_grad():
                bundle.audio_transformer.decoder.trg_word_emb.weight[7, 0].add_(0.125)
                next(bundle.text_transformer.encoder.parameters()).view(-1)[0].add_(
                    0.0625
                )
                list(bundle.vq_decoder.parameters())[-1].add_(0.03125)
                bundle.text_transformer.decoder.load_state_dict(
                    bundle.audio_transformer.decoder.state_dict(), strict=True
                )
        text = pipe.infer_text_to_motion(
            "A person walks forward.",
            num_frames=32,
            output_fps=60.0,
            sample=True,
            seed=81,
        )
        features = torch.sin(torch.arange(62 * 438).reshape(62, 438) * 0.017) * 0.03
        dance = pipe.infer_music_to_dance(
            music_features=features, initial_token=423, sample=True, seed=87
        )
        values = tensors(
            dict(
                text=vars(text),
                audio=vars(dance),
                neural={
                    name: hashes(getattr(bundle, name))
                    for name in [
                        "audio_transformer",
                        "text_transformer",
                        "vq_encoder",
                        "quantizer",
                        "vq_decoder",
                    ]
                },
            )
        )
    elif args.method == "dart":
        bundle.load_text_encoder()
        if args.worker == "export":
            with torch.no_grad():
                list(bundle.denoiser_model.parameters())[-1].add_(0.03125)
                next(bundle.vae_model.parameters()).view(-1)[0].add_(0.0625)
                bundle.clip_model.ln_final.bias[0].add_(0.125)
        pipe.respacing = "ddim5"
        from types import SimpleNamespace
        from motius.diffusion.dart import create_gaussian_diffusion

        options = SimpleNamespace(**vars(pipe.denoiser_args.diffusion_args))
        options.respacing = pipe.respacing
        pipe.diffusion = create_gaussian_diffusion(options)
        output = pipe.infer_text_to_motion(
            ["A person walks forward.", ""],
            [17, 20],
            seed=81,
        )
        values = tensors(
            dict(
                neural={
                    "denoiser": hashes(bundle.denoiser_model),
                    "vae": hashes(bundle.vae_model),
                    "text": hashes(bundle.clip_model),
                },
                motion=output,
                options={
                    name: getattr(pipe, name)
                    for name in [
                        "translation_source",
                        "rollout_body_type",
                        "initial_transform",
                    ]
                },
            )
        )
    elif args.method == "edge":
        if args.worker == "export":
            with torch.no_grad():
                bundle.network.final_layer.bias[0].add_(0.03125)
        music = (
            torch.sin(torch.arange(2 * 150 * 4800).reshape(2, 150, 4800) * 0.017) * 0.03
        )
        output = pipe.infer_music_to_dance(music_features=music, seed=81)
        values = tensors(
            dict(
                neural=hashes(bundle.network),
                joints=output.joints,
                edge_motion=output.edge_motion,
                motion135=output.motion_135,
                scale=bundle.normalizer_scale,
                offset=bundle.normalizer_min,
            )
        )
    elif args.method in {"mdm", "condmdi", "omnicontrol"}:
        from motius.utils.gaussian_motion_checkpoint import unwrap_network

        net = unwrap_network(bundle.net)
        if args.worker == "export":
            with torch.no_grad():
                net.clip_model.ln_final.bias[0].add_(0.125)
                list(
                    p
                    for name, p in net.named_parameters()
                    if not name.startswith("clip_model.")
                )[-1].add_(0.03125)
        torch.manual_seed(81)
        options = {"return_absolute": True} if args.method == "condmdi" else {}
        output = pipe.infer_t2m(["A person walks forward."], [40], **options)
        values = tensors(dict(neural=hashes(net), motion=output))
    elif args.method == "motionclr":
        if args.worker == "export":
            with torch.no_grad():
                bundle.network.clip_model.ln_final.bias[0].add_(0.125)
                list(bundle.network.unet.parameters())[-1].add_(0.03125)
        output = pipe.infer_t2m(["A person walks forward."], [48], seed=81)
        values = tensors(
            dict(
                neural=hashes(bundle.network),
                motion=output,
                mean=bundle.mean,
                std=bundle.std,
            )
        )
    elif args.method == "projflow":
        if args.worker == "export":
            with torch.no_grad():
                bundle.net.clip_model.ln_final.bias[0].add_(0.125)
                list(bundle.net.parameters())[-1].add_(0.03125)
        motion = (
            torch.sin(torch.arange(40 * 22 * 3).reshape(40, 22, 3) * 0.017) * 0.03
        ).numpy()
        motion[..., 1] += 0.9
        output = pipe.infer_control(["A person walks forward."], [motion], seed=81)
        values = tensors(
            dict(
                neural=hashes(bundle.net),
                motion=output,
                mean=bundle.joints_mean,
                std=bundle.joints_std,
            )
        )
    elif args.method == "tm2t":
        if args.worker == "export":
            with torch.no_grad():
                bundle.transformer.decoder.trg_word_emb.weight[7].add_(0.125)
        input = (
            torch.sin(torch.arange(40 * 263, device="cuda").reshape(40, 263) * 0.017)
            * 0.03
        )
        input[..., 3] = 0.9
        with torch.no_grad():
            output = pipe.infer_m2t([input])
            tokens = torch.tensor([[1024, 3, 4, 5, 1025, 1026]], device="cuda")
            words = torch.tensor([[bundle.text_start_index, 4, 7, 6]], device="cuda")
            logits = bundle.transformer(tokens, words)
        state = {
            "vq_encoder": hashes(bundle.vq_encoder),
            "quantizer": hashes(bundle.quantizer),
            "transformer": hashes(bundle.transformer),
        }
        values = tensors(
            dict(
                neural=state,
                caption=output,
                logits=logits,
                mean=bundle.mean,
                std=bundle.std,
            )
        )
    elif args.method == "motionmillion":
        if args.worker == "export":
            with torch.no_grad():
                bundle.text_model.shared.weight[7, 0].add_(0.125)
                next(bundle.ar.parameters()).view(-1)[0].add_(0.03125)
                next(bundle.vqvae.parameters()).view(-1)[0].add_(0.0625)
        torch.manual_seed(81)
        output = pipe.infer_t2m(["A person walks forward."])
        from motius.motion.conditioning.motionmillion_text import (
            prepare_motionmillion_text,
        )

        with torch.no_grad():
            features, mask = bundle.encode_text(
                prepare_motionmillion_text(
                    bundle.text_model_name, ["A person walks forward.", ""]
                )
            )
        state = {
            "vqvae": hashes(bundle.vqvae),
            "ar": hashes(bundle.ar),
            "text": hashes(bundle.text_model),
        }
        values = tensors(
            dict(
                neural=state,
                motion=output,
                text=features,
                mask=mask,
                mean=bundle.mean,
                std=bundle.std,
            )
        )
    else:
        if args.worker == "export":
            with torch.no_grad():
                text = bundle.text_encoder.text_model
                projection = (
                    text.text_projection
                    if bundle.text_encoder.name == "clip"
                    else text.projection
                )
                projection.weight[0, 0].add_(0.125)
                list(bundle.denoiser.parameters())[-1].add_(0.03125)
        torch.manual_seed(81)
        output = pipe.infer_t2m(
            ["A person walks forward."],
            [40],
            num_inference_steps=50 if args.method == "mld" else 2,
        )
        state = {
            "vae": hashes(bundle.vae),
            "denoiser": hashes(bundle.denoiser),
            "text": hashes(bundle.text_encoder),
        }
        values = tensors(
            dict(neural=state, motion=output, mean=bundle.mean, std=bundle.std)
        )
    torch.save(values, args.output_dir / f"{args.worker}.pt")
    if args.worker == "export":
        options = (
            {"use_ema": True} if args.method == "edge" and args.edge_use_ema else {}
        )
        if args.method in {"dart", "prism", "vermo"}:
            pipe.save_pretrained(str(args.output_dir / "exported"), **options)
        else:
            bundle.save_pretrained(str(args.output_dir / "exported"), **options)
    else:
        assert not any(
            name == "networks"
            or name.startswith("networks.")
            or name == "mld"
            or name.startswith("mld.")
            for name in sys.modules
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method",
        choices=[
            "tm2t",
            "motionlcm",
            "mld",
            "edge",
            "dart",
            "tm2d",
            "prism",
            "vermo",
            "motioncanvas",
            "hymotion_t2m",
            "motionmillion",
            "mdm",
            "condmdi",
            "omnicontrol",
            "projflow",
            "motionclr",
        ],
        required=True,
    )
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=["export", "cold"])
    parser.add_argument("--training-checkpoint", type=Path)
    parser.add_argument("--edge-use-ema", action="store_true")
    args = parser.parse_args()
    args.artifact = args.artifact.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = {
        "complete": False,
        "scope": args.method
        + " actual modified current neural tensors, complete public inference and frozen/current states through offline cold AutoPipeline loading. Export/migration evidence, not an independent original-source comparison.",
        "tests": [],
    }
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["export", "cold"]:
        checkpoint_args = (
            ["--training-checkpoint", str(args.training_checkpoint.resolve())]
            if args.training_checkpoint is not None
            else []
        )
        if args.edge_use_ema:
            checkpoint_args += ["--edge-use-ema"]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--method",
                args.method,
                "--artifact",
                str(args.artifact),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ]
            + checkpoint_args,
            cwd=ROOT,
            env={
                **os.environ,
                "PYTHONPATH": str(ROOT),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            },
            check=True,
        )
    try:
        compare_full(
            torch.load(args.output_dir / "export.pt", weights_only=True),
            torch.load(args.output_dir / "cold.pt", weights_only=True),
            "",
            report["tests"],
        )
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    report["complete"] = True
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(args.method, "current weights and cold AutoPipeline exact")


if __name__ == "__main__":
    main()
