"""Compare author TM2D networks and sampling with the released native artifact."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from verify_motionstreamer_source import hashes

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = json.loads((args.artifact / "tm2d_config.json").read_text())["config"]
    c = config
    if args.worker == "source":
        sys.path.insert(0, str(args.upstream / "tm2d_60fps"))
        from networks.transformer_x_lf3 import TransformerV1, TransformerV2
        from networks.modules import VQEncoderV3, VQDecoderV3
        from networks.quantizer import Quantizer

        # Keep the original geometry functions intact while omitting unrelated
        # application imports using the removed NumPy np.float alias.
        from verify_dart_source_parity import source_definitions

        geometry = dict(torch=torch, np=np)
        source_definitions(
            args.upstream / "tm2d_60fps/common/quaternion.py",
            geometry,
            {"qinv", "qrot"},
        )
        source_definitions(
            args.upstream / "tm2d_60fps/scripts/motion_process.py",
            geometry,
            {"recover_root_rot_pos", "recover_from_ric"},
        )
        recover_from_ric = geometry["recover_from_ric"]

        common = dict(
            d_src_word_vec=512,
            d_trg_word_vec=512,
            d_model=c["d_model"],
            d_inner=c["d_inner"],
            n_enc_layers=c["n_encoder_layers"],
            n_dec_layers=c["n_decoder_layers"],
            n_head=c["n_head"],
            d_k=c["d_k"],
            d_v=c["d_v"],
            dropout=c["dropout"],
            n_src_position=100,
            n_trg_position=100,
        )
        text = TransformerV2(
            c["text_vocabulary_size"],
            c["text_pad_id"],
            c["motion_vocabulary_size"],
            c["motion_pad_id"],
            **common,
        )
        common["d_src_word_vec"] = 438
        audio = TransformerV1(c["motion_vocabulary_size"], c["motion_pad_id"], **common)
        encoder = VQEncoderV3(283, [1024] * 3, 3)
        decoder = VQDecoderV3(1024, [1024] * 3 + [287], 3, 3)
        quantizer = Quantizer(1024, 1024, 1.0)
        roots = dict(
            text_transformer=text,
            audio_transformer=audio,
            vq_encoder=encoder,
            vq_decoder=decoder,
            quantizer=quantizer,
        )
        for name, network in roots.items():
            network.load_state_dict(
                load_file(str(args.artifact / (name + ".safetensors"))), strict=True
            )
            network.to(args.device).eval()
        mean = torch.from_numpy(np.load(args.artifact / "mean.npy")).to(args.device)
        std = torch.from_numpy(np.load(args.artifact / "std.npy")).to(args.device)

        def decode(tokens):
            value = decoder(quantizer.get_codebook_entry(tokens)) * std + mean
            return value, recover_from_ric(value, 24)

    else:
        from motius.pipelines.tm2d import TM2DPipeline

        pipe = TM2DPipeline.from_pretrained(str(args.artifact)).to(args.device)
        bundle = pipe.bundle
        roots = {
            name: getattr(bundle, name)
            for name in [
                "text_transformer",
                "audio_transformer",
                "vq_encoder",
                "vq_decoder",
                "quantizer",
            ]
        }
        text, audio, encoder, decoder, quantizer = [
            roots[name]
            for name in [
                "text_transformer",
                "audio_transformer",
                "vq_encoder",
                "vq_decoder",
                "quantizer",
            ]
        ]
        decode = pipe._decode
    output = {"neural": {name: hashes(value) for name, value in roots.items()}}
    vocabulary = json.loads((args.artifact / "vocab.json").read_text())
    for case, (count, sample, top_k) in enumerate(
        [(4, False, None), (9, True, None), (12, True, 50)]
    ):
        words = ["a", "person", "walks"]
        ids = [
            vocabulary.get(word, vocabulary["unk"]) for word in ["sos", *words, "eos"]
        ]
        ids += [c["text_length_id"]] * max(0, min(count, 22) - len(ids))
        ids += [c["text_pad_id"]] * (22 - len(ids))
        ids = torch.tensor([ids], device=args.device)
        with torch.no_grad():
            if args.worker == "source":
                encoded, mask = text.encoding(ids)
                torch.manual_seed(91 + case)
                tokens = text.sample_with_enc_output(
                    ids,
                    encoded,
                    c["motion_start_id"],
                    c["motion_end_id"],
                    end_idx=count,
                    sample=sample,
                    top_k=top_k,
                )[:, 1:]
                motion, joints = decode(tokens)
            else:
                encoded, mask = text.encode(ids)
                result = pipe.infer_text_to_motion(
                    "prepared lexical boundary",
                    num_frames=count * 8,
                    output_fps=60.0,
                    pretokenized=words,
                    sample=sample,
                    top_k=top_k,
                    seed=91 + case,
                )
                tokens = torch.as_tensor(result.motion_tokens, device=args.device)[None]
                motion = torch.as_tensor(result.model_motion, device=args.device)[None]
                joints = torch.as_tensor(result.native_joints, device=args.device)[None]
        output["text" + str(case)] = dict(
            encoded=encoded.cpu(),
            mask=mask.cpu(),
            tokens=tokens.cpu(),
            motion=motion.cpu(),
            joints=joints.cpu(),
        )
    for case, (frames, sample, top_k) in enumerate(
        [(10, False, None), (62, True, None), (62, True, 50)]
    ):
        features = torch.from_numpy(
            np.random.default_rng(817 + case)
            .normal(size=(1, frames, 438))
            .astype(np.float32)
        ).to(args.device)
        with torch.no_grad():
            if args.worker == "source":
                values, masks = [], []
                for start in range(0, frames, 49):
                    value = features[:, start : start + 50]
                    encoded, mask = audio.encoding(
                        value, src_non_pad_lens=torch.tensor([value.shape[1]])
                    )
                    values.append(encoded[:, 1:] if start else encoded)
                    masks.append(mask[:, :, 1:] if start else mask)
                encoded = torch.cat(values, 1)
                mask = torch.cat(masks, 2)
                torch.manual_seed(193 + case)
                start = 0
                previous = torch.tensor([[423]], device=args.device)
                parts = []
                while start < frames:
                    end = min(start + 50, frames)
                    tokens = audio.sample_with_trg_seq_enc_output(
                        features[:, start:end],
                        torch.tensor([end - start]),
                        encoded[:, start:end],
                        c["motion_start_id"],
                        c["motion_end_id"],
                        previous,
                        start_idx=start + 1,
                        end_idx=end,
                        sample=sample,
                        top_k=top_k,
                    )
                    parts.append(tokens if start == 0 else tokens[:, 1:])
                    if end == frames:
                        break
                    previous = tokens[:, -1:]
                    start = end - 1
                tokens = torch.cat(parts, 1)
                motion, joints = decode(tokens)
            else:
                encoded, mask = pipe._encode_music_chunks(features)
                result = pipe.infer_music_to_dance(
                    music_features=features,
                    initial_token=423,
                    sample=sample,
                    top_k=top_k,
                    seed=193 + case,
                )
                tokens = torch.as_tensor(result.motion_tokens, device=args.device)[None]
                motion = torch.as_tensor(result.model_motion, device=args.device)[None]
                joints = torch.as_tensor(result.native_joints, device=args.device)[None]
        output["audio" + str(case)] = dict(
            encoded=encoded.cpu(),
            mask=mask.cpu(),
            tokens=tokens.cpu(),
            motion=motion.cpu(),
            joints=joints.cpu(),
        )
    torch.manual_seed(183)
    physical = torch.randn(2, 64, 283, device=args.device)
    with torch.no_grad():
        latent = encoder(physical)
        codes = (
            quantizer.map2index(latent).reshape(2, -1)
            if args.worker == "source"
            else quantizer.indices(latent)
        )
    output["codec"] = dict(latent=latent.cpu(), codes=codes.cpu())
    # Full original joint objective: both text/music encoders use the audio
    # decoder during training; source save later copies it into the text branch.
    from types import SimpleNamespace
    import hashlib

    for case, smoothing in enumerate([False, True]):
        text.zero_grad(set_to_none=True)
        audio.zero_grad(set_to_none=True)
        text.train()
        audio.train()
        torch.manual_seed(113 + case)
        dance = torch.randint(0, 1024, (2, 8), device=args.device)
        motion = torch.randint(0, 1024, (2, 7), device=args.device)
        dance[:, 0] = motion[:, 0] = 1024
        dance[1, -2:] = motion[1, -2:] = 1026
        features = torch.randn(2, 8, 438, device=args.device)
        lengths = torch.tensor([8, 5], device=args.device)
        words = ids.expand(2, -1).clone()
        if args.worker == "source":
            from collections import OrderedDict
            from networks.transformer import cal_performance
            from verify_dart_source_parity import source_definitions

            scope = dict(
                Trainer=object,
                torch=torch,
                OrderedDict=OrderedDict,
                cal_performance=cal_performance,
            )
            source_definitions(
                args.upstream / "tm2d_60fps/networks/trainers_x5.py",
                scope,
                {"TransformerATMTrainerV2"},
            )
            original_class = scope["TransformerATMTrainerV2"]
            trainer = original_class.__new__(original_class)
            trainer.a2d_transformer, trainer.t2m_transformer = audio, text
            trainer.device = torch.device(args.device)
            trainer.opt = SimpleNamespace(
                mot_pad_idx=1026,
                label_smoothing=smoothing,
                lambda_a2d=0.7,
                lambda_t2m=1.3,
            )
            torch.manual_seed(819 + case)
            trainer.forward((features, lengths, dance, None, words, None, motion))
            trainer.backward()
            loss = trainer.loss["loss/net"]
            audio_loss, text_loss = trainer.loss["a2d"], trainer.loss["t2m"]
        else:
            from motius.trainers.tm2d import TM2DTrainer

            trainer = TM2DTrainer(
                bundle, label_smoothing=smoothing, lambda_a2d=0.7, lambda_t2m=1.3
            )
            bundle.train()
            torch.manual_seed(819 + case)
            terms = trainer.train_step(
                dict(
                    audio_features=features,
                    audio_lengths=lengths,
                    dance_tokens=dance,
                    motion_tokens=motion,
                    text_tokens=words,
                )
            )
            loss, audio_loss, text_loss = (
                terms["loss"],
                terms["tm2d_audio"],
                terms["tm2d_text"],
            )
        loss.backward()
        output["training" + str(case)] = dict(
            loss=loss.detach().cpu(),
            audio_loss=audio_loss.detach().cpu(),
            text_loss=text_loss.detach().cpu(),
            gradients={
                root: {
                    name: hashlib.sha256(
                        parameter.grad.detach().cpu().contiguous().numpy().tobytes()
                    ).hexdigest()
                    for name, parameter in network.named_parameters()
                    if parameter.grad is not None
                }
                for root, network in [("audio", audio), ("text", text)]
            },
            cpu_rng=torch.get_rng_state(),
            cuda_rng=(
                torch.cuda.get_rng_state()
                if str(args.device).startswith("cuda")
                else torch.empty(0, dtype=torch.uint8)
            ),
        )
    torch.save(output, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--worker", choices=["source", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    from verify_prompthmr_video_parity import compare_full
    from verify_intergen_source_parity import digest
    import torch

    report = dict(
        complete=False,
        scope="Independent author TM2D source and released mirrored weights. Prepared lexical tokens/438-D music features; three text and three single/multiple-window music sampling cases, all five neural hashes, VQ encoding, physical motion and joints. Raw caption NLP/raw music extraction and joint bimodal fusion are separate gates.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        artifact_files={
            p.name: digest(p)
            for p in args.artifact.iterdir()
            if p.suffix in [".safetensors", ".json", ".npy"]
        },
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for kind in ["source", "native"]:
            subprocess.run(
                [sys.executable, __file__, *sys.argv[1:], "--worker", kind],
                cwd=ROOT,
                env={
                    **os.environ,
                    "PYTHONPATH": str(ROOT),
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
                check=True,
            )
        compare_full(
            torch.load(args.output_dir / "source.pt", weights_only=False),
            torch.load(args.output_dir / "native.pt", weights_only=False),
            "",
            report["tests"],
        )
        report["complete"] = True
    finally:
        path.write_text(json.dumps(report, indent=2) + "\n")
    print("TM2D original source comparison exact")


if __name__ == "__main__":
    main()
