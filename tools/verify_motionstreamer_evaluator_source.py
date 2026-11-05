"""Author MotionStreamer matching encoders and original dataset/collate arithmetic."""

from pathlib import Path
import argparse, json, os, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from verify_motionstreamer_source import hashes
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    state = load_file(str(args.artifact / "model.safetensors"))
    mean = np.load(args.artifact / "stats/mean.npy")
    std = np.load(args.artifact / "stats/std.npy")
    if args.worker == "source":
        sys.path.insert(0, str(args.upstream / "Evaluator_272"))
        checkpoint = torch.load(args.original, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        for source, target in [
            ("textencoder.", "text_encoder."),
            ("motionencoder.", "motion_encoder."),
        ]:
            original = {
                target + k[len(source) :]: v
                for k, v in checkpoint.items()
                if k.startswith(source)
            }
            assert original.keys() == {k for k in state if k.startswith(target)}
            assert all(torch.equal(v, state[k]) for k, v in original.items())
        del checkpoint
        # The original constructor uses AutoModel.from_pretrained. Materialize
        # the actual author's DistilBERT weights for that unchanged constructor.
        backbone = args.output_dir / "source-backbone"
        backbone.mkdir(parents=True, exist_ok=True)
        model = AutoModel.from_config(
            AutoConfig.from_pretrained(
                args.artifact / "tokenizer", local_files_only=True
            )
        )
        model.load_state_dict(
            {
                k.removeprefix("text_encoder.text_model."): v
                for k, v in state.items()
                if k.startswith("text_encoder.text_model.")
            },
            strict=True,
        )
        model.save_pretrained(backbone)
        AutoTokenizer.from_pretrained(
            args.artifact / "tokenizer", local_files_only=True
        ).save_pretrained(backbone)
        del model
        sys.path.insert(0, str(args.upstream / "Evaluator_272"))
        from mld.models.architectures.temos.textencoder.distillbert_actor import (
            DistilbertActorAgnosticEncoder,
        )
        from mld.models.architectures.temos.motionencoder.actor import (
            ActorAgnosticEncoder,
        )
        from mld.data.humanml.data.dataset import Text2MotionDatasetV2
        from mld.data.utils import mld_collate

        text = (
            DistilbertActorAgnosticEncoder(str(backbone), vae=True)
            .to(args.device)
            .eval()
        )
        motion = ActorAgnosticEncoder(272, vae=True).to(args.device).eval()
        text.load_state_dict(
            {
                k.removeprefix("text_encoder."): v
                for k, v in state.items()
                if k.startswith("text_encoder.")
            },
            strict=True,
        )
        motion.load_state_dict(
            {
                k.removeprefix("motion_encoder."): v
                for k, v in state.items()
                if k.startswith("motion_encoder.")
            },
            strict=True,
        )
    else:
        from motius.pipelines.motionstreamer_matching import (
            MotionStreamerMatchingPipeline,
        )

        pipe = MotionStreamerMatchingPipeline.from_pretrained(
            str(args.artifact), bundle_kwargs={"device": args.device}
        )
        text, motion = pipe.bundle.text_encoder, pipe.bundle.motion_encoder
    output = dict(neural=dict(text=hashes(text), motion=hashes(motion)))
    for case, (batch, dtype) in enumerate(
        [(1, np.float32), (8, np.float32), (8, np.float64)]
    ):
        captions = [
            "A person walks.",
            "",
            "Someone is turning around slowly.",
            "A person raises the left arm.",
            "STEP forward!",
            "Dance in-place.",
            "A person crouches.",
            "a person runs",
        ][:batch]
        lengths = [20, 72, 36, 100, 24, 52, 300, 28][:batch]
        motions = [
            np.random.default_rng(117 + i).normal(size=(length, 272)).astype(dtype)
            * 0.1
            for i, length in enumerate(lengths)
        ]
        if args.worker == "source":
            import random

            dataset = Text2MotionDatasetV2.__new__(Text2MotionDatasetV2)
            dataset.pointer = 0
            dataset.unit_length = 4
            dataset.mean = mean
            dataset.std = std
            dataset.name_list = ["fixture_" + str(i) for i in range(batch)]
            dataset.data_dict = {
                name: dict(
                    motion=motions[i],
                    length=lengths[i],
                    text=[dict(caption=captions[i])],
                )
                for i, name in enumerate(dataset.name_list)
            }
            rows = []
            for i in range(batch):
                np.random.seed(0)
                random.seed(0)
                rows.append(dataset[i])
            prepared = mld_collate(rows)
            order = [int(name) for name in prepared["retrieval_name"]]
            inverse = torch.argsort(torch.tensor(order, device=args.device))
            with torch.no_grad():
                a = text(captions).loc
                b = motion(prepared["motion"].to(args.device), prepared["length"]).loc[
                    inverse
                ]
        else:
            a = torch.from_numpy(pipe.encode_texts(captions)).to(args.device)
            b = torch.from_numpy(pipe.encode_motions(motions)).to(args.device)
        output["case" + str(case)] = dict(text=a.cpu(), motion=b.cpu())
    torch.save(output, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "original", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["source", "native"])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in ["upstream", "artifact", "original", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        return worker(args)
    import torch
    from verify_prompthmr_video_parity import compare_full
    from verify_intergen_source_parity import digest

    report = dict(
        complete=False,
        scope="Actual original MotionStreamer matching encoders, original released epoch99 checkpoint, complete raw-text frontend and unchanged original dataset/collate on explicit aligned single-crop fixtures. Single/ragged Float32/Float64 physical motions; undo only original collate output permutation. Full benchmark caption selection/random-crop protocol excluded.",
        original_sha256=digest(args.original),
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
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
    print("MotionStreamer original matching evaluator exact")


if __name__ == "__main__":
    main()
