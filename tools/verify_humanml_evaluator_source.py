"""Compare original HumanML3D raw text/dataset/wrapper with native embeddings."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import numpy as np
    import torch
    import random
    from types import SimpleNamespace
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        from networks.evaluator_wrapper import EvaluatorModelWrapper
        from data.dataset import RawTextDataset, Text2MotionDatasetV2, collate_fn
        from utils.word_vectorizer import WordVectorizer

        options = SimpleNamespace(
            dataset_name="t2m",
            device=torch.device("cuda"),
            checkpoints_dir=str(args.checkpoints),
            dim_movement_enc_hidden=512,
            dim_movement_latent=512,
            unit_length=4,
        )
        runtime = EvaluatorModelWrapper(options)
        vectorizer = WordVectorizer(str(args.upstream / "glove"), "our_vab")
        mean = np.load(args.artifact / "stats/mean.npy")
        std = np.load(args.artifact / "stats/std.npy")
        networks = {
            name: getattr(runtime, name)
            for name in ["movement_encoder", "text_encoder", "motion_encoder"]
        }
    else:
        from motius.pipelines.humanml3d_matching import HumanML3DMatchingPipeline

        pipeline = HumanML3DMatchingPipeline.from_pretrained(str(args.artifact))
        networks = {
            name: getattr(pipeline.bundle, name)
            for name in ["movement_encoder", "text_encoder", "motion_encoder"]
        }
        mean = pipeline.bundle.mean.cpu().numpy()
        std = pipeline.bundle.std.cpu().numpy()
    result = dict(
        neural={name: hashes(net) for name, net in networks.items()}, cases=[]
    )
    captions = [
        "A Person walks forward.",
        "The left arm is raised, then lowered.",
        "",
        "TURN left!",
        "a person walks " * 12,
        "The person walks counter-clockwise.",
        "Both feet move quickly.",
        "A man picks up a ball and places it on a chair.",
    ]
    lengths = [40, 196, 80, 120, 64, 100, 156, 92]
    for case, (batch, dtype) in enumerate(
        [(1, np.float32), (8, np.float32), (8, np.float64)]
    ):
        texts = captions[:batch]
        motions = [
            (
                np.sin(np.arange(length * 263).reshape(length, 263) * 0.017 + i) * 0.3
                + 0.1
            ).astype(dtype)
            for i, length in enumerate(lengths[:batch])
        ]
        if args.worker == "upstream":
            text_file = args.output_dir / f"captions-{case}.txt"
            text_file.write_text("\n".join(texts) + "\n")
            raw = RawTextDataset(options, mean, std, str(text_file), vectorizer)
            # Execute the original dataset's complete token/pad/normalization
            # path on an in-memory prepared cohort. Fixed single-unit crops
            # retain each already 4-aligned physical sequence in full.
            dataset = Text2MotionDatasetV2.__new__(Text2MotionDatasetV2)
            dataset.opt = options
            dataset.mean, dataset.std = mean, std
            dataset.w_vectorizer = vectorizer
            dataset.pointer = 0
            dataset.max_motion_length = 196
            dataset.name_list = [str(index) for index in range(batch)]
            dataset.data_dict = {
                str(index): dict(
                    motion=motion, length=len(motion), text=[raw.data_dict[index]]
                )
                for index, motion in enumerate(motions)
            }
            samples = []
            for index in range(batch):
                np.random.seed(0)
                random.seed(71 + index)
                sample = dataset[index]
                assert sample[5] == lengths[index]
                samples.append(sample)
            values = collate_fn(samples.copy())
            words, pos, sorted_captions, cap_lengths, normalized, motion_lengths, _ = (
                values
            )
            text_features, motion_features = runtime.get_co_embeddings(
                words, pos, cap_lengths, normalized, motion_lengths
            )
            # The original wrapper explicitly returns motion-length order.
            # The unified public API promises input order; undo only that
            # documented external permutation, without changing its networks.
            stripped_texts = [text.strip() for text in texts]
            caption_order = np.asarray(
                [stripped_texts.index(text) for text in sorted_captions]
            )
            motion_order = np.argsort(motion_lengths.tolist())[::-1].copy()
            inverse = np.argsort(caption_order[motion_order])
            output = dict(text=text_features[inverse], motion=motion_features[inverse])
        else:
            output = dict(
                text=pipeline.encode_texts(texts),
                motion=pipeline.encode_motions(motions),
            )
        result["cases"].append(tensors(output))
        print(
            args.worker,
            "raw text/ragged physical motion case",
            case,
            "saved",
            flush=True,
        )
    result["mean"] = tensors(mean)
    result["std"] = tensors(std)
    torch.save(result, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["upstream", "artifact", "checkpoints", "output-dir"]:
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    for name in ["upstream", "artifact", "checkpoints", "output_dir"]:
        setattr(args, name, getattr(args, name).resolve())
    args.output_dir.mkdir(exist_ok=True, parents=True)
    if args.worker:
        return worker(args)
    import torch
    from safetensors.torch import load_file
    from verify_prompthmr_video_parity import compare_full

    torch.set_num_threads(4)

    checkpoint = args.checkpoints / "t2m/text_mot_match/model/finest.tar"
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    native = load_file(args.artifact / "model.safetensors")
    for name, tensor in native.items():
        module, key = name.split(".", 1)
        assert torch.equal(source[module][key], tensor), name
    for name in ["mean", "std"]:
        assert (args.artifact / f"stats/{name}.npy").read_bytes() == (
            args.checkpoints / f"t2m/Comp_v6_KLD01/meta/{name}.npy"
        ).read_bytes()
    for name in ["our_vab_data.npy", "our_vab_idx.pkl", "our_vab_words.pkl"]:
        assert (args.artifact / "glove" / name).read_bytes() == (
            args.upstream / "glove" / name
        ).read_bytes()
    report = dict(
        complete=False,
        scope="Original Guo HumanML3D raw-text dataset, actual prepared-motion DatasetV2 normalization/collation and complete EvaluatorModelWrapper versus public native pipeline. Actual author checkpoint, all 42 neural tensors, statistics and GloVe bytes independently verified. Three single/ragged cases cover case-sensitive/empty/long/punctuated captions and Float32/Float64 physical inputs, with explicit inverse of original output sorting. Shared spaCy 3.7.5/en_core_web_sm3.7.1, controlled FP32 deterministic CUDA. Random crop/caption-selection full corpus and historical benchmark metrics excluded.",
        upstream_commit=subprocess.check_output(
            ["git", "-C", str(args.upstream), "rev-parse", "HEAD"], text=True
        ).strip(),
        checkpoint_sha256=hashlib.file_digest(
            checkpoint.open("rb"), "sha256"
        ).hexdigest(),
        tests=[],
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["upstream", "native"]:
        command = [sys.executable, str(Path(__file__).resolve())]
        for name in ["upstream", "artifact", "checkpoints", "output_dir"]:
            command += ["--" + name.replace("_", "-"), str(getattr(args, name))]
        subprocess.run(
            command + ["--worker", kind],
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
    print("Original HumanML3D evaluator all outputs and ordering exact", flush=True)


if __name__ == "__main__":
    main()
