"""Compare native MotionStreamer dataset sampling with the original data readers."""

from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import random
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def fixture(directory):
    import numpy as np

    for root in [directory / "humanml3d_272", directory / "babel_272"]:
        for name in ["split", "texts", "motion_data", "mean_std", "t2m_babel_mean_std"]:
            (root / name).mkdir(parents=True, exist_ok=True)
        for name in ["mean_std", "t2m_babel_mean_std"]:
            np.save(root / name / "Mean.npy", np.arange(272, dtype=np.float32) * 0.001)
            np.save(root / name / "Std.npy", np.ones(272, dtype=np.float32) * 1.3)
    hml = directory / "humanml3d_272"
    (hml / "split/train.txt").write_text("clip_a\nclip_b\n")
    for index, name in enumerate(["clip_a", "clip_b"]):
        motion = np.sin(
            np.arange((32 + index * 8) * 272, dtype=np.float32).reshape(-1, 272) * 0.013
        )
        np.save(hml / "motion_data" / (name + ".npy"), motion)
        (hml / "texts" / (name + ".txt")).write_text(
            "a person walks#walk/VERB#0.0#0.0\na person turns#turn/VERB#0.2#0.7\n"
        )
    babel = directory / "babel_272"
    (babel / "split/train.txt").write_text("babel001\n")
    np.save(
        babel / "motion_data/babel001.npy", np.ones((24, 272), dtype=np.float32) * 0.17
    )
    latent = directory / "latents"
    latent.mkdir(exist_ok=True)
    for index, name in enumerate(["clip_a", "clip_b", "seq_case"]):
        np.save(
            latent / (name + ".npy"),
            np.sin(
                np.arange((9 + index) * 16, dtype=np.float32).reshape(-1, 16) * 0.03
            ),
        )
    stream = directory / "babel_272_stream/train_stream_text"
    stream.mkdir(parents=True, exist_ok=True)
    (stream / "seq_case.txt").write_text(
        "first action*second action#second/OTHER#4\nfirst action*another action#another/OTHER#4\n"
    )


def worker(args):
    import numpy as np
    import torch
    from verify_prompthmr_video_parity import tensors

    data = args.output_dir / "fixture"
    hml, babel, stream = [
        data / name for name in ["humanml3d_272", "babel_272", "babel_272_stream"]
    ]
    if args.worker == "upstream":
        sys.path.insert(0, str(args.upstream))
        readers = []
        for filename, name in [
            ("dataset_TM_train.py", "Text2MotionDataset"),
            ("dataset_TM_train_motionstreamer.py", "Text2MotionDataset"),
            ("dataset_tae.py", "MotionDataset"),
        ]:
            spec = importlib.util.spec_from_file_location(
                "original_data_" + filename[:-3],
                args.upstream / "humanml3d_272" / filename,
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            readers.append(getattr(module, name))
        os.chdir(data)
        datasets = [
            readers[0]("t2m_272", latent_dir=str(data / "latents")),
            readers[1]("t2m_babel_272", latent_dir=str(data / "latents")),
            readers[2]("t2m_272", window_size=16),
            readers[2]("t2m_babel_272", window_size=16),
        ]
    else:
        from motius.datasets.motionstreamer import (
            MotionStreamerTextTokenDataset,
            MotionStreamerStreamingTokenDataset,
            MotionStreamerTAEDataset,
        )

        options = dict(hml_root=hml, babel_root=babel, babel_stream_root=stream)
        datasets = [
            MotionStreamerTextTokenDataset(
                "t2m_272", latent_dir=str(data / "latents"), **options
            ),
            MotionStreamerStreamingTokenDataset(
                "t2m_babel_272", latent_dir=str(data / "latents"), **options
            ),
            MotionStreamerTAEDataset("t2m_272", window_size=16, **options),
            MotionStreamerTAEDataset("t2m_babel_272", window_size=16, **options),
        ]
    output = []
    for index, dataset in enumerate(datasets):
        rows = []
        for repeat in range(3):
            for item in range(len(dataset)):
                random.seed(712 + repeat * 11 + item)
                np.random.seed(118 + repeat * 11 + item)
                value = dataset[item]
                if args.worker == "upstream":
                    if index < 2:
                        value = dict(
                            caption=value[0],
                            tokens=value[1],
                            lengths=value[2],
                            prefix_lengths=value[3] if index == 1 else 0,
                        )
                    else:
                        value = dict(motion=value)
                rows.append(
                    tensors(
                        dict(
                            data=value,
                            python_rng=hashlib.sha256(
                                pickle.dumps(random.getstate())
                            ).hexdigest(),
                            numpy_rng=hashlib.sha256(
                                pickle.dumps(np.random.get_state())
                            ).hexdigest(),
                        )
                    )
                )
        output.append(dict(length=len(dataset), rows=rows))
    torch.save(output, args.output_dir / (args.worker + ".pt"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=["upstream", "native"])
    args = parser.parse_args()
    args.upstream, args.output_dir = args.upstream.resolve(), args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    fixture(args.output_dir / "fixture")
    report = dict(
        complete=False,
        scope="Original and native HML token, HML/BABEL streaming-token and TAE motion readers, full/subclip captions, prefix lengths, random token trimming, padding dtype, normalization and stochastic crop behavior. Three seeded passes, exact Python/NumPy RNG state; generated filesystem fixture, not released dataset coverage.",
        tests=[],
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
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ],
            env={**os.environ, "PYTHONPATH": str(ROOT)},
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
        "Original/native MotionStreamer dataset sampling exact:", len(report["tests"])
    )


if __name__ == "__main__":
    main()
