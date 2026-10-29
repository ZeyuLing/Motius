"""Check installed-package imports and real inference without the source tree."""

from pathlib import Path
import argparse, importlib, json, os, sys, zipfile


def equal(reference, value, path=""):
    import numpy as np
    import torch

    if isinstance(reference, dict):
        assert reference.keys() == value.keys(), path
        for name in reference:
            equal(reference[name], value[name], path + "/" + name)
    elif isinstance(reference, (tuple, list)):
        assert len(reference) == len(value), path
        for i, (left, right) in enumerate(zip(reference, value)):
            equal(left, right, path + "/" + str(i))
    elif isinstance(reference, torch.Tensor):
        torch.testing.assert_close(
            reference.cpu(), torch.as_tensor(value).cpu(), rtol=0, atol=0, msg=path
        )
    elif isinstance(reference, np.ndarray):
        assert np.array_equal(reference, value), path
    else:
        assert reference == value, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--install-dir", type=Path, required=True)
    parser.add_argument("--method-audit", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.install_dir.resolve()
    assert not target.exists(), target
    target.mkdir(parents=True)
    with zipfile.ZipFile(args.wheel) as archive:
        archive.extractall(target)
    sys.path.insert(0, str(target))
    os.chdir(target.parent)
    import torch
    import numpy as np

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    report = dict(
        complete=False,
        scope="Fresh wheel extracted outside the checkout, isolated Python -I imports of every audited neural package, then real matching/TM2D/HYMotion current-checkpoint inference versus previously certified cold-process outputs. No source tree on sys.path; loaded Motius file origins must all belong to this wheel.",
        methods=[],
        inference=[],
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    try:
        for item in json.loads(args.method_audit.read_text())["methods"]:
            name = item["method"]
            module = importlib.import_module("motius.models." + name)
            assert Path(module.__file__).resolve().is_relative_to(target), name
            report["methods"].append(name)
        print(
            "All",
            len(report["methods"]),
            "neural packages imported from installed wheel",
            flush=True,
        )
        from motius.pipelines.motionstreamer_matching import (
            MotionStreamerMatchingPipeline,
        )

        root = args.validation_dir / "motionstreamer_evaluator_checkpoint"
        reference = torch.load(root / "cold.pt", map_location="cpu", weights_only=False)
        pipe = MotionStreamerMatchingPipeline.from_pretrained(
            str(root / "exported"), bundle_kwargs={"device": "cuda"}
        )
        texts = ["A person walks.", "", "Someone turns and waves."]
        motions = [
            np.random.default_rng(81 + i).normal(size=(length, 272)).astype(np.float64)
            * 0.1
            for i, length in enumerate([20, 72, 300])
        ]
        equal(reference["text"], pipe.encode_texts(texts), "matching/text")
        equal(reference["motion"], pipe.encode_motions(motions), "matching/motion")
        report["inference"].append(
            "MotionStreamer matching: raw text and ragged Float64 motions exact"
        )
        del pipe
        torch.cuda.empty_cache()
        from motius import Pipeline

        root = args.validation_dir / "tm2d_trained_current"
        reference = torch.load(root / "cold.pt", map_location="cpu", weights_only=False)
        pipe = Pipeline.from_pretrained(
            str(root / "exported"), bundle_kwargs={"device": "cuda"}
        )
        text = pipe.infer_text_to_motion(
            "A person walks forward.",
            num_frames=32,
            output_fps=60.0,
            sample=True,
            seed=81,
        )
        equal(reference["text"], vars(text), "tm2d/text")
        report["inference"].append(
            "TM2D trained joint checkpoint: complete raw-caption motion/joints/tokens exact"
        )
        del pipe
        torch.cuda.empty_cache()
        root = args.validation_dir / "hymotion_t2m_trained_current"
        reference = torch.load(root / "cold.pt", map_location="cpu", weights_only=False)
        pipe = Pipeline.from_pretrained(
            str(root / "exported"), bundle_kwargs={"device": "cuda"}
        )
        result = pipe.infer_text_to_motion(
            ["A person walks forward.", "A person turns and raises both arms."],
            num_frames=[24, 20],
            seed_input=[81, 923],
            use_special_game_feat=True,
        )
        equal(reference["motion"], result, "hymotion/motion")
        report["inference"].append(
            "HYMotion trained/current full text checkpoint: 50-step ragged special-condition motion and packaged wooden geometry exact"
        )
        for name, module in list(sys.modules.items()):
            path = getattr(module, "__file__", None)
            if name == "motius" or name.startswith("motius."):
                assert path is None or Path(path).resolve().is_relative_to(target), (
                    name,
                    path,
                )
        report["complete"] = True
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Installed wheel inference and import origins exact", flush=True)


if __name__ == "__main__":
    main()
