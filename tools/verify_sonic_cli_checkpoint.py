"""Accept trained/resumed SONIC states and a cold export from the shared checkpoint."""

from pathlib import Path
import argparse
import json
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(args):
    import torch
    from motius import Pipeline
    from motius.trainers.sonic.train import export_policy
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = args.training_dir / "checkpoint-iter_3"
    if args.worker == "export":
        meta = torch.load(checkpoint / "meta.pt", weights_only=False)
        assert meta["global_step"] == 3
        for iteration in [1, 2, 3]:
            assert (
                args.training_dir / f"checkpoint-iter_{iteration}" / "optimizer.bin"
            ).is_file()
        path = args.training_dir / "policy"
    else:
        path = args.output_dir / "exported"
    pipe = Pipeline.from_pretrained(str(path), bundle_kwargs={"device": "cuda"})
    if args.worker == "export":
        saved = torch.load(
            checkpoint / "model.pt", map_location="cpu", weights_only=True
        )
        for root in ["policy", "critic"]:
            actual = getattr(pipe.bundle, root).state_dict()
            assert actual.keys() == saved[root].keys()
            for name, value in actual.items():
                torch.testing.assert_close(
                    value.cpu(), saved[root][name], rtol=0, atol=0
                )
        export_policy(checkpoint, args.output_dir / "exported")
    torch.manual_seed(124)
    encoder = torch.randn(3, 1762, device="cuda") * 0.12
    encoder[:, 0] = torch.arange(3, device="cuda")
    proprioception = torch.randn(3, 930, device="cuda") * 0.12
    output = pipe(encoder, proprioception)
    torch.save(
        tensors(dict(neural=hashes(pipe.bundle), output=output)),
        args.output_dir / (args.worker + ".pt"),
    )
    assert not any(
        name == "gear_sonic" or name.startswith("gear_sonic.") for name in sys.modules
    )
    assert not any(
        name == "isaaclab" or name.startswith("isaaclab.") for name in sys.modules
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--worker", choices=["export", "cold"])
    args = parser.parse_args()
    args.training_dir = args.training_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    import torch
    from verify_prompthmr_video_parity import compare_full

    report = dict(
        complete=False,
        tests=[],
        scope="Actual public native SONIC CLI, "
        "four Isaac Lab G1 environments, two rollout iterations then full "
        "shared-checkpoint resume to iteration three. Checkpoint neural "
        "tensors equal the current policy artifact. Export directly from "
        "the shared checkpoint and cold offline Auto Pipeline reload: "
        "all neural hashes and three-mode public inference exact. "
        "Training integration/export evidence, not original-source parity "
        "or full-dataset convergence.",
    )
    path = args.output_dir / "source_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    for kind in ["export", "cold"]:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--training-dir",
                str(args.training_dir),
                "--output-dir",
                str(args.output_dir),
                "--worker",
                kind,
            ],
            cwd=ROOT,
            env={**os.environ, "HF_HUB_OFFLINE": "1"},
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
    print("SONIC real physical CLI resume and cold checkpoint export accepted")


if __name__ == "__main__":
    main()
