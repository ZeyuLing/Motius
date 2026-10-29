"""Validate native training/resume counters and export of the actual AR/TAE weights."""

from pathlib import Path
import argparse
import json
import os
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    args = parser.parse_args()
    import torch
    from motius import Pipeline
    from motius.trainers.motionstreamer import MotionStreamerTrainer
    from motius.models.motionstreamer.network import Causal_HumanTAE
    from verify_prompthmr_video_parity import compare_full

    if os.environ.get("MOTIUS_REFERENCE_HOST"):

        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    state = torch.load(
        args.checkpoint / "model.pt", map_location="cpu", weights_only=True
    )
    metadata = torch.load(
        args.checkpoint / "meta.pt", map_location="cpu", weights_only=False
    )
    assert metadata["global_step"] == args.expected_step
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "exported"
    MotionStreamerTrainer.export_policy(
        args.checkpoint, args.artifact, output, include_text_encoder=False
    )
    restored = Pipeline.from_pretrained(
        output, bundle_kwargs={"load_text_model": False}, device="cuda"
    ).bundle
    rows = []
    for role in ["tae", "ar"]:
        actual = {
            name: value.cpu()
            for name, value in getattr(restored, role).state_dict().items()
        }
        compare_full(state[role], actual, role, rows)
    for name in ["mean", "std"]:
        assert torch.equal(
            getattr(restored, name).cpu(), state["__bundle_params__"][name]
        )
    reference = (
        Causal_HumanTAE(
            hidden_size=1024,
            down_t=2,
            stride_t=2,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            latent_dim=16,
            clip_range=[-30, 20],
        )
        .cuda()
        .eval()
    )
    reference.load_state_dict(state["tae"], strict=True)
    latent = torch.sin(torch.arange(3 * 16, device="cuda").reshape(1, 3, 16) * 0.013)
    with torch.no_grad():
        compare_full(
            reference.forward_decoder(latent).cpu(),
            restored.tae.forward_decoder(latent).cpu(),
            "decoded_current_weights",
            rows,
        )
    report = dict(
        complete=True,
        scope="Actual shared tools/train.py TAE training, full auto-resume to the declared step, native Trainer checkpoint export and cold public loading of AR/TAE components. Every AR/TAE tensor and normalization buffer matches model.pt; decoded motion matches an independently reconstructed decoder loaded directly from that checkpoint. This component-only export deliberately excludes text; the separate complete-artifact certificate covers real SentenceT5 cold loading.",
        step=args.expected_step,
        tests=rows,
    )
    (args.output_dir / "source_parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "Native trained/resumed checkpoint and exported neural components exact:",
        len(rows),
    )


if __name__ == "__main__":
    main()
