"""Check the native Runner checkpoint against its source-compatible GenTrack handoff."""

from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from motius.models.protomotions import ProtoMotionsPolicyBundle
    from motius.pipelines.protomotions import ProtoMotionsPipeline
    from motius import Pipeline
    from verify_prompthmr_video_parity import compare_full, tensors

    checkpoint = args.run_dir / f"checkpoint-iter_{args.expected_epoch}"
    interchange = torch.load(
        args.run_dir / "last.ckpt", weights_only=False, map_location="cpu"
    )
    native = torch.load(checkpoint / "model.pt", weights_only=False, map_location="cpu")
    assert interchange["epoch"] == args.expected_epoch
    assert interchange["step_count"] == args.expected_steps
    converted = ProtoMotionsPolicyBundle.convert_original_state(interchange["model"])
    expected = {
        role + "." + name: value
        for role, state in native.items()
        if role != "__motius_meta__"
        for name, value in state.items()
    }
    rows = []
    compare_full(expected, converted, "neural", rows)
    for index, role in enumerate(
        [
            "actor_optimizer",
            "critic_optimizer",
            "discriminator_optimizer",
            "disc_critic_optimizer",
        ]
    ):
        suffix = "" if index == 0 else "_" + str(index)
        state = torch.load(
            checkpoint / ("optimizer" + suffix + ".bin"),
            weights_only=False,
            map_location="cpu",
        )
        compare_full(tensors(state), tensors(interchange[role]), role, rows)
    pipeline = ProtoMotionsPipeline.from_original_checkpoint(args.run_dir / "last.ckpt")
    restored = Pipeline.from_pretrained(args.run_dir / "exported")
    compare_full(
        pipeline.bundle.state_dict(), restored.bundle.state_dict(), "exported", rows
    )
    report = dict(
        complete=True,
        scope="Actual native Isaac Lab CLI training and original-format resume handoff. Final source-compatible last.ckpt counters, every neural tensor and all four Adam states compared with the same Accelerate checkpoint; cold public native Pipeline has the trained neural tensors. Physical source rollout parity is certified separately.",
        epoch=args.expected_epoch,
        step_count=args.expected_steps,
        tests=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("Native/source checkpoint handoff exact:", len(rows))


if __name__ == "__main__":
    main()
