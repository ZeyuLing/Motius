"""Export the actual trained MotionStreamer neural state from a native checkpoint."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--artifact",
        type=Path,
        required=True,
        help="Architecture and frozen-component reference artifact",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-text",
        action="store_true",
        help="Export only AR/TAE components; raw-text inference then requires an explicit text encoder",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    from motius.trainers.motionstreamer import MotionStreamerTrainer

    print(
        MotionStreamerTrainer.export_policy(
            args.checkpoint,
            args.artifact,
            args.output,
            include_text_encoder=not args.exclude_text,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()
