"""Convert local motion CSV to the native reference NPZ schema."""

import argparse, os
from pathlib import Path


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    parser.add_argument('--input_fps', type=int, default=30)
    parser.add_argument('--output_fps', type=int, default=50)
    parser.add_argument('--frame_range', nargs=2, type=int)
    parser.add_argument(
        '--asset-root',
        type=Path,
        default=Path(
            os.environ.get('MOTIUS_BEYONDMIMIC_ASSET_ROOT', 'checkpoints/robots')
        ),
    )
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args(arguments)
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=args.headless, device=args.device)
    try:
        from motius.simulators.isaaclab.beyondmimic.prepare import prepare_csv_motion

        return prepare_csv_motion(
            args.input_file,
            args.output_file,
            args.asset_root,
            input_fps=args.input_fps,
            output_fps=args.output_fps,
            device=args.device,
            frame_range=args.frame_range,
        )
    finally:
        from isaaclab.sim import SimulationContext

        SimulationContext.clear_instance()
        launcher.app.close()


if __name__ == '__main__':
    main()
