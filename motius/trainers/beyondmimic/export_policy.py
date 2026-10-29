"""Export a native actor/reference artifact without an original application."""

import argparse, json
from pathlib import Path
import numpy as np
from motius.models.beyondmimic import BeyondMimicPolicyBundle
from motius.pipelines.beyondmimic import BeyondMimicPipeline


def export_policy(
    checkpoint,
    output_dir,
    *,
    reference_motion=None,
    metadata=None,
    device='cpu',
    onnx=False,
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    bundle = (
        BeyondMimicPolicyBundle.from_pretrained(checkpoint, device=device)
        if checkpoint.is_dir()
        else BeyondMimicPolicyBundle.from_original_checkpoint(
            checkpoint, device=device
        ).eval()
    )
    pipeline = BeyondMimicPipeline(
        bundle, reference_motion=reference_motion, metadata=metadata
    )
    pipeline.save_pretrained(output)
    if (
        metadata is None
        and checkpoint.is_dir()
        and (checkpoint / 'tracking_metadata.json').is_file()
    ):
        metadata = pipeline.metadata
    if metadata is not None:
        (output / 'tracking_metadata.json').write_text(
            json.dumps(metadata, indent=2) + '\n'
        )
    if onnx:
        if pipeline.reference_motion is None:
            raise ValueError('ONNX deployment export requires reference_motion')
        from motius.pipelines.beyondmimic.export import export_onnx_policy

        export_onnx_policy(
            bundle, pipeline.reference_motion, output / 'policy.onnx', metadata=metadata
        )
    return output


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--motion_file', '--reference-motion', dest='reference_motion', type=Path
    )
    parser.add_argument('--metadata', type=Path)
    parser.add_argument('--onnx', action='store_true')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args(arguments)
    metadata = None if args.metadata is None else json.loads(args.metadata.read_text())
    return export_policy(
        args.checkpoint,
        args.output_dir,
        reference_motion=args.reference_motion,
        metadata=metadata,
        device=args.device,
        onnx=args.onnx,
    )


if __name__ == '__main__':
    main()
