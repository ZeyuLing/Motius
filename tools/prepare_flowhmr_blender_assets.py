"""Export a saved FlowHMR capture to the Blender motion protocol and mesh cache.

Run in the Motius Python environment, before rendering with Blender.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motius.motion.representation.monocular_capture import load_monocular_capture_result
from motius.motion.body_models.flowhmr import SMPLMesh


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--body-model', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(1)
    capture = load_monocular_capture_result(args.capture)
    if capture.source_model != 'FlowHMR' or len(capture.tracks) != 1:
        raise ValueError('Expected a single-person FlowHMR capture')
    track, = capture.tracks
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT/'blender_extension/motius_flowhmr'))
    from worker import export_capture
    export_capture(capture, args.body_model, args.output_dir/'motion.blender.npz')
    model = SMPLMesh(model_path=str(args.body_model)).eval()
    vertices = []
    with torch.inference_mode():
        for start in range(0, track.num_frames, 16):
            end = min(start+16, track.num_frames)
            shapes = track.shape_parameters
            parameters = {
                'rot6d': torch.from_numpy(track.native_parameters['rot6d'][start:end]),
                'trans': torch.from_numpy(track.native_parameters['trans'][start:end]),
                'shapes': torch.from_numpy(shapes if len(shapes) == 1 else shapes[start:end]),
            }
            vertices.append(model(parameters)['vertices'].numpy())
    np.savez_compressed(args.output_dir/'mesh.npz', vertices=np.concatenate(vertices),
                        faces=np.asarray(model.faces, dtype=np.int32))
    print(f'Exported {track.num_frames} SMPL-H meshes and 52-joint poses')


if __name__ == '__main__':
    main()
