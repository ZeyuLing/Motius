"""Check the frozen runtime and ABI before a job is allowed to execute."""
import argparse
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', default='/opt/motius-runtime-manifest.json')
    parser.add_argument('--build-check', action='store_true')
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text())
    if platform.machine() != manifest['architecture'] or platform.python_version() != manifest['python']:
        raise RuntimeError('Runtime overlay requires Linux x86_64 and Python 3.12')
    mismatch = {}
    for name, expected in manifest['distributions'].items():
        try: actual = metadata.version(name)
        except metadata.PackageNotFoundError: actual = None
        if actual != expected: mismatch[name] = {'expected':expected,'actual':actual}
    if mismatch:
        raise RuntimeError(f'Runtime packages do not match the captured environment: {mismatch}')
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RuntimeError("FFmpeg and FFprobe are required")
    import torch
    import torchvision
    import cv2
    import roma
    import mmengine
    from motius.pipelines.flowhmr import FlowHMRPipeline
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam.inference import run_official_frame_batch, run_fast_frame_batch
    from motius.pipelines.flowhmr.tracking import YOLOXHumanDetector
    from motius.models.vision.sam3d_body.bundle import SAM3DBodyBundle
    from motius.models.vision.vggt_omega.bundle import VGGTOmegaBundle
    from motius.models.flowhmr.network.hymotion_mmdit_for_v2m import HunyuanMotionMMDiT
    # Verify the PyTorch/torchvision compiled ABI, including CPU NMS.
    boxes = torch.tensor([[0.,0.,2.,2.],[.1,.1,1.9,1.9]])
    assert torchvision.ops.nms(boxes,torch.tensor([.9,.8]),.5).tolist() == [0]
    print(json.dumps({'python':platform.python_version(),'torch':torch.__version__,
                      'torchvision':torchvision.__version__, 'opencv':cv2.__version__,
                      'cuda_available':torch.cuda.is_available(), 'pipeline':FlowHMRPipeline.__name__}))


if __name__ == '__main__':
    main()
