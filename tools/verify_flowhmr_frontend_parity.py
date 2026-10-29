"""Run native and upstream video frontends in isolated processes and compare exactly."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def worker(args):
    root = Path(__file__).resolve().parents[1]
    torch.set_num_threads(4)
    torch.manual_seed(20261021)
    device = torch.device('cuda')
    assets = args.assets
    if args.worker == 'upstream':
        for directory in (args.upstream, args.upstream / 'third_party/YOLOX',
                          args.upstream / 'third_party/sam-3d-body',
                          args.upstream / 'third_party/vggt-omega'):
            sys.path.insert(0, str(directory))
        from flowhmr.utils.runtime_video_processing import YOLOXHumanDetector, VideoProcessor
        from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera
        from flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor, _maybe_convert_backbone_to_fp16
        from sam_3d_body import load_sam_3d_body, SAM3DBodyEstimator
        detector = YOLOXHumanDetector(str(assets / 'yolox_l.pth'), device)
        boxes = VideoProcessor(detector).compute_bboxes(str(args.video))
        del detector
        gc.collect(); torch.cuda.empty_cache()
        camera = extract_vggt_camera(str(args.video), frame_range=(0, len(boxes)-1),
            expected_len=len(boxes), device=device, checkpoint_path=str(assets / 'vggt_omega_1b_512.pt'))
        model, cfg = load_sam_3d_body(checkpoint_path=str(assets / 'sam/model.ckpt'),
                                   mhr_path=str(assets / 'sam/assets/mhr_model.pt'), device=device)
        _maybe_convert_backbone_to_fp16(model, device)
        extractor = Sam3DTokenExtractor(SAM3DBodyEstimator(model, cfg))
        features = extractor.extract_video_tokens(str(args.video), bbox_xyxy=boxes,
            K_all=torch.as_tensor(camera.K), token_dim=3072, max_frames=len(boxes))
    else:
        sys.path.insert(0, str(root))
        from motius.pipelines.flowhmr.preprocess import FlowHMRVideoPreprocessor
        from motius.pipelines.flowhmr.camera_estimation import extract_vggt_camera
        pre = FlowHMRVideoPreprocessor(device='cuda', sam3d_ckpt=str(assets / 'sam/model.ckpt'),
            sam3d_mhr=str(assets / 'sam/assets/mhr_model.pt'),
            sam3d_config=str(assets / 'sam/model_config.yaml'), yolox_ckpt=str(assets / 'yolox_l.pth'))
        boxes, _ = pre.detect_and_track(str(args.video))
        pre._detector = None
        gc.collect(); torch.cuda.empty_cache()
        camera = extract_vggt_camera(str(args.video), frame_range=(0, len(boxes)-1),
            expected_len=len(boxes), device=device, checkpoint_path=str(assets / 'vggt_omega_1b_512.pt'))
        features = pre.extract_features(str(args.video), boxes, torch.as_tensor(camera.K),
                                        token_dim=3072, max_frames=len(boxes))
    output = dict(bboxes=torch.as_tensor(boxes).cpu(), camera_RT=torch.as_tensor(camera.camera_RT).cpu(),
                  camera_K=torch.as_tensor(camera.K).cpu(), features=torch.as_tensor(features).cpu())
    torch.save(output, args.output_dir / f'{args.worker}.pt')
    print(args.worker, {k: list(v.shape) for k, v in output.items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--worker', choices=('native', 'upstream'))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return
    (args.output_dir / 'frontend_parity.json').write_text(json.dumps(dict(complete=False)) + '\n')
    for mode in ('upstream', 'native'):
        command = [sys.executable, str(Path(__file__).resolve()), '--upstream', str(args.upstream),
                   '--video', str(args.video), '--assets', str(args.assets),
                   '--output-dir', str(args.output_dir), '--worker', mode]
        subprocess.run(command, check=True)
    source = torch.load(args.output_dir / 'upstream.pt', weights_only=True)
    native = torch.load(args.output_dir / 'native.pt', weights_only=True)
    for name in source:
        torch.testing.assert_close(source[name], native[name], rtol=0, atol=0)
    report = dict(complete=True, atol=0, rtol=0, frames=len(source['features']), exact_keys=list(source),
                  input_sha256=sha256(args.video), torch_version=torch.__version__,
                  weights_sha256={str(p.relative_to(args.assets)): sha256(p) for p in (
                      args.assets / 'yolox_l.pth', args.assets / 'vggt_omega_1b_512.pt',
                      args.assets / 'sam/model.ckpt', args.assets / 'sam/assets/mhr_model.pt')})
    (args.output_dir / 'frontend_parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Independent upstream/native frontend comparison passed:', report, flush=True)


if __name__ == '__main__':
    main()
