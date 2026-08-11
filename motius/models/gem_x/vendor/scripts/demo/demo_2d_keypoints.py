# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E402, I001
"""
Standalone 2D keypoint extraction demo.

Pipeline: human detection -> VitPose 77-joint extraction -> overlay video.
No GEM model, no 3D rendering, no SAM3D, no Hydra.
"""
import argparse
import os

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import sys
import functools
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# PyTorch >=2.6 defaults weights_only=True, but our saved files contain numpy arrays.
torch.load = functools.partial(torch.load, weights_only=False)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gem.utils.geo_transform import get_bbx_xys_from_xyxy
from gem.utils.kp2d_utils import render_2d_keypoints, smooth_bbx_xyxy
from gem.utils.video_io_utils import get_video_lwh, read_video_np
from gem.utils.vitpose_extractor import VitPoseExtractor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect humans, extract VitPose 2D keypoints, and render an overlay video."
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/demo_2d_kp/<video_name>/)",
    )
    parser.add_argument(
        "--detector_name",
        type=str,
        default="vitdet",
        help="Human detector: 'vitdet' or 'sam3' (default: vitdet)",
    )
    parser.add_argument(
        "--conf_thr",
        type=float,
        default=0.5,
        help="Confidence threshold for keypoint visualization (default: 0.5)",
    )
    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Keep raw bbx.pt and vitpose.pt (they are saved during processing regardless)",
    )
    return parser.parse_args()


@torch.no_grad()
def run_detection(video_path, output_dir, detector_name):
    """Run per-frame human detection, pick largest box, smooth, and save bbx.pt."""
    bbx_path = Path(output_dir) / "bbx.pt"
    if bbx_path.exists():
        print(f"[Detection] Found cached {bbx_path}, skipping.")
        return bbx_path

    # Add sam-3d-body to path for HumanDetector
    sam3db_root = str(PROJECT_ROOT / "third_party" / "sam-3d-body")
    if sam3db_root not in sys.path:
        sys.path.insert(0, sam3db_root)
    from tools.build_detector import HumanDetector

    L, W, H = get_video_lwh(video_path)
    print(f"[Detection] Running '{detector_name}' on {L} frames ({W}x{H})...")
    detector = HumanDetector(name=detector_name, device="cuda")

    frames = read_video_np(video_path)
    all_boxes = []
    for i in tqdm(range(len(frames)), desc="Detect Humans"):
        # HumanDetector expects BGR; read_video_np returns RGB
        img_bgr = frames[i][..., ::-1].copy()
        boxes = detector.run_human_detection(img_bgr)  # (N, 4) xyxy
        if len(boxes) == 0:
            all_boxes.append(np.array([0.0, 0.0, W - 1, H - 1]))
        else:
            # Pick the largest-area detection (single-person assumption)
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            all_boxes.append(boxes[areas.argmax()])

    bbx_xyxy = torch.from_numpy(np.stack(all_boxes, axis=0)).float()  # (L, 4)
    bbx_xyxy = smooth_bbx_xyxy(bbx_xyxy, window=5)
    # Clamp to image bounds
    bbx_xyxy[:, [0, 2]] = bbx_xyxy[:, [0, 2]].clamp(0, W - 1)
    bbx_xyxy[:, [1, 3]] = bbx_xyxy[:, [1, 3]].clamp(0, H - 1)
    bbx_xys = get_bbx_xys_from_xyxy(bbx_xyxy, base_enlarge=1.2).float()

    torch.save({"bbx_xyxy": bbx_xyxy, "bbx_xys": bbx_xys}, bbx_path)
    print(f"[Detection] Saved {L} bboxes to {bbx_path}")
    return bbx_path


@torch.no_grad()
def run_vitpose(video_path, bbx_path, output_dir):
    """Extract 77-joint VitPose keypoints and save vitpose.pt."""
    vitpose_path = Path(output_dir) / "vitpose.pt"
    if vitpose_path.exists():
        print(f"[VitPose] Found cached {vitpose_path}, skipping.")
        return vitpose_path

    print("[VitPose] Extracting 77-joint keypoints...")
    bbx_data = torch.load(bbx_path)
    bbx_xys = bbx_data["bbx_xys"]

    extractor = VitPoseExtractor(device="cuda:0", pose_type="soma")
    frames = read_video_np(video_path)
    vitpose = extractor.extract(frames, bbx_xys, img_ds=1.0, path_type="np")

    torch.save(vitpose, vitpose_path)
    print(f"[VitPose] Saved keypoints {tuple(vitpose.shape)} to {vitpose_path}")
    return vitpose_path


def main():
    args = parse_args()

    video_path = Path(args.video)
    assert video_path.exists(), f"Video not found: {video_path}"

    # Resolve output directory
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("outputs") / "demo_2d_kp" / video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Setup] Video : {video_path}")
    print(f"[Setup] Output: {output_dir}")

    # Step 1: Human detection
    bbx_path = run_detection(str(video_path), str(output_dir), args.detector_name)

    # Step 2: VitPose extraction
    vitpose_path = run_vitpose(str(video_path), str(bbx_path), str(output_dir))

    # Step 3: Render overlay video
    overlay_name = f"{video_path.stem}_kp2d77_overlay.mp4"
    overlay_path = output_dir / overlay_name
    print(f"[Render] Creating overlay video at {overlay_path}...")
    render_2d_keypoints(
        video_path=str(video_path),
        vitpose_path=str(vitpose_path),
        bbx_path=str(bbx_path),
        output_path=str(overlay_path),
        fps=30,
        conf_thr=args.conf_thr,
    )
    print(f"[Render] Saved overlay to {overlay_path}")

    # Step 4: Optionally clean up raw files
    if not args.save_raw:
        print("[Cleanup] Removing intermediate .pt files (use --save_raw to keep them).")
        Path(bbx_path).unlink(missing_ok=True)
        Path(vitpose_path).unlink(missing_ok=True)

    print("[Done] Finished 2D keypoint demo.")


if __name__ == "__main__":
    main()
