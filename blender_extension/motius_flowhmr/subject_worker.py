# SPDX-License-Identifier: GPL-3.0-or-later
"""Detect all people before motion inference. Runs outside Blender."""
from pathlib import Path
import json


def run(request, status):
    import cv2
    import torch
    from subjects import digest
    from worker import write_json
    from motius.pipelines.flowhmr.tracking import VideoProcessor, YOLOXHumanDetector
    from huggingface_hub import hf_hub_download
    torch.set_num_threads(4)
    output = Path(request['output_dir'])
    checkpoint = Path(request['checkpoint'])/'preprocessor/yolox_l.pth'
    if not checkpoint.is_file():
        checkpoint = Path(hf_hub_download(request['checkpoint'], 'preprocessor/yolox_l.pth', cache_dir=request.get('cache_dir') or None))
    status('running', 'Loading person detector', phase='Detect people')
    detector = YOLOXHumanDetector(str(checkpoint), torch.device(request['device']))
    processor = VideoProcessor(detector)
    used = processor.transcode(request['video'], str(output/'video'), force_fps=30)
    previews = output/'previews'; previews.mkdir(exist_ok=True)
    def progress(frame, total, image):
        cv2.imwrite(str(previews/f'{frame:06d}.jpg'), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if frame % 10 == 0:
            status('running', f'Detecting people · {frame+1} / {total} frames', phase='Detect people', current=frame+1, total=total)
    result = processor.detect_and_track(used, progress=progress)
    data = dict(schema='motius.flowhmr.subjects.v1', video_sha256=digest(request['video']),
                frames=result.total_frames, width=result.width, height=result.height,
                video=used, preview_sequence=str(previews/'000000.jpg'), tracks=[dict(id=t.track_id, frames=t.frames.tolist(), boxes=t.bboxes.tolist()) for t in result.tracks])
    path = output/'subjects.json'; write_json(path, data)
    status('completed', f'{len(data["tracks"])} trajectories found · click your subject', subjects_file=str(path))
