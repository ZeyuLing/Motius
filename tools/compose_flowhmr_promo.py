"""Compose synchronized input / FlowHMR / Mixamo plates into a 1080p film.

All 750 source frames are retained, with original audio. Tracking changes only
the source crop; 3D plates are rendered from the same saved prediction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter1d, maximum_filter1d


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video', type=Path, required=True)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--plates', type=Path, required=True)
    p.add_argument('--font', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint-label', default='FlowHMR')
    p.add_argument('--preview', type=int, help='Compose one zero-based frame')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    import sys
    sys.path.insert(0, str(root))
    from motius.motion.representation.monocular_capture import load_monocular_capture_result
    capture = load_monocular_capture_result(args.capture)
    track, = capture.tracks
    boxes = np.asarray(track.native_parameters['bbox_xyxy'])
    centers = gaussian_filter1d((boxes[:, 0]+boxes[:, 2])/2, 8)
    centers_y = gaussian_filter1d((boxes[:, 1]+boxes[:, 3])/2, 8)
    # A portrait reframe keeps the performer as readable as the 3D stages.
    # Anticipate extended arms with a one-second envelope; preserve the full
    # body rather than filling the plate when a pose needs a wider crop.
    crop_heights = np.clip(gaussian_filter1d(maximum_filter1d(boxes[:, 3]-boxes[:, 1], 31)+100, 8), 620, 720)
    crop_widths = np.clip(np.maximum(crop_heights*568/700,
        gaussian_filter1d(maximum_filter1d(boxes[:, 2]-boxes[:, 0], 31)+100, 8)), 1, 1280)
    color = dict(bg='#08131f', text='#f1f4f5', muted='#8d9daa', line='#263440', aqua='#85e5d3')
    def font(size, weight=450):
        f = ImageFont.truetype(str(args.font), size)
        f.set_variation_by_axes([weight])
        return f
    base = Image.new('RGB', (1920, 1080), color['bg'])
    d = ImageDraw.Draw(base)
    d.text((64, 37), 'motius', font=font(33, 650), fill=color['text'])
    d.text((209, 49), '/  MOTION, CONNECTED.', font=font(15, 500), fill=color['muted'])
    d.text((1856, 49), 'FLOWHMR × BLENDER', anchor='ra', font=font(16, 500), fill=color['aqua'])
    d.text((60, 93), 'From video to character motion.', font=font(78, 480), fill=color['text'])
    d.line((64, 197, 1856, 197), fill=color['line'], width=1)
    positions = (64, 676, 1288)
    labels = (('01', 'Video', 'SOURCE'), ('02', 'FlowHMR output', 'SMPL-H'), ('03', 'Mixamo', 'RETARGETED'))
    for x, (num, title, tag) in zip(positions, labels):
        d.text((x, 219), num, font=font(17, 500), fill=color['aqua'])
        d.text((x+40, 213), title, font=font(29, 500), fill=color['text'])
        d.text((x+568, 224), tag, anchor='ra', font=font(13, 500), fill=color['muted'])
    for x in (650, 1262):
        d.line((x-8, 231, x+8, 231), fill=color['aqua'], width=2)
        d.line((x+2, 225, x+8, 231, x+2, 237), fill=color['aqua'], width=2)
    d.text((64, 994), 'One performance. Three synchronized views.', font=font(24, 450), fill=color['text'])
    d.text((1856, 1000), f'{args.checkpoint_label}  /  Blender  /  Mixamo', anchor='ra', font=font(17), fill=color['muted'])
    d.line((64, 1052, 1720, 1052), fill=color['line'], width=2)
    mask = Image.new('L', (568, 700))
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, 567, 699), radius=14, fill=255)
    cap = cv2.VideoCapture(str(args.video))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == track.num_frames == 750
    assert abs(cap.get(cv2.CAP_PROP_FPS)-30) < 1e-4
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.preview is None:
        writer = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '1920x1080', '-r', '30', '-i', '-', '-i', str(args.video),
            '-map', '0:v:0', '-map', '1:a:0?', '-c:v', 'libx264', '-preset', 'slow', '-crf', '17',
            '-vf', 'scale=out_color_matrix=bt709:out_range=tv',
            '-pix_fmt', 'yuv420p', '-colorspace', 'bt709', '-color_primaries', 'bt709',
            '-color_trc', 'bt709', '-color_range', 'tv',
            '-c:a', 'copy', '-t', '25', '-movflags', '+faststart', str(args.output)], stdin=subprocess.PIPE)
    indices = range(750) if args.preview is None else [args.preview]
    for i in indices:
        if args.preview is not None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        assert ok, i
        source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        crop_width, crop_height = int(round(crop_widths[i])), int(round(crop_heights[i]))
        left = int(np.clip(centers[i]-crop_width/2, 0, source.width-crop_width))
        top = int(np.clip(centers_y[i]-crop_height/2, 0, source.height-crop_height))
        source = source.crop((left, top, left+crop_width, top+crop_height))
        source.thumbnail((568, 700), Image.Resampling.LANCZOS)
        input_plate = Image.new('RGB', (568, 700), '#000000')
        input_plate.paste(source, ((568-source.width)//2, (700-source.height)//2))
        image = base.copy()
        for x, stage in zip(positions, ('video', 'flowhmr', 'mixamo')):
            plate = input_plate if stage == 'video' else Image.open(args.plates/stage/f'{i+1:04d}.png').convert('RGB').resize((568, 700), Image.Resampling.LANCZOS)
            image.paste(plate, (x, 266), mask)
        d = ImageDraw.Draw(image)
        d.line((64, 1052, 64+round(1656*i/749), 1052), fill=color['aqua'], width=2)
        d.text((1856, 1040), f'{i/30:05.2f} / 25.00', anchor='ra', font=font(15), fill=color['muted'])
        if writer:
            writer.stdin.write(image.tobytes())
        else:
            image.save(args.output)
        if i % 150 == 0:
            print('COMPOSED', i, flush=True)
    cap.release()
    if writer:
        writer.stdin.close()
        assert writer.wait() == 0
        provenance = dict(frames=750, fps=30, seconds=25, resolution=[1920, 1080],
                          source_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                          capture_sha256=hashlib.sha256(args.capture.read_bytes()).hexdigest(),
                          checkpoint_sha256=capture.checkpoint_sha256,
                          checkpoint_label=args.checkpoint_label,
                          video_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                          original_audio_stream_copy=True,
                          source_crop=dict(method='Smoothed full-body portrait reframe; letterbox wide poses',
                                           width_range=[float(crop_widths.min()), float(crop_widths.max())],
                                           height_range=[float(crop_heights.min()), float(crop_heights.max())],
                                           envelope_frames=31, gaussian_sigma_frames=8),
                          flowhmr_inference=capture.metadata.get('inference_settings'),
                          character_credit='Mixamo character from the three.js Samba Dancing FBX example',
                          character_source='https://github.com/mrdoob/three.js/blob/dev/examples/models/fbx/Samba%20Dancing.fbx')
        args.output.with_suffix('.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    main()
