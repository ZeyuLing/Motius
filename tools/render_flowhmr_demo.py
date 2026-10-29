"""Render FlowHMR with Motius Three.js mesh shading and tracked source video.

Requires the flowhmr and flowhmr-render extras and a Playwright Chromium install.
Mesh positions come directly from the saved SMPL-H parameters; only the camera
and smooth vertex normals affect presentation. No upstream checkout is used.
"""

import argparse
import base64
from functools import partial
import hashlib
import http.server
from io import BytesIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from motius.motion.representation.monocular_capture import load_monocular_capture_result
from motius.motion.body_models.flowhmr.smpl_skeleton import SMPLMesh, SMPLSkeleton

WIDTH, HEIGHT = 1600, 800
STAGE_WIDTH, STAGE_HEIGHT = 500, 585
INPUT_X, INPUT_Y, INPUT_WIDTH = 20, 112, 1040
STAGE_X = 1080


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


def _font(size, bold=False):
    name = 'DejaVuSans-Bold.ttf' if bold else 'DejaVuSans.ttf'
    for folder in ('/usr/share/fonts/truetype/dejavu', '/usr/share/fonts/dejavu'):
        path = Path(folder) / name
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.truetype(name, size)


def reconstruct_meshes(track, model_path, device, frame_count, batch_size=16):
    """Use the same bundled SMPL-H implementation as FlowHMR inference."""
    rot6d = np.asarray(track.native_parameters["rot6d"], dtype=np.float32)
    trans = np.asarray(track.native_parameters["trans"], dtype=np.float32)
    shapes = np.asarray(track.shape_parameters, dtype=np.float32)
    if rot6d.shape != (track.num_frames, 52, 6) or trans.shape != (track.num_frames, 3):
        raise ValueError("Expected FlowHMR (T,52,6) rotations and (T,3) translation")
    if shapes.shape not in ((1, 16), (track.num_frames, 16)):
        raise ValueError("Expected one or T SMPL-H shape vectors of length 16")
    mesh_model = SMPLMesh(model_path=str(model_path)).to(device).eval()
    faces = np.asarray(mesh_model.faces, dtype=np.int32)
    vertices = np.empty((frame_count, mesh_model.v_template.shape[0], 3), np.float32)
    with torch.inference_mode():
        for start in range(0, frame_count, batch_size):
            end = min(start + batch_size, frame_count)
            params = {
                "rot6d": torch.from_numpy(rot6d[start:end]).to(device),
                "trans": torch.from_numpy(trans[start:end]).to(device),
                "shapes": torch.from_numpy(shapes if len(shapes) == 1 else shapes[start:end]).to(device),
            }
            vertices[start:end] = mesh_model(params)["vertices"].cpu().numpy()
            if start == 0 and track.joints_world is not None:
                skeleton = SMPLSkeleton(model_path=str(model_path)).to(device).eval()
                actual = skeleton(params)["keypoints3d"][:, :22].cpu().numpy()
                difference = float(np.max(np.abs(actual - track.joints_world[start:end, :22])))
                if difference > 1e-3:
                    raise ValueError(f"Reconstructed SMPL-H joints differ from capture: {difference:.6f} m")
                del skeleton
                print(f"SMPL-H joint agreement: max error {difference:.7f} m", flush=True)
            if end % 96 == 0 or end == frame_count:
                print(f"Reconstructed {end}/{frame_count} meshes", flush=True)
    return vertices, faces


def prepare_scene(directory, vertices, faces, roots, reference=None):
    centered = vertices.copy()
    centered[:, :, 0] -= roots[:, None, 0]
    centered[:, :, 2] -= roots[:, None, 2]
    # Fit the entire sequence, including jumps and extended arms, once. This
    # avoids both clipped heads and camera zoom changing during the animation.
    backward = np.array([0.25, 0.15, -1.0])
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 1.0, 0.0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    framing = centered.reshape(-1, 3)
    if reference is not None:
        reference_vertices, reference_roots = reference
        reference_vertices = reference_vertices.copy()
        reference_vertices[:, :, 0] -= reference_roots[:, None, 0]
        reference_vertices[:, :, 2] -= reference_roots[:, None, 2]
        framing = np.concatenate((framing, reference_vertices.reshape(-1, 3)))
    projection = framing @ np.stack([right, up], axis=1)
    low, high = projection.min(axis=0), projection.max(axis=0)
    center = (low + high) / 2
    target = right * center[0] + up * center[1]
    half_height = max((high[1]-low[1])/2, (high[0]-low[0])/2 * STAGE_HEIGHT/STAGE_WIDTH) * 1.13
    spec = dict(width=STAGE_WIDTH, height=STAGE_HEIGHT, frames=len(vertices),
                vertex_count=vertices.shape[1], half_height=float(half_height),
                camera_target=target.tolist(), camera_position=(target+backward*6).tolist())
    centered.astype('<f4').tofile(directory / 'vertices.bin')
    faces.astype('<u4').tofile(directory / 'faces.bin')
    (directory / 'scene.json').write_text(json.dumps(spec))
    shutil.copyfile(ROOT / 'tools/flowhmr_mesh_scene.html', directory / 'index.html')
    shutil.copyfile(ROOT / 'docs/leaderboards/hf_space_motion_edit/three/three.module.js', directory / 'three.module.js')
    return spec


def compose_frame(frame, stage, bbox, index, total, fps, checkpoint_label):
    image = Image.new('RGB', (WIDTH, HEIGHT), '#f5f7fa')
    draw = ImageDraw.Draw(image)
    draw.text((20, 22), checkpoint_label, font=_font(30, True), fill='#182c3b')
    draw.text((1080, 31), 'MOTIUS  /  NATIVE INFERENCE', font=_font(16, True), fill='#426477')
    draw.text((20, 82), 'INPUT VIDEO + TRACKING', font=_font(14, True), fill='#526472')
    draw.text((1080, 82), 'SMPL-H MESH', font=_font(14, True), fill='#526472')
    source = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize(
        (INPUT_WIDTH, STAGE_HEIGHT), Image.Resampling.LANCZOS)
    image.paste(source, (INPUT_X, INPUT_Y))
    image.paste(stage.resize((STAGE_WIDTH, STAGE_HEIGHT), Image.Resampling.LANCZOS), (STAGE_X, INPUT_Y))
    draw = ImageDraw.Draw(image)
    sx, sy = INPUT_WIDTH / frame.shape[1], STAGE_HEIGHT / frame.shape[0]
    x1, y1, x2, y2 = np.asarray(bbox, dtype=float)
    box = [INPUT_X + round(np.clip(x1*sx, 0, INPUT_WIDTH-1)),
           INPUT_Y + round(np.clip(y1*sy, 0, STAGE_HEIGHT-1)),
           INPUT_X + round(np.clip(x2*sx, 0, INPUT_WIDTH-1)),
           INPUT_Y + round(np.clip(y2*sy, 0, STAGE_HEIGHT-1))]
    draw.rectangle(box, outline='#43e6b5', width=3)
    label_y = max(INPUT_Y+4, box[1]-27)
    draw.rounded_rectangle((box[0], label_y, box[0]+108, label_y+24), radius=4, fill='#43e6b5')
    draw.text((box[0]+9,label_y+4), 'PERSON 01', font=_font(13,True), fill='#123d35')
    draw.text((20, 721), 'Original video · tracking bbox from the Motius frontend', font=_font(16), fill='#526472')
    draw.text((1080, 721), 'Fixed camera · root-centered world view', font=_font(16), fill='#526472')
    draw.text((20, 760), f'{index/fps:05.2f} / {total/fps:05.2f} s', font=_font(14,True), fill='#284d64')
    draw.text((1425, 760), f'{fps:g} fps · {total} frames', font=_font(14), fill='#526472')
    draw.rounded_rectangle((210,766,1375,770),radius=2,fill='#dce4eb')
    if index:
        draw.rounded_rectangle((210,766,210+round(1165*index/max(1,total-1)),770),radius=2,fill='#4687ad')
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('capture', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--body-model', type=Path, required=True)
    parser.add_argument('--checkpoint-label', default='FlowHMR')
    parser.add_argument('--camera-reference', type=Path,
                        help='Also fit this capture so two variants use identical camera framing')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max-frames', type=int)
    parser.add_argument('--preview', action='store_true', help='Write a PNG contact sheet instead of video')
    parser.add_argument('--ffmpeg', default=os.environ.get('FLOWHMR_FFMPEG') or imageio_ffmpeg.get_ffmpeg_exe())
    args = parser.parse_args()
    result = load_monocular_capture_result(args.capture)
    if result.source_model != 'FlowHMR' or len(result.tracks) != 1:
        raise ValueError('Expected one Motius FlowHMR track')
    track = result.tracks[0]
    bbox = np.asarray(track.native_parameters.get('bbox_xyxy'))
    if bbox.shape != (track.num_frames,4) or not np.isfinite(bbox).all():
        raise ValueError('Capture lacks a finite tracking bbox for each frame')
    if track.joints_world is None or track.joints_world.shape[1:] != (22,3):
        raise ValueError('Capture lacks predicted SMPL-H body joints')
    count = min(args.max_frames or track.num_frames, track.num_frames)
    if count <= 0:
        raise ValueError('No frames to render')
    if not np.array_equal(track.frame_ids[:count], np.arange(count)):
        raise ValueError('Demo requires a contiguous track starting at video frame 0')
    fps = float(result.output_fps)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened() or abs(cap.get(cv2.CAP_PROP_FPS)-fps)>0.05:
        raise ValueError('Cannot open source video, or its FPS differs from capture')
    if abs(cap.get(cv2.CAP_PROP_FRAME_WIDTH)/cap.get(cv2.CAP_PROP_FRAME_HEIGHT)-16/9)>0.01:
        raise ValueError('This side-by-side layout expects a 16:9 source video')
    vertices, faces = reconstruct_meshes(track, args.body_model, args.device, count)
    reference = None
    if args.camera_reference:
        reference_capture = load_monocular_capture_result(args.camera_reference)
        if len(reference_capture.tracks) != 1:
            raise ValueError('Expected one camera-reference track')
        reference_track = reference_capture.tracks[0]
        if reference_track.num_frames != track.num_frames:
            raise ValueError('Camera-reference capture has a different frame count')
        reference_vertices, _ = reconstruct_meshes(reference_track, args.body_model, args.device, count)
        reference = (reference_vertices, reference_track.joints_world[:count,0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame_indices = sorted(set([0, count//4, count//2, 3*count//4, count-1])) if args.preview else range(count)
    process = None
    with tempfile.TemporaryDirectory(prefix='motius-flowhmr-scene-') as temporary:
        directory = Path(temporary)
        spec = prepare_scene(directory, vertices, faces, track.joints_world[:count,0], reference)
        server = http.server.ThreadingHTTPServer(('127.0.0.1',0), partial(QuietHandler,directory=str(directory)))
        threading.Thread(target=server.serve_forever,daemon=True).start()
        try:
            with sync_playwright() as playwright:
                launch = dict(headless=True, args=['--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
                if os.environ.get('MOTIUS_CHROMIUM_EXECUTABLE'):
                    launch['executable_path'] = os.environ['MOTIUS_CHROMIUM_EXECUTABLE']
                browser = playwright.chromium.launch(**launch)
                page = browser.new_page(viewport=dict(width=STAGE_WIDTH,height=STAGE_HEIGHT))
                page_errors = []
                page.on('pageerror', lambda error: page_errors.append(str(error)))
                page.goto(f'http://127.0.0.1:{server.server_port}/index.html')
                page.wait_for_function('window.sceneReady === true',timeout=60000)
                if not args.preview:
                    # Log to a regular file to avoid an ffmpeg stderr pipe filling
                    # while long videos are streamed into stdin.
                    log = open(directory/'ffmpeg.log','wb')
                    command = [args.ffmpeg,'-y','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24',
                               '-s',f'{WIDTH}x{HEIGHT}','-r',str(fps),'-i','-',
                               '-i',str(args.video),'-map','0:v:0','-map','1:a:0?',
                               '-c:v','libx264','-preset','medium','-crf','19','-pix_fmt','yuv420p',
                               '-c:a','aac','-b:a','160k','-t',f'{count/fps:.6f}',
                               '-movflags','+faststart',str(args.output)]
                    process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=log)
                previews = []
                for index in frame_indices:
                    if args.preview:
                        cap.set(cv2.CAP_PROP_POS_FRAMES,index)
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f'Source video ended at frame {index}')
                    data = page.evaluate('(index) => window.renderFrame(index)',index)
                    stage = Image.open(BytesIO(base64.b64decode(data))).convert('RGB')
                    picture = compose_frame(frame,stage,bbox[index],index,count,fps,args.checkpoint_label)
                    if args.preview:
                        previews.append(picture)
                    else:
                        process.stdin.write(picture.tobytes())
                        if index == min(180,count-1):
                            picture.save(args.output.with_name(args.output.stem+'_preview.png'))
                    if args.preview or (index+1)%60==0 or index+1==count:
                        print(f'Rendered {index+1}/{count} frames',flush=True)
                if page_errors:
                    raise RuntimeError('; '.join(page_errors))
                browser.close()
                if args.preview:
                    sheet = Image.new('RGB',(WIDTH,HEIGHT*len(previews)),'white')
                    for index,picture in enumerate(previews):
                        sheet.paste(picture,(0,index*HEIGHT))
                    sheet.save(args.output)
                else:
                    process.stdin.close()
                    code = process.wait()
                    log.close()
                    if code:
                        raise RuntimeError((directory/'ffmpeg.log').read_text()[-3000:])
                    metadata = dict(renderer='motius-threejs-smooth-mesh', checkpoint_label=args.checkpoint_label,
                                    checkpoint_sha256=result.checkpoint_sha256, source_revision=result.source_revision,
                                    frames=count,fps=fps,width=WIDTH,height=HEIGHT,vertices=int(vertices.shape[1]),
                                    faces=int(len(faces)),view='root-centered world; fixed camera; original Y retained',
                                    camera=spec,source_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                                    video_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest())
                    metadata['inference'] = dict(result.metadata)
                    if args.camera_reference:
                        metadata['camera_reference_sha256'] = hashlib.sha256(args.camera_reference.read_bytes()).hexdigest()
                    args.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        finally:
            cap.release()
            server.shutdown()
            if process is not None and process.poll() is None:
                process.kill()
                process.wait()
    print(f'Saved {args.output}',flush=True)


if __name__ == '__main__':
    main()
