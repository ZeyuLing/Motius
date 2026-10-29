"""Original frame timing, resizing and codec normalization at video input."""
import os
import json
import subprocess
import cv2
import glob
import numpy as np

def video2frames(vidfile, save_folder=None, max_height=None, return_images=False, frame_skip=None):
    """Convert input video to images with optional resizing and frame skipping."""
    if frame_skip is None:
        frame_skip = 1
    frames = []
    count = 0
    frame_index = 0
    cap = cv2.VideoCapture(vidfile)
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_index += 1
            if frame_index % frame_skip != 0:
                continue
            processed_frame = _process_frame_dimensions(frame, max_height)
            if return_images:
                frames.append(processed_frame[..., ::-1])
            else:
                cv2.imwrite(f'{save_folder}/{count:04d}.jpg', processed_frame)
            count += 1
    finally:
        cap.release()
    return (count, np.array(frames))

def _process_frame_dimensions(frame, max_height=None):
    """Process frame dimensions to ensure even width/height and apply max_height constraint."""
    height, width = frame.shape[:2]
    adjusted_width = width + width % 2
    adjusted_height = height + height % 2
    if max_height is not None and adjusted_height > max_height:
        target_height = max_height - max_height % 2
        target_width = int(adjusted_width * target_height / adjusted_height)
        target_width = target_width - target_width % 2
        adjusted_width, adjusted_height = (target_width, target_height)
    if adjusted_width != width or adjusted_height != height:
        return cv2.resize(frame, (adjusted_width, adjusted_height), interpolation=cv2.INTER_CUBIC)
    return frame

def get_video_codec(video_path):
    """Get the codec of a video file using ffprobe."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=codec_name', '-of', 'json', video_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        codec_info = json.loads(result.stdout)
        return codec_info['streams'][0]['codec_name']
    except (subprocess.CalledProcessError, KeyError, IndexError, json.JSONDecodeError):
        return None

def convert_av1_to_h264(video_path, output_dir=None):
    """Convert AV1 video to H.264 format using ffmpeg."""
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    output_path = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '_h264.mp4')
    cmd = ['/usr/bin/ffmpeg', '-y', '-loglevel', 'error', '-i', video_path, '-c:v', 'libx264', '-preset', 'fast', '-crf', '23', '-c:a', 'aac', '-b:a', '128k', output_path]
    try:
        subprocess.run(cmd, check=True)
        return output_path
    except subprocess.CalledProcessError:
        raise Exception(f'Failed to convert AV1 to H.264: {video_path}')

def load_video_frames(video_path, output_folder=None, max_height=None, max_fps=None):
    assert os.path.exists(video_path), f'{video_path} does not exist'
    isdir = False
    if os.path.isdir(video_path):
        print(f'Processing image folder {video_path}')
        seq = os.path.basename(video_path)
        isdir = True
        fps = 30
        imgfiles = sorted(glob.glob(f'{video_path}/*.jpg'))
        if output_folder is None:
            seq_folder = f'results/{seq}'
        else:
            seq_folder = output_folder
        frames = np.stack([cv2.imread(f)[..., ::-1] for f in imgfiles])
        return (frames, seq_folder, fps)
    print(f'Processing video file {video_path}')
    codec = get_video_codec(video_path)
    if codec == 'av1':
        print('Converting AV1 to H.264...')
        video_path = convert_av1_to_h264(video_path, output_dir=output_folder)
    seq = os.path.basename(video_path).split('.')[0]
    cap = cv2.VideoCapture(video_path)
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert input_fps > 0, f'input_fps is not positive: {input_fps}'
    assert max_fps is not None and max_fps > 0, f'max_fps is not positive: {max_fps}'
    if input_fps > max_fps:
        target_fps = max_fps
    else:
        target_fps = input_fps
    frame_skip = round(input_fps / target_fps)
    assert frame_skip > 0, f'frame_skip is not positive: {frame_skip}'
    if output_folder is None:
        seq_folder = f'results/{seq}'
    else:
        seq_folder = output_folder
    os.makedirs(seq_folder, exist_ok=True)
    nframes, frames = video2frames(video_path, max_height=max_height, return_images=True, frame_skip=frame_skip)
    print(f'Loaded {nframes} frames, fps: {target_fps}')
    frames = np.array(frames)
    return (frames, seq_folder, target_fps)
