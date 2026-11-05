import imageio.v3 as iio

def get_video_lwh(video_path):
    L, H, W, _ = iio.improps(video_path, plugin='pyav').shape
    return (L, W, H)

def read_video_np(video_path, start_frame=0, end_frame=-1, scale=1.0):
    """
    Args:
        video_path: str
    Returns:
        frames: np.array, (N, H, W, 3) RGB, uint8
    """
    filter_args = []
    should_check_length = False
    if not (start_frame == 0 and end_frame == -1):
        if end_frame == -1:
            filter_args.append(('trim', f'start_frame={start_frame}'))
        else:
            should_check_length = True
            filter_args.append(('trim', f'start_frame={start_frame}:end_frame={end_frame}'))
    if scale != 1.0:
        filter_args.append(('scale', f'iw*{scale}:ih*{scale}'))
    frames = iio.imread(video_path, plugin='pyav', filter_sequence=filter_args)
    if should_check_length:
        assert len(frames) == end_frame - start_frame
    return frames
