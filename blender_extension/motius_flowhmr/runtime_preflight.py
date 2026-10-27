# SPDX-License-Identifier: GPL-3.0-or-later
"""External-environment checks and conservative measured scheduling policy."""
import json
import os
from pathlib import Path
import shutil
import subprocess

GiB = 1024**3


class PreflightError(RuntimeError):
    pass


def memory_policy(free_bytes, total_bytes, frames, mode, requested):
    """Empirical H200 planning envelope, not a hardware compatibility promise."""
    free = free_bytes/GiB
    margin = max(.75,min(2.,total_bytes/GiB*.08))
    # Full temporal keys remain: even streamed execution grows with video length.
    estimate = 5.6+max(0,frames-750)*.004
    if requested.get('frontend_precision', 'source') == 'float32':
        estimate = 9.0+max(0,frames-750)*.008
    warnings = []
    if free < estimate+margin:
        raise PreflightError(f'Insufficient available GPU memory for the current planning envelope: {free:.2f} GiB free, about {estimate+margin:.2f} GiB required for {frames} frames in {mode}. Close GPU applications or use a larger GPU'+(' / explicitly select Lightweight mode.' if mode != 'LIGHTWEIGHT' else '.'))
    schedule = {}
    if mode in {'LIGHTWEIGHT', 'STANDARD'}:
        batch = 1 if free < 8 else 4 if free < 12 else 8
        chunk = 1024 if free < 8 else 2048 if free < 12 else 4096
        for key,default,limit in [('sam_frame_batch_size',8,batch),('vggt_frame_batch_size',8,batch),
                                  ('vggt_query_chunk_size',4096,chunk),('vggt_token_chunk_size',4096,chunk),
                                  ('query_chunk_size',256,256)]:
            value = int(requested.get(key,default))
            if value < 1:
                raise PreflightError(f'{key} must be positive')
            schedule[key] = min(value,limit)
    warnings.append('Memory estimates are calibrated on H200; other GPUs and concurrent allocations can differ. All frames and the selected precision are retained.')
    return dict(estimated_backend_gib=estimate,headroom_gib=margin,schedule=schedule,warnings=warnings)


def inspect_runtime(request, torch):
    device = torch.device(request.get('device','cuda'))
    if device.type != 'cuda':
        raise PreflightError('This inference backend requires a CUDA GPU. CPU/MPS inference has not been validated; choose a CUDA device.')
    if not torch.cuda.is_available():
        raise PreflightError('CUDA is unavailable. Install a compatible CUDA-enabled PyTorch and NVIDIA driver in External Python.')
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index < 0 or index >= torch.cuda.device_count():
        raise PreflightError(f'CUDA device {index} does not exist')
    torch.cuda.set_device(index)
    capability = torch.cuda.get_device_capability(device)
    arch_list = getattr(torch.cuda, 'get_arch_list', lambda: [])()
    majors = {int(arch.split('_', 1)[1]) // 10 for arch in arch_list
              if arch.startswith('sm_') and arch.split('_', 1)[1].isdigit()}
    if majors and capability[0] not in majors:
        raise PreflightError(f"GPU {torch.cuda.get_device_name(device)} (compute {capability[0]}.{capability[1]}) is unsupported by the installed PyTorch CUDA build. Update the runtime or select a compatible External Python.")
    # Reject instead of silently changing BF16 weights to FP16 on older cards.
    use_source_precision = request.get('frontend_precision', 'source') == 'source'
    if use_source_precision and (torch.cuda.get_device_capability(device)[0] < 8 or not torch.cuda.is_bf16_supported()):
        raise PreflightError('This backend uses BF16 frontend weights. The selected GPU does not support the original precision; automatic precision reduction is disabled.')
    probe = torch.ones(8,device=device);probe = probe+1;torch.cuda.synchronize(device);del probe
    # Check the original frontend dtype can execute on this driver/build.
    dtype = torch.bfloat16 if use_source_precision else (torch.float16 if request.get('frontend_precision') == 'float16' else torch.float32)
    matrix=torch.ones((16,16),device=device,dtype=dtype)
    product=matrix@matrix;torch.cuda.synchronize(device);del matrix,product
    free,total = torch.cuda.mem_get_info(device)
    result = dict(device=str(device),gpu=torch.cuda.get_device_name(device),compute_capability=list(torch.cuda.get_device_capability(device)),
                  free_gib=free/GiB,total_gib=total/GiB,torch=torch.__version__,cuda=torch.version.cuda)
    if shutil.which('ffmpeg') is None or shutil.which('ffprobe') is None:
        raise PreflightError('Install FFmpeg and FFprobe on External Python PATH')
    try:
        import psutil
        result['host_available_gib'] = psutil.virtual_memory().available/GiB
    except ImportError:
        if hasattr(os, 'sysconf'):
            result['host_available_gib'] = os.sysconf('SC_AVPHYS_PAGES')*os.sysconf('SC_PAGE_SIZE')/GiB
    # Respect Linux container memory ceilings, not just host-wide free memory.
    try:
        ceiling = Path('/sys/fs/cgroup/memory.max').read_text().strip()
        used = int(Path('/sys/fs/cgroup/memory.current').read_text())
        if ceiling != 'max':
            result['host_available_gib'] = min(result.get('host_available_gib',float('inf')),max(0,int(ceiling)-used)/GiB)
    except (OSError,ValueError):
        pass
    if result.get('host_available_gib',12) < 4:
        raise PreflightError('Less than 4 GiB available system RAM; free host/container memory before inference')
    if not request.get('video') or request.get('kind') == 'doctor':
        return result
    video = Path(request['video'])
    if not video.is_file():
        raise PreflightError('The input video no longer exists')
    data = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames,r_frame_rate,duration:format=duration','-of','json',str(video)],text=True,timeout=30))
    if not data['streams']:
        raise PreflightError('No video stream found')
    info = data['streams'][0]
    try:
        frames = int(info.get('nb_frames',''))
    except ValueError:
        # Header estimate only; never used to truncate the input.
        n,d = map(float,info['r_frame_rate'].split('/'))
        frames = max(1,round(float(info.get('duration') or data['format']['duration'])*n/d))
    if frames < 1:
        raise PreflightError('The video contains no frames')
    result['video'] = dict(frames=frames,width=int(info['width']),height=int(info['height']))
    available = shutil.disk_usage(Path(request['output_dir'])).free
    required_disk = max(.5,frames*.001)*GiB
    if available < required_disk:
        raise PreflightError(f'Insufficient output disk space: {available/GiB:.2f} GiB available, planning allowance {required_disk/GiB:.2f} GiB')
    result['disk_free_gib'] = available/GiB
    if request.get('kind') not in {'doctor','subjects'} and not request.get('capture'):
        result.update(memory_policy(free,total,frames,request.get('mode','STANDARD'),request))
        result['schedule_applied'] = result['schedule']
        if result.get('host_available_gib',12) < 12:
            result['warnings'].append('Less than 12 GiB system RAM available; the complete 750-frame backend measured about 11 GiB RSS. Host memory exhaustion remains possible.')
    return result
