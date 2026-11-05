# SPDX-License-Identifier: GPL-3.0-or-later
"""External Python entry point. Blender's interpreter never imports Motius/Torch."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def export_capture(capture, body_model, output):
    import numpy as np
    import torch
    from motius.motion.representation.flowhmr.geometry import rot6d_to_rotation_matrix
    from motius.motion.body_models.flowhmr import SMPLSkeleton
    from motion_data import SCHEMA, validate_motion
    if capture.source_model != 'FlowHMR' or len(capture.tracks) != 1:
        raise ValueError('Expected one FlowHMR capture track')
    track = capture.tracks[0]
    rotations = torch.as_tensor(track.native_parameters['rot6d']).float()
    shapes = torch.as_tensor(track.shape_parameters).float()
    translation = torch.as_tensor(track.native_parameters['trans']).float()
    if rotations.shape[1:] != (52, 6):
        raise ValueError('FlowHMR capture must retain all 52 body and hand joints')
    body = SMPLSkeleton(model_path=str(body_model)).cpu().eval()
    with torch.no_grad():
        fk = body(dict(rot6d=rotations, shapes=shapes, trans=translation))
        local = rot6d_to_rotation_matrix(rotations)
        parents = body.parents[:52].cpu().numpy()
        global_rotations = [local[:, 0]]
        for index in range(1, 52):
            global_rotations.append(global_rotations[int(parents[index])] @ local[:, index])
        # Reconstruct with each frame's shape, rather than dropping hands or
        # deriving orientations from joint directions. Preserve source poses.
        data = dict(schema=np.array(SCHEMA), fps=np.array(capture.output_fps),
                    parents=parents, rest_joints=fk['j_shaped'][0, :52].numpy(),
                    joints_world=fk['keypoints3d'][:, :52].numpy(),
                    global_rotations=torch.stack(global_rotations, 1).numpy(),
                    checkpoint_sha256=np.array(capture.checkpoint_sha256),
                    inference_settings=np.array(json.dumps(capture.metadata.get('inference_settings', {}))))
    validate_motion(data)
    path = Path(output)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, path)
    return len(rotations)


def run(request, status_path):
    if request.get('offline'):
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
    project = Path(request['project_root']).expanduser().resolve()
    if not (project / 'motius').is_dir():
        raise ValueError('Project directory must point to the Motius checkout')
    sys.path.insert(0, str(project))
    os.chdir(project)

    def status(state, message, **extra):
        print(message, flush=True)
        write_json(status_path, dict(state=state, message=message, **extra))

    status('running', 'Checking external Python and FlowHMR environment')
    import inspect
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    torch.backends.cuda.matmul.allow_tf32 = False
    if request.get('frontend_precision', 'source') == 'float32':
        torch.backends.cudnn.allow_tf32 = False
    import cv2  # noqa: F401
    from motius.pipelines.flowhmr import FlowHMRPipeline
    from motius.pipelines.flowhmr.lightweight import resolve_body_model
    if not {'lightweight', 'profile', 'sam_backend', 'vggt_memory_mode', 'sam_frame_batch_size', 'vggt_attention_backend', 'vggt_frame_batch_size', 'vggt_query_chunk_size', 'vggt_token_chunk_size', 'frontend_precision'}.issubset(inspect.signature(FlowHMRPipeline.__init__).parameters):
        raise RuntimeError('Use a Motius 0.3.16 or newer checkout with queued runtime scheduling and precision controls')
    device = request.get('device', 'cuda')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in the external Python environment')
    if device.startswith('cuda') and not request.get('capture'):
        if __package__:
            from .runtime_preflight import inspect_runtime
        else:
            from runtime_preflight import inspect_runtime
        status('running', 'Checking GPU, available memory, video and dependencies', phase='Preflight')
        report = inspect_runtime(request, torch)
        write_json(Path(request['output_dir'])/'preflight.json', report)
        request.update(report.get('schedule', {}))
    elif not device.startswith('cuda') and not request.get('capture') and request.get('kind') != 'doctor':
        # Direct CPU calls are retained for unit/codec tests; the queued product
        # entrypoint requires a CUDA device before submitting inference.
        pass
    if request.get('kind') == 'subjects':
        from subject_worker import run as detect_subjects
        detect_subjects(request, status)
        return
    body = resolve_body_model(request.get('body_model') or None)
    if request.get('kind') == 'doctor':
        import importlib
        import shutil
        for module in ('motius.pipelines.flowhmr.preprocess', 'motius.models.vision.vggt_omega.bundle'):
            importlib.import_module(module)
        if shutil.which('ffmpeg') is None:
            raise RuntimeError('Install ffmpeg and make it available on the external Python PATH')
        # Initialize the codec before importing the body-model package.
        importlib.import_module('motius.motion.representation.flowhmr')
        from motius.motion.body_models.flowhmr import SMPLSkeleton
        model = SMPLSkeleton(model_path=body)
        if model.j_template.shape != (52, 3) or model.j_shapedirs.shape[-1] != 16:
            raise ValueError('Use the official 52-joint neutral SMPL-H with 16 shape coefficients')
        status('completed', f'Environment ready: Python {sys.version.split()[0]}, Torch {torch.__version__}, {device}')
        return
    from motius.motion.representation.monocular_capture import (
        load_monocular_capture_result, save_monocular_capture_result)
    output = Path(request['output_dir'])
    performance = None
    if request.get('capture'):
        status('running', 'Preparing existing FlowHMR capture for Blender')
        capture = load_monocular_capture_result(request['capture'])
    else:
        mode = request.get('mode', 'STANDARD')
        if mode not in ('STANDARD', 'LIGHTWEIGHT'):
            raise ValueError('Unknown inference mode')
        status('running', 'Loading FlowHMR and frontend weights (first run may download weights)')
        profiling_cuda = device.startswith('cuda')
        if profiling_cuda:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        inference_started = time.perf_counter()
        pipeline = FlowHMRPipeline.from_pretrained(request['checkpoint'],
            bundle_kwargs=dict(body_model_path=body, device=device,
                               cache_dir=request.get('cache_dir') or None),
            lightweight=True, profile=True,
            sam_backend='fast' if mode == 'LIGHTWEIGHT' else 'official',
            vggt_memory_mode='streamed',
            vggt_attention_backend=request.get('vggt_attention_backend', 'auto'),
            sam_frame_batch_size=int(request.get('sam_frame_batch_size', 8)),
            frontend_precision=request.get('frontend_precision', 'source'),
            attention_query_chunk_size=int(request.get('query_chunk_size', 256)),
            vggt_frame_batch_size=int(request.get('vggt_frame_batch_size', 8)),
            vggt_query_chunk_size=int(request.get('vggt_query_chunk_size', 4096)),
            vggt_token_chunk_size=int(request.get('vggt_token_chunk_size', 4096)))
        if profiling_cuda:
            torch.cuda.synchronize(device)
        model_loading_seconds = time.perf_counter()-inference_started
        selected = None
        subject_data = None
        if request.get('subjects_file'):
            from subjects import selected_boxes
            selected, subject_data = selected_boxes(request['subjects_file'], request['video'], int(request.get('subject_id', -1)))
        prepared = {}
        if request.get('preprocessing_cache'):
            if mode == 'STANDARD' or request.get('frontend_precision', 'source') != 'source':
                raise ValueError('Existing caches do not record the updated standard/precision settings. Recompute preprocessing for this run.')
            from preprocessing_cache import load
            status('running', 'Verifying video SHA and cached camera / body / hand features')
            prepared = load(request['preprocessing_cache'], request['video'], mode,
                            int(request.get('camera_interval', 1)),
                            dict({k:pipeline.preprocessor_kwargs.get(k) for k in
                                  ('sam3d_ckpt', 'sam3d_config', 'sam3d_mhr', 'yolox_ckpt')},
                                 vggt_checkpoint=pipeline.vggt_checkpoint))
            if selected is not None:
                from subjects import verify_cached_subject
                association = verify_cached_subject(selected, prepared['bbox_xyxy'].numpy(), subject_data, int(request['subject_id']))
                write_json(output/'cached-subject-association.json', association)
            status('running', 'Cached preprocessing verified · running real 20-step FlowHMR sampling')
        else:
            status('running', f'{mode}: tracking → camera estimation → full-hand features → motion sampling')
        if selected is not None and not prepared:
            prepared = dict(bbox_xyxy=torch.from_numpy(selected), transcode=False)
        if __package__:
            from .progress import InferenceProgress
        else:
            from progress import InferenceProgress
        pipeline.progress_callback = InferenceProgress(status)
        capture = pipeline.infer_monocular_motion_capture(subject_data['video'] if subject_data and not request.get('preprocessing_cache') else request['video'],
            work_dir=str(output), max_frames=request.get('max_frames') or None,
            vggt_frame_interval=int(request.get('camera_interval', 1)),
            seeds=[int(request.get('seed', 0))], postprocess=bool(request.get('postprocess', False)), **prepared)
        if profiling_cuda:
            torch.cuda.synchronize(device)
        performance = dict(cpu_threads=torch.get_num_threads(), motion_tf32=False, cudnn_tf32=torch.backends.cudnn.allow_tf32, total_seconds_including_model_loading=time.perf_counter()-inference_started,
                           model_loading_seconds=model_loading_seconds,
                           stages=pipeline.stage_timings,
                           preprocessing_cache_used=bool(request.get('preprocessing_cache')),
                           inference_settings=capture.metadata.get('inference_settings', {}))
        if profiling_cuda:
            performance.update(gpu=torch.cuda.get_device_name(device),
                               peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                               peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        write_json(output/'inference_profile.json', performance)
        save_monocular_capture_result(capture, output / 'result.npz')
        del pipeline
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    status('running', 'Exporting all 52 joints and hand rotations')
    motion = output / 'motion.blender.npz'
    frames = export_capture(capture, body, motion)
    if request.get('surface', False):
        status('running', 'Reconstructing the predicted SMPL-H surface')
        import numpy as np
        from motius.motion.body_models.flowhmr import SMPLMesh
        from motion_data import YUP_TO_BLENDER
        track, = capture.tracks
        model = SMPLMesh(model_path=str(body)).cpu().eval()
        vertices = []
        with torch.no_grad():
            for start in range(0, frames, 16):
                end = min(start+16, frames)
                shapes = track.shape_parameters
                params = dict(rot6d=torch.from_numpy(track.native_parameters['rot6d'][start:end]),
                              trans=torch.from_numpy(track.native_parameters['trans'][start:end]),
                              shapes=torch.from_numpy(shapes if len(shapes) == 1 else shapes[start:end]))
                vertices.append(model(params)['vertices'].numpy())
                status('running', f'Reconstructing surface · {end} / {frames}', phase='Reconstruct surface', current=end, total=frames)
        np.savez_compressed(output/'surface.blender.npz',
            vertices=(np.concatenate(vertices) @ YUP_TO_BLENDER.T).astype(np.float32),
            faces=np.asarray(model.faces, dtype=np.int32))
    rates = {}
    if performance is not None and frames > 0:
        seconds = performance['total_seconds_including_model_loading']
        if seconds > 0:
            rates = dict(inference_ms_per_frame=1000.0*seconds/frames,
                         inference_frames_per_second=frames/seconds)
            performance.update(frames=frames, **rates)
            write_json(output/'inference_profile.json', performance)
    status('completed', f'Ready: {frames} frames · source {capture.output_fps:g} fps',
           motion_file=str(motion), frames=frames, fps=capture.output_fps,
           checkpoint_sha256=capture.checkpoint_sha256,
           inference_settings=capture.metadata.get('inference_settings', {}), **rates)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding='utf-8'))
    output = Path(request['output_dir'])
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'status.json'
    try:
        run(request, status_path)
    except Exception as error:
        traceback.print_exc()
        torch = sys.modules.get('torch')
        oom = torch is not None and isinstance(error, torch.cuda.OutOfMemoryError)
        preflight = output/'preflight.json'
        report = json.loads(preflight.read_text()) if preflight.exists() else {}
        write_json(status_path, dict(state='failed', message=str(error),
            error_code='cuda_oom' if oom else 'preflight' if type(error).__name__ == 'PreflightError' else 'runtime',
            schedule=report.get('schedule', {}),
            recovery='Retry with smaller workspace batches; all frames and original precision are retained.' if oom else 'Read job.log for details.'))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
