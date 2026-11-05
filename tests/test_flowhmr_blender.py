"""Blender-free protocol, packaging and subprocess isolation checks."""
from pathlib import Path
import sys
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest

from blender_extension.motius_flowhmr import jobs, motion_data, worker
from tools.package_flowhmr_blender import build


def test_contact_grounding_pins_support_without_flattening_a_jump():
    from blender_extension.motius_flowhmr.grounding_math import solve, forward
    rest = np.array([[0, 0, 1.], [.12, 0, 1.], [.12, .06, .55], [.12, 0, .10],
                     [-.12, 0, 1.], [-.12, .06, .55], [-.12, 0, .10]])
    parents = np.array([-1, 0, 1, 2, 0, 4, 5])
    rotations = np.broadcast_to(np.eye(3), (65, 7, 3, 3)).copy()
    joints = []
    for t in range(65):
        root = rest[0]+[t*.004, 0, .08]
        if 20 <= t <= 44:
            root[2] += .42*np.sin(np.pi*(t-20)/24)
        joints.append(forward(rest, parents, rotations[t], root))
    joints = np.asarray(joints)
    soles = joints[:, [3, 6]].copy()
    soles[:, :, 2] -= .1
    corrected, rs, report = solve(joints, rotations, rest, parents, soles,
        soles[:, :, 2], soles[:, :, 2].min(1), [(1, 2, 3), (4, 5, 6)], 30)
    planted = report['weights'] >= .99
    np.testing.assert_allclose((corrected[:, [3, 6], 2]-.1)[planted], 0, atol=1e-6)
    np.testing.assert_allclose(corrected[:, [3, 6], :2][planted], report['sole_anchors'][:, :, :2][planted], atol=1e-6)
    assert corrected[:, [3, 6], 2].min(1).max()-.1 > .35
    for i in range(1, len(parents)):
        np.testing.assert_allclose(np.linalg.norm(corrected[:, i]-corrected[:, parents[i]], axis=-1),
                                   np.linalg.norm(rest[i]-rest[parents[i]]), atol=1e-6)
    np.testing.assert_allclose(rs @ rs.swapaxes(-1, -2), np.broadcast_to(np.eye(3), rs.shape), atol=1e-6)


def test_frontend_cache_is_bound_to_video_mode_and_file_contents(tmp_path):
    import hashlib
    import json
    import torch
    from blender_extension.motius_flowhmr.preprocessing_cache import load
    video = tmp_path/'source.mp4'
    video.write_bytes(b'fixture-video-identity')
    torch.save(torch.ones(3, 3072), tmp_path/'features.pt')
    np.save(tmp_path/'boxes.npy', np.ones((3, 4), np.float32))
    np.savez(tmp_path/'camera.npz', camera_RT=np.tile(np.eye(4), (3, 1, 1)), K=np.tile(np.eye(3), (3, 1, 1)))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = dict(schema='motius.flowhmr.frontend.v1', video_sha256=sha(video),
                    mode='LIGHTWEIGHT', camera_interval=100, frames=3,
                    files={p:sha(tmp_path/p) for p in ['features.pt', 'boxes.npy', 'camera.npz']})
    (tmp_path/'frontend-cache.json').write_text(json.dumps(manifest))
    prepared = load(tmp_path, video, 'LIGHTWEIGHT', 100)
    assert prepared['preprocessor'].extract_features(None, None, None).shape == (3, 3072)
    with pytest.raises(ValueError, match='mode'):
        load(tmp_path, video, 'STANDARD', 100)
    video.write_bytes(b'different-input')
    with pytest.raises(ValueError, match='different video'):
        load(tmp_path, video, 'LIGHTWEIGHT', 100)


def motion():
    parents = np.array([-1]+list(range(51)))
    return dict(schema=np.array(motion_data.SCHEMA), fps=np.array(30.), parents=parents,
                rest_joints=np.zeros((52, 3)), joints_world=np.zeros((2, 52, 3)),
                global_rotations=np.broadcast_to(np.eye(3), (2, 52, 3, 3)).copy())


def test_coordinate_conversion_is_proper_rotation():
    data = motion()
    data['joints_world'][:] = np.array([1., 2., 3.])
    converted = motion_data.to_blender(data)
    np.testing.assert_allclose(converted['joints_world'][0, 0], [1., -3., 2.])
    np.testing.assert_allclose(converted['global_rotations'], data['global_rotations'])
    assert np.linalg.det(motion_data.YUP_TO_BLENDER) == 1
    assert len(motion_data.JOINT_NAMES) == len(motion_data.MIXAMO_NAMES) == 52


@pytest.mark.parametrize('corruption', ['schema', 'parents', 'nan', 'rotation', 'empty'])
def test_invalid_motion_is_rejected_before_scene_changes(corruption):
    data = motion()
    if corruption == 'schema': data['schema'] = np.array('original_capture_not_interchange')
    if corruption == 'parents': data['parents'][1] = 51
    if corruption == 'nan': data['joints_world'][0, 0, 0] = np.nan
    if corruption == 'rotation': data['global_rotations'][0, 0] *= 2
    if corruption == 'empty': data['joints_world'] = np.zeros((0, 52, 3))
    with pytest.raises(ValueError):
        motion_data.validate_motion(data)


def test_pickle_payload_is_never_loaded(tmp_path):
    path = tmp_path/'malformed.npz'
    np.savez(path, schema=np.array([object()], dtype=object))
    with pytest.raises(ValueError, match='Object arrays'):
        motion_data.read_motion(path)


@pytest.mark.parametrize('prefix', ['mixamorig:', 'mixamorig', 'Character:'])
def test_mixamo_namespaces_and_full_finger_order(prefix):
    names = [prefix+name for name in motion_data.MIXAMO_NAMES]
    matches = motion_data.match_mixamo(names)
    assert len(matches) == 52
    assert matches[22] == prefix+'LeftHandIndex1'
    assert matches[34] == prefix+'LeftHandThumb1'
    assert matches[37] == prefix+'RightHandIndex1'
    assert matches[51] == prefix+'RightHandThumb3'
    with pytest.raises(ValueError, match='missing'):
        motion_data.match_mixamo(['Hips'])
    with pytest.raises(ValueError, match='Ambiguous'):
        motion_data.match_mixamo(names+['Other:Hips'])


def test_environment_does_not_inherit_blender_python(monkeypatch):
    monkeypatch.setenv('PYTHONHOME', '/Blender/Python')
    monkeypatch.setenv('PYTHONPATH', '/Blender/scripts')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/Blender/lib:/usr/local/cuda/lib64')
    env = jobs.clean_environment()
    assert 'PYTHONHOME' not in env and 'PYTHONPATH' not in env
    assert env['LD_LIBRARY_PATH'] == '/usr/local/cuda/lib64'


def test_job_reports_invalid_project_and_releases_log(tmp_path):
    # A real external child exercises argv handling with spaces and failure
    # propagation, without importing Torch or requiring GPU/licensed assets.
    directory = jobs.new_directory(tmp_path/'output with spaces')
    job = jobs.Job(sys.executable, dict(project_root=str(tmp_path), output_dir=str(directory)))
    assert job.process.wait(timeout=20) == 1
    assert job.read_status()['state'] == 'failed'
    assert 'Motius checkout' in job.read_status()['message']
    job.close()
    assert job.log.closed
    assert 'ValueError' in jobs.tail_log(job.log_path)


def test_unified_reproducible_package_contains_no_backend_weights(tmp_path):
    reports = build(tmp_path)
    assert [r['default_mode'] for r in reports] == ['STANDARD']
    snapshots = [r['sha256'] for r in reports]
    assert snapshots == [r['sha256'] for r in build(tmp_path)]
    for report in reports:
        with zipfile.ZipFile(tmp_path/report['file']) as archive:
            names = archive.namelist()
            assert 'blender_manifest.toml' in names and '__init__.py' in names
            assert 'LICENSE' in names and 'NOTICE.md' in names
            assert report['default_mode'] in archive.read('defaults.py').decode()
            assert not any('__pycache__' in n or n.endswith(('.npz', '.pt', '.ckpt', '.safetensors')) for n in names)


@pytest.mark.parametrize('mode', ['STANDARD', 'LIGHTWEIGHT'])
def test_worker_dispatches_selected_native_mode_without_changing_sampling(tmp_path, monkeypatch, mode):
    import json
    from motius.pipelines.flowhmr import FlowHMRPipeline
    from motius.motion.representation import monocular_capture
    calls = {}
    capture = SimpleNamespace(output_fps=30, checkpoint_sha256='f'*64,
                              metadata={'inference_settings': {'lightweight': mode == 'LIGHTWEIGHT'}})
    def infer(video, **kwargs):
        calls.update(video=video, inference=kwargs)
        return capture
    def load(checkpoint, **kwargs):
        calls.update(checkpoint=checkpoint, loader=kwargs)
        return SimpleNamespace(infer_monocular_motion_capture=infer,stage_timings=[])
    monkeypatch.setattr(FlowHMRPipeline, 'from_pretrained', load)
    monkeypatch.setattr(monocular_capture, 'save_monocular_capture_result', lambda result, path: None)
    monkeypatch.setattr(worker, 'export_capture', lambda capture, body, motion: 8)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'path', list(sys.path))
    body = tmp_path/'neutral.npz'; body.write_bytes(b'fixture')
    request = dict(project_root=str(Path(__file__).resolve().parents[1]), output_dir=str(tmp_path),
                   body_model=str(body), checkpoint='local/checkpoint', video='input.mp4',
                   device='cpu', mode=mode, seed=17, max_frames=8, query_chunk_size=128)
    worker.run(request, tmp_path/'status.json')
    assert calls['loader']['lightweight'] is True
    assert calls['loader']['profile'] is True
    assert calls['loader']['sam_backend'] == ('fast' if mode == 'LIGHTWEIGHT' else 'official')
    assert calls['loader']['vggt_memory_mode'] == 'streamed'
    assert calls['loader']['sam_frame_batch_size'] == 8
    assert calls['loader']['vggt_attention_backend'] == 'auto'
    report=json.loads((tmp_path/'inference_profile.json').read_text())
    assert report['total_seconds_including_model_loading'] >= report['model_loading_seconds']
    assert report['frames'] == 8
    assert report['inference_ms_per_frame'] == pytest.approx(1000 * report['total_seconds_including_model_loading']/8)
    assert report['inference_frames_per_second'] == pytest.approx(8/report['total_seconds_including_model_loading'])
    assert report['preprocessing_cache_used'] is False
    assert calls['loader']['attention_query_chunk_size'] == 128
    assert calls['inference']['seeds'] == [17]
    assert calls['inference']['vggt_frame_interval'] == 1
    assert calls['inference']['postprocess'] is False
    assert calls['inference']['max_frames'] == 8
    assert json.loads((tmp_path/'status.json').read_text())['state'] == 'completed'


def test_subject_picker_reaches_overlapping_people_without_nearest_guess():
    from blender_extension.motius_flowhmr.subjects import candidates, hit_test
    data = {'tracks': [dict(id=1, frames=[0, 2], boxes=[[0, 0, 100, 100]]*2),
                       dict(id=2, frames=[0], boxes=[[20, 20, 40, 60]])]}
    assert hit_test(candidates(data, 0), (30, 30)) == [2, 1]
    assert hit_test(candidates(data, 0), (99, 99)) == [1]
    assert hit_test(candidates(data, 0), (120, 20)) == []
    assert candidates(data, 1) == []


def test_subject_selection_rejects_wrong_video_missing_people_and_long_gaps(tmp_path):
    import json
    from blender_extension.motius_flowhmr.subjects import digest, selected_boxes
    video = tmp_path/'input.mp4'; video.write_bytes(b'video-a')
    path = tmp_path/'subjects.json'
    data = dict(schema='motius.flowhmr.subjects.v1', video_sha256=digest(video), frames=4,
                tracks=[dict(id=2, frames=[0, 3], boxes=[[0, 0, 10, 20], [3, 0, 13, 20]])])
    path.write_text(json.dumps(data))
    boxes, _ = selected_boxes(path, video, 2)
    np.testing.assert_allclose(boxes[:,0], [0,1,2,3])
    with pytest.raises(ValueError,match='Choose'):
        selected_boxes(path,video,9)
    with pytest.raises(ValueError,match='interrupted'):
        selected_boxes(path,video,2,max_gap=1)
    video.write_bytes(b'video-b')
    with pytest.raises(ValueError,match='another video'):
        selected_boxes(path,video,2)


def test_cached_features_require_unambiguous_selected_subject():
    from blender_extension.motius_flowhmr.subjects import verify_cached_subject
    boxes=np.array([[0,0,100,200],[1,0,101,200]],float)
    data={'tracks':[dict(id=1,frames=[0,1],boxes=boxes.tolist())]}
    assert verify_cached_subject(boxes,boxes+.1,data,1)['minimum_iou']>.99
    with pytest.raises(ValueError,match='different subject'):
        verify_cached_subject(boxes,boxes+100,data,1)
    data['tracks'].append(dict(id=2,frames=[0],boxes=[[10,0,110,200]]))
    with pytest.raises(ValueError,match='ambiguous'):
        verify_cached_subject(boxes,boxes,data,1)


def test_subject_manifest_paths_survive_directory_move(tmp_path):
    import json
    from blender_extension.motius_flowhmr.subjects import load_subjects
    folder = tmp_path/'moved demo'/'detection'; folder.mkdir(parents=True)
    path = folder/'subjects.json'
    path.write_text(json.dumps(dict(video='../input.mp4', preview_sequence='previews/000000.jpg')))
    data = load_subjects(path)
    assert Path(data['video']).resolve() == tmp_path/'moved demo'/'input.mp4'
    assert Path(data['preview_sequence']) == folder/'previews/000000.jpg'


def test_inference_progress_is_measured_throttled_and_resets_between_phases():
    from blender_extension.motius_flowhmr.progress import InferenceProgress
    now = [100.]
    events = []
    progress = InferenceProgress(lambda *args, **kwargs: events.append((args, kwargs)), clock=lambda: now[0])
    progress('Extract body and hand features', 0, 750)
    assert events[-1][1]['eta_seconds'] == -1
    now[0] += .1
    progress('Extract body and hand features', 1, 750)
    assert len(events) == 1
    now[0] += 9.9
    progress('Extract body and hand features', 50, 750)
    assert events[-1][1]['rate_per_second'] == 5
    assert events[-1][1]['eta_seconds'] == 140
    assert events[-1][1]['rate_unit'] == 'frames/s'
    progress('Sample motion', 0, 20)
    assert events[-1][1]['eta_seconds'] == -1
    assert events[-1][1]['rate_unit'] == 'steps/s'
    progress('Sample motion', 20, 20)
    assert events[-1][1]['current'] == 20
