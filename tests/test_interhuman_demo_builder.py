"""Test export contracts with a recording body model; no licensed assets required."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools'))
import build_interhuman_representation_demo as builder


def test_export_centering_is_one_common_translation():
    rng = np.random.default_rng(1)
    joints = rng.normal(size=(5, 2, 22, 3)).astype(np.float32)
    vertices = rng.normal(size=(5, 2, 80, 3)).astype(np.float32)
    vertices[:, 1, :, 1] += 4.
    j, v = builder._center_representations(joints, vertices)
    np.testing.assert_allclose(j - joints, np.broadcast_to((v - vertices)[0, 0, 0], j.shape), atol=1e-6)
    np.testing.assert_allclose(v[:, 1] - v[:, 0], vertices[:, 1] - vertices[:, 0], atol=1e-6)


def test_loader_keeps_zero_shape_and_original_hands(tmp_path, monkeypatch):
    calls = []
    model_options = []
    class Model:
        def to(self, device): return self
        def eval(self): return self
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(joints=torch.zeros(4, 52, 3), vertices=torch.zeros(4, 6890, 3))
    def create(*args, **kwargs):
        model_options.append(kwargs)
        return Model()
    import smplx
    monkeypatch.setattr(smplx, 'create', create)
    monkeypatch.setattr(builder, 'resolve_smpl_model_path', lambda *a, **k: tmp_path)
    for person in ('P1', 'P2'):
        directory = tmp_path / 'smplh_52_2p' / 'sample'
        directory.mkdir(parents=True, exist_ok=True)
        np.savez(directory / f'{person}.npz', global_orient=np.zeros((4, 3)),
                 body_pose=np.zeros((4, 63)), transl=np.ones((4, 3)),
                 betas=np.zeros(16), raw_betas=np.ones(16), gender='male',
                 left_hand_pose=np.full((4, 45), .2), right_hand_pose=np.full((4, 45), -.3))
    builder._load_interx_smplh_pair(tmp_path, 'sample', tmp_path, SimpleNamespace(device='cpu'))
    assert len(calls) == 2
    assert model_options[0]['flat_hand_mean'] is True
    for call in calls:
        assert torch.count_nonzero(call['betas']) == 0
        torch.testing.assert_close(call['left_hand_pose'], torch.full((4, 45), .2))
        torch.testing.assert_close(call['right_hand_pose'], torch.full((4, 45), -.3))
        torch.testing.assert_close(call['transl'], torch.ones(4, 3))


def test_viewer_quantization_preserves_skeleton_mesh_alignment(tmp_path):
    rng = np.random.default_rng(4)
    joints = rng.normal(size=(4, 2, 22, 3)).astype(np.float32)
    vertices = joints.copy()
    payload = builder.write_threejs_viewer_data(joints, vertices, np.array([[0, 1, 2]]),
        tmp_path, sample_id='test', source_label='synthetic test only', route='identity', fps=30, max_frames=None)
    meta = payload['representations']['smpl']
    restored = np.fromfile(tmp_path / 'smpl_pair_vertices.u16', dtype='<u2').reshape(vertices.shape)
    restored = restored * np.array(meta['quantization_scale']) + np.array(meta['quantization_min'])
    np.testing.assert_allclose(restored, payload['representations']['interhuman']['positions'], atol=1e-4)
