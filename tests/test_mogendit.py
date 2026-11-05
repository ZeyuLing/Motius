"""Checkpoint, representation, masking and repair regressions for recovered MoreDiff."""
import json

import pytest
import torch

from motius import Pipeline
from motius.models.mogendit import MoGenDiTBundle
from motius.motion.representation.mogendit.motion_representation import OccamMotionRep
from motius.pipelines.mogendit import MoGenDiTPipeline


@pytest.fixture
def bundle():
    return MoGenDiTBundle(network_config=dict(d_model=16, n_head=2, n_stack=1, window_size=4))


def motion(bundle, frames=6):
    rep = OccamMotionRep(global_pose=bundle.config["global_pose"], fps=bundle.config["fps"])
    poses = torch.eye(3).expand(frames, 22, 3, 3).clone()
    joints = rep.body_model.joint_fk_local(poses, rep.body_model.joint_offset)[..., :3]
    trans = torch.zeros(frames, 3)
    trans[:, 1] = 1
    return rep.encode(poses, joints, trans)


@pytest.mark.parametrize('global_pose', [True, False])
def test_occam_encode_decode_roundtrip(global_pose, bundle):
    rep = OccamMotionRep(global_pose=bundle.config["global_pose"], fps=bundle.config["fps"])
    rep.global_pose = global_pose
    from motius.motion.representation.mogendit.rotation_conversions import axis_angle_to_matrix
    poses = axis_angle_to_matrix(torch.randn(5, 22, 3) * 0.2)
    joints = torch.randn(5, 22, 3)
    trans = torch.randn(5, 3)
    encoded = rep.encode(poses, joints, trans)
    for actual, expected in zip(rep.decode(encoded), (poses, joints, trans)):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize('mode', ['denoise', 'ada_denoise', 'trans_regen'])
@pytest.mark.parametrize('windowed', [False, True])
def test_repair_modes_preserve_input_and_shapes(bundle, mode, windowed):
    x = motion(bundle)
    original = x.clone()
    with torch.no_grad():
        bundle.net.token_2_motion.bias.copy_(x[0])
    pipeline = MoGenDiTPipeline(bundle)
    result = pipeline(x, mode=mode, step=2, use_windowed=windowed,
                      window_size=4, prev_padding=1, imputation_mode='all')
    torch.testing.assert_close(x, original, rtol=0, atol=0)
    assert result['motion'].shape == (6, 201)
    assert result['poses'].shape == (6, 22, 3, 3)
    assert result['joints'].shape == (6, 22, 3)
    assert result['transl'].shape == (6, 3)
    assert all(torch.isfinite(v).all() for v in result.values())
    expected = MoGenDiTPipeline(bundle).motion_rep.normalization(x, height_reset=True)
    observed = slice(0, 198) if mode == 'trans_regen' else slice(None)
    torch.testing.assert_close(result['motion'][0, observed], expected[0, observed], atol=1e-5, rtol=1e-5)


def test_artifact_roundtrip_and_strict_legacy_checkpoint(bundle, tmp_path):
    bundle.save_pretrained(tmp_path)
    restored = Pipeline.from_pretrained(str(tmp_path))
    for key, tensor in bundle.net.state_dict().items():
        torch.testing.assert_close(restored.bundle.net.state_dict()[key], tensor, rtol=0, atol=0)
    (tmp_path / 'mogendit_config.json').unlink()
    (tmp_path / 'train_args.json').write_text(json.dumps({'model_version': '0.1B', 'global_pose': True}))
    for step, value in [(2, 2.), (11, 11.)]:
        state = bundle.net.state_dict()
        state['token_2_motion.bias'] = torch.full((201,), value)
        torch.save({'model': state}, tmp_path / f'ema_model_{step}.pth')
    restored = MoGenDiTBundle.from_pretrained(tmp_path, network_config=bundle.config['network_config'])
    assert torch.all(restored.net.token_2_motion.bias == 11)
    with pytest.raises(FileNotFoundError):
        MoGenDiTBundle.from_pretrained(tmp_path, use_ema=False)
    torch.save({'model': {'wrong': torch.zeros(1)}}, tmp_path / 'bad.pth')
    with pytest.raises(RuntimeError):
        MoGenDiTBundle.from_pretrained(tmp_path, checkpoint='bad.pth', network_config=bundle.config['network_config'])


def test_diffusion_preserves_observed_channels_exactly(bundle):
    x = torch.randn(2, 4, 201)
    keep = torch.rand_like(x) > 0.5
    wrapped = bundle.net.wrap_inputs(x.clone(), None, keep.float(), None)
    result = MoGenDiTPipeline(bundle).diffusion.denoise(wrapped, bundle.net, num_timesteps=2,
                                     eta=0, device=x.device, mask=keep, imputation_mode='all')
    torch.testing.assert_close(result[keep], x[keep], rtol=0, atol=0)


def test_batch_world_restoration(bundle):
    x = motion(bundle)
    x[:, -3] = 5
    with torch.no_grad():
        bundle.net.token_2_motion.bias.copy_(x[0])
    result = MoGenDiTPipeline(bundle)(x.unsqueeze(0).repeat(2, 1, 1), step=1,
                                    restore_world=True, imputation_mode='all')
    assert result['motion'].shape == (2, 6, 201)
    torch.testing.assert_close(result['transl'][:, 0], x[0, -3:].expand(2, 3), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize('kwargs', [{'step': 1000}, {'step': 0}, {'eta': float('nan')},
                                    {'prev_padding': 4, 'window_size': 4}, {'mode': 'other'}])
def test_invalid_sampling_settings(bundle, kwargs):
    with pytest.raises(ValueError):
        MoGenDiTPipeline(bundle)(motion(bundle), **kwargs)


def test_reject_wrong_representation_and_nonfinite(bundle):
    pipeline = MoGenDiTPipeline(bundle)
    for x in (torch.zeros(4, 198), torch.zeros(0, 201), torch.full((4, 201), float('nan'))):
        with pytest.raises(ValueError):
            pipeline(x)


def test_network_backward(bundle):
    x = torch.randn(2, 4, 201)
    y = bundle(x, torch.tensor([2, 5]))
    (y - x).square().mean().backward()
    assert bundle.net.token_2_motion.weight.grad.abs().sum() > 0
