"""KIMODO bundle holds trainable neural components, independent of its sampler."""
from types import SimpleNamespace
import numpy as np
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.models.kimodo import KIMODOBundle
from motius.models.kimodo.network.model.networks import KimodoNetworks
from motius.models.kimodo.network.model.twostage_denoiser import TwostageDenoiser
from motius.motion.body_models.kimodo import SOMASkeleton30
from motius.motion.representation.kimodo import KimodoMotionRep


def test_atomic_denoiser_forward_backward_and_checkpoint(tmp_path):
    skeleton=SOMASkeleton30()
    body_dim=skeleton.nbjoints*12+4
    for part,dim in [('global_root',5),('local_root',4),('body',body_dim)]:
        folder=tmp_path/part;folder.mkdir()
        np.save(folder/'mean.npy',np.zeros(dim,dtype=np.float32))
        np.save(folder/'std.npy',np.ones(dim,dtype=np.float32))
    representation=KimodoMotionRep(skeleton=skeleton,fps=30,stats_path=str(tmp_path))
    denoiser=TwostageDenoiser(representation,'concat',llm_shape=[1,8],use_text_mask=False,
        latent_dim=16,ff_size=32,num_layers=1,num_heads=2,activation='gelu',dropout=0.,
        pe_dropout=0.,num_text_tokens_override=3,input_first_heading_angle=True)
    bundle=KIMODOBundle.__new__(KIMODOBundle);ModelBundle.__init__(bundle)
    bundle._model=KimodoNetworks(denoiser,num_base_steps=10,device='cpu')
    bundle._save_ckpt_modules=['_model'];bundle._trainable_modules=['_model']
    calls=[]
    handle=bundle._model.register_forward_hook(lambda *args:calls.append(True))
    motion=torch.randn(2,5,representation.motion_rep_dim)
    output=bundle.predict_motion(motion,torch.tensor([[True]*5,[True]*3+[False]*2]),
        torch.randn(2,3,8),torch.ones(2,3,dtype=torch.bool),torch.tensor([3,7]),
        torch.zeros(2),torch.zeros_like(motion),torch.zeros_like(motion))
    assert output.shape==motion.shape and torch.isfinite(output).all()
    output.square().mean().backward()
    assert calls==[True]
    assert denoiser.root_model.output_linear.weight.grad.abs().sum()>0
    assert denoiser.body_model.output_linear.weight.grad.abs().sum()>0
    assert not hasattr(bundle,'generate')
    saved=bundle.state_dict_to_save()
    assert saved
    bundle.load_state_dict_selective(saved)
    handle.remove()


def test_g1_mujoco_qpos_conversion_completes_motion():
    from motius.motion.body_models.kimodo.registry import build_skeleton
    from motius.motion.export.kimodo import MujocoQposConverter

    converter = MujocoQposConverter(build_skeleton(34))
    qpos = torch.zeros(8, 36)
    qpos[:, 3] = 1.0
    qpos[:, 2] = 0.8
    motion = converter.qpos_to_motion_dict(qpos, 30.0)
    assert motion["posed_joints"].shape == (8, 34, 3)
    assert motion["foot_contacts"].shape == (8, 4)
    recovered = converter.dict_to_qpos(motion, numpy=False)
    torch.testing.assert_close(recovered.squeeze(0), qpos, atol=1e-5, rtol=1e-5)
