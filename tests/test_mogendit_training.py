"""Dedicated MoGenDiT training: true optimization, native corruption, and resume."""
import copy
import json
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motius import Pipeline
from motius.models.mogendit import MoGenDiTBundle
from motius.motion.representation.mogendit.motion_representation import OccamMotionRep
from motius.datasets.mogendit import MoGenDiTDataset
from motius.trainers.mogendit import MoGenDiTTrainer
from motius.trainers.mogendit.run import run


@pytest.fixture(autouse=True)
def single_thread():
    count = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(count)


@pytest.fixture
def setup(tmp_path):
    bundle = MoGenDiTBundle(network_config=dict(d_model=16, n_head=2, n_stack=1, window_size=4))
    root = tmp_path/'data'
    root.mkdir()
    rep = OccamMotionRep(global_pose=bundle.config["global_pose"], fps=bundle.config["fps"])
    for i, length in enumerate([12, 9, 7, 8]):
        pose = torch.eye(3).expand(length,22,3,3).clone()
        joint = rep.body_model.joint_fk_local(pose, rep.body_model.joint_offset)[...,:3]
        trans = torch.zeros(length,3)
        trans[:,1] = 1
        trans[:,0] = torch.linspace(0,0.2,length)
        torch.save(dict(pose=pose,joint=joint,trans=trans),root/f'{i}.pt')
    data = MoGenDiTDataset(root, fix_len=12, min_len=2)
    batch = torch.utils.data.default_collate([data[0],data[2]])
    return bundle, root, data, batch


def test_dataset_native_encoding_and_padding(setup):
    _, _, data, batch = setup
    assert batch['motion'].shape == (2,12,201)
    assert batch['length'].tolist() == [12,7]
    assert batch['motion'][1,7:].count_nonzero() == 0
    assert data.fingerprint


def test_masks_use_real_sequence_end_and_seed(setup):
    bundle, _, _, batch = setup
    trainer = MoGenDiTTrainer(bundle, motion_degradation=False)
    scheduler = trainer.mask_scheduler
    for mode in ['random_start_end','random_frame','random_phrase','block_trans','joint_only','uncond']:
        random.seed(9); np.random.seed(9)
        a=scheduler.get_formulated_mask(batch['motion'],batch['length'],{mode:1.0})
        random.seed(9); np.random.seed(9)
        b=scheduler.get_formulated_mask(batch['motion'],batch['length'],{mode:1.0})
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert a[1,7:].count_nonzero()==0
        if mode=='random_start_end':
            assert a[1,6].all()


def test_corruption_preserves_target_and_padding(setup):
    bundle, _, _, batch=setup
    trainer=MoGenDiTTrainer(bundle,degrade_rate=1.0)
    original=batch['motion'].clone()
    # Force a visible corruption to audit clean-target supervision and anchoring.
    def corrupt(motion, keyframe_mask, **kwargs):
        return motion + (1-keyframe_mask)*0.3
    trainer.corruption.motion_degradation_batch=corrupt
    random.seed(1)
    p=trainer.prepare_batch(batch)
    torch.testing.assert_close(p['clean'],original,rtol=0,atol=0)
    torch.testing.assert_close(batch['motion'],original,rtol=0,atol=0)
    assert p['noisy'][1,7:].count_nonzero()==0
    observed=p['keep_mask'].bool()
    torch.testing.assert_close(p['noisy'][observed],original[observed],rtol=0,atol=0)


def test_native_degradation_and_consistency_backward(setup):
    bundle, _, _, batch=setup
    trainer=MoGenDiTTrainer(bundle,degrade_rate=1.,consis_start_step=0)
    trainer.runner=SimpleNamespace(global_step=1)
    output=trainer.train_step(batch)
    assert torch.isfinite(output['loss'])
    output['loss'].backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in bundle.parameters())
    assert bundle.net.token_2_motion.weight.grad.abs().sum()>0


def test_losses_ignore_padding_and_consistency_boundary(setup):
    bundle, _, _, batch=setup
    trainer=MoGenDiTTrainer(bundle,consis_start_step=3)
    clean=batch['motion']; lengths=batch['length']; weight=torch.tensor([1.,2.])
    prediction=(clean+torch.randn_like(clean)*.02).requires_grad_()
    a, parts=trainer.compute_losses(prediction,clean,lengths,weight,step=3)
    assert parts['loss_consis']==0
    altered=prediction.detach().clone(); altered[1,7:]=123
    b,_=trainer.compute_losses(altered,clean,lengths,weight,step=3)
    torch.testing.assert_close(a,b)
    active, parts=trainer.compute_losses(prediction,clean,lengths,weight,step=4)
    assert parts['loss_consis']>0
    torch.testing.assert_close(active,a+parts['loss_consis'])
    active.backward()
    assert torch.isfinite(prediction.grad).all()


def tiny_config(root):
    return dict(model=dict(version='0.1B',global_pose=True,network_config=dict(d_model=16,n_head=2,n_stack=1,window_size=4)),
                data=dict(data_roots=[str(root)],fix_len=12,min_len=2),
                trainer=dict(motion_degradation=False,consis_start_step=0),
                training=dict(batch_size=2,max_steps=4,lr=1e-3,weight_decay=1e-4,grad_clip=1.,
                              ema_decay=.9,ema_start_step=0,seed=10,save_interval=2,log_interval=1))


def test_exact_resume_optimizer_ema_rng_and_inference_export(setup,tmp_path):
    _,root,_,_=setup
    config=tiny_config(root)
    run(config,tmp_path/'full',device='cpu')
    partial=copy.deepcopy(config);partial['training']['max_steps']=1
    run(partial,tmp_path/'split',device='cpu')
    run(config,tmp_path/'split',resume=tmp_path/'split/latest.pt',device='cpu')
    full=torch.load(tmp_path/'full/latest.pt',weights_only=True)
    split=torch.load(tmp_path/'split/latest.pt',weights_only=True)
    for group in ['model','ema']:
        for key in full[group]:
            torch.testing.assert_close(full[group][key],split[group][key],rtol=0,atol=0)
    for key, state in full['optimizer']['state'].items():
        for name,value in state.items():
            torch.testing.assert_close(value,split['optimizer']['state'][key][name],rtol=0,atol=0)
    assert full['step']==split['step']==4
    for name in ['model','ema']:
        pipeline=Pipeline.from_pretrained(str(tmp_path/'split'/name))
        assert pipeline.bundle.net.token_2_motion.weight.abs().sum()>0
    # Warm-start the original checkpoint format emitted by an artifact.
    bundle=MoGenDiTBundle(network_config=config['model']['network_config'])
    bundle.load_weights(tmp_path/'split/model/model.pt')


def test_tiny_set_loss_decreases(setup):
    bundle,_,_,batch=setup
    trainer=MoGenDiTTrainer(bundle,motion_degradation=False,consis_loss=False,keyframe_modes={'uncond':1.})
    optimizer=torch.optim.AdamW(bundle.trainable_parameters(),lr=.01)
    losses=[]
    for _ in range(35):
        # Fixed noise/timesteps isolate the optimization path.
        random.seed(17);np.random.seed(17);torch.manual_seed(17)
        optimizer.zero_grad()
        out=trainer.train_step(batch);out['loss'].backward();optimizer.step()
        losses.append(float(out['loss'].detach()))
    assert losses[-1] < losses[0]*.6


@pytest.mark.parametrize('name', ['joint_orientation_pops','joint_rotation_pops','pose_twist',
                                  'candy_wrapper_twist','frozen_frame','d_translation_drift',
                                  'd_translation_distortion'])
def test_each_native_corruption_is_finite(setup,name):
    bundle,_,_,batch=setup
    degradation=MoGenDiTTrainer(bundle).corruption.degradation
    degradation.degradation_functions=[name]
    degradation.degradation_probabilities=torch.ones(1)
    trainer=MoGenDiTTrainer(bundle,degrade_rate=1.)
    random.seed(21);np.random.seed(21);torch.manual_seed(21)
    prepared=trainer.prepare_batch(batch)
    assert torch.isfinite(prepared['noisy']).all()
    assert prepared['noisy'][1,7:].count_nonzero()==0
