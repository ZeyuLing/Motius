"""Control optimization and training use the neural bundle's atomic forward."""
from types import SimpleNamespace
from unittest.mock import patch
import torch
from torch import nn
from motius.models.base_model_bundle import ModelBundle
from motius.models.maskcontrol import MaskControlBundle, MaskControlTransformer
from motius.models.momask.network.vq.model import RVQVAE
from motius.pipelines.maskcontrol.sampling import MaskControlSampler
from motius.trainers.maskcontrol import MaskControlTrainer
from test_momask_native import TinyTextEncoder


def bundle():
    model=MaskControlBundle.__new__(MaskControlBundle)
    ModelBundle.__init__(model)
    opt=SimpleNamespace(num_tokens=8,num_quantizers=3,code_dim=8,joints_num=22,device='cpu')
    with patch('motius.models.momask.network.mask_transformer.transformer._load_openai_clip_model',
               side_effect=lambda *args:TinyTextEncoder()):
        model.control_model=MaskControlTransformer(code_dim=8,cond_mode='text',latent_dim=8,
            ff_size=16,num_layers=1,num_heads=2,dropout=0.,clip_dim=8,opt=opt)
    vqopt=SimpleNamespace(num_quantizers=3,shared_codebook=False,quantize_dropout_prob=0.,mu=.99)
    model.vq_model=RVQVAE(vqopt,nb_code=8,code_dim=8,output_emb_width=8,down_t=2,width=8,depth=1)
    for layer in model.vq_model.quantizer.layers:layer.codebook.normal_()
    model.residual_model=nn.Identity()
    model.length_estimator=None
    for name in ['mean','control_mean']:model.register_buffer(name,torch.zeros(263))
    for name in ['std','control_std']:model.register_buffer(name,torch.ones(263))
    model.register_buffer('mask_emb_vq',model.vq_model.quantizer.layers[0].codebook.mean(0).detach())
    model.eval().requires_grad_(False)
    return model


def test_training_and_gradient_control_optimization():
    torch.manual_seed(7)
    model=bundle()
    trainer=MaskControlTrainer(model)
    calls=[]
    handle=model.control_model.register_forward_hook(lambda *args:calls.append(True))
    try:
        result=trainer.train_step(dict(text=['walk'],motion=torch.randn(1,40,263)*.01,motion_lens=torch.tensor([40])))
        result['loss'].backward()
        assert calls==[True] and torch.isfinite(result['loss'])
        assert model.control_model.first_zero_linear.weight.grad.abs().sum()>0
        assert model.control_model.token_emb.weight.grad is None
        model.eval().requires_grad_(False)
        target=torch.zeros(1,196,22,3);mask=torch.zeros(1,196,22,dtype=torch.bool)
        target[:,::8,20]=torch.tensor([.1,1.,.2]);mask[:,::8,20]=True
        sampler=MaskControlSampler(model)
        # Small test text tower has eight channels; supplied text exercises it.
        normalized,physical=sampler.sample_motion(['walk'],torch.tensor([40]),target,mask,
            residual_model=None,time_steps=2,each_iterations=1,final_iterations=1)
        assert normalized.shape==(1,196,263) and torch.isfinite(physical).all()
        assert len(calls)==5  # one training call plus conditional/unconditional predictions
    finally:handle.remove()
