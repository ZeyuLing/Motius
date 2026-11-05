"""Real small GEM graphs, including DDP through both learned components."""
import importlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from motius.models.gem_bundle import GEMModelBundle
from motius.models.base_model_bundle import ModelBundle


def graph(variant):
    root=Path(__file__).resolve().parents[1]
    cfg=OmegaConf.create(json.loads((root/'motius/configuration'/(variant+'.json')).read_text()))
    cfg.denoiser.latent_dim=32;cfg.denoiser.num_layers=2;cfg.denoiser.num_heads=4
    if variant=='gem_x':cfg.denoiser.fix_scale_comps=False
    conditioner=importlib.import_module('motius.models.'+variant+'.conditioning').ConditionEncoder(cfg.pipeline_args,cfg.model_config,cfg.denoiser,33 if variant=='gem_x' else 17)
    options=OmegaConf.to_container(cfg.denoiser,resolve=True);options.pop('_target_')
    network=importlib.import_module('motius.models.'+variant+'.denoiser').NetworkEncoderRoPE(**options)
    bundle=GEMModelBundle.__new__(GEMModelBundle);ModelBundle.__init__(bundle)
    bundle.conditioner=conditioner;bundle.denoiser=network;bundle.VARIANT=variant;bundle.config=cfg
    return bundle,cfg


def forward(bundle,inputs):
    data=inputs['batch']
    conditional,_,_=bundle.encode_conditions(data,inputs['existence'],inputs['visible'])
    return bundle.predict_denoise(inputs['x'],inputs['timesteps'],y=dict(f_cond=conditional,length=data['length'],encoded_text=data['encoded_text']),inputs=data,sample_indices_dict=inputs['sample_indices_dict'])


def batch(variant,cfg):
    b,l=2,8;j=17 if variant=='gem_smpl' else 33;dims=151 if variant=='gem_smpl' else 585
    data=dict(B=b,L=l,device=torch.device('cpu'),length=torch.tensor([8,6]),obs=torch.randn(b,l,j,3),
              f_cliffcam=torch.randn(b,l,3),f_cam_angvel=torch.randn(b,l,6),f_imgseq=torch.randn(b,l,1024),
              encoded_text=torch.randn(b,5,cfg.denoiser.encoded_text_dim),target_x=torch.randn(b,l,dims),
              music_embed=torch.randn(b,l,35))
    existence={k:torch.ones(b,l,dtype=torch.bool) for k in cfg.pipeline_args.in_attr}
    visible=torch.ones(b,l,j,dtype=torch.bool);existence['obs']=visible
    indices=({'betas':(126,136)} if variant=='gem_smpl' else {'identity_coeffs':(456,501),'scale_params':(501,570)})
    return dict(x=torch.randn(b,l,dims),timesteps=torch.tensor([12,99]),batch=data,existence=existence,visible=visible,sample_indices_dict=indices)


@pytest.mark.parametrize('variant',['gem_smpl','gem_x'])
def test_actual_learned_condition_and_denoiser(variant):
    torch.set_num_threads(1);model,cfg=graph(variant)
    result=forward(model,batch(variant,cfg));loss=sum(x.square().mean() for x in result.values() if isinstance(x,torch.Tensor));loss.backward()
    assert torch.isfinite(loss)
    assert model.conditioner.embed_noisyobs.fc1.weight.grad.abs().sum()>0
    assert model.denoiser.final_layer.fc1.weight.grad.abs().sum()>0


def _worker(rank,path,variant):
    from torch.nn.parallel import DistributedDataParallel as DDP
    import torch.distributed as dist
    torch.set_num_threads(1);dist.init_process_group('gloo',init_method='file://'+path,rank=rank,world_size=2)
    torch.manual_seed(91);model,cfg=graph(variant);model.conditioner=DDP(model.conditioner,find_unused_parameters=True);model.denoiser=DDP(model.denoiser,find_unused_parameters=True)
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-4)
    for i in range(2):
        torch.manual_seed(101+rank+i);optimizer.zero_grad(set_to_none=True)
        loss=0
        for stage in range(2):
            result=forward(model,batch(variant,cfg))
            loss=loss+sum(x.square().mean() for x in result.values() if isinstance(x,torch.Tensor))
        loss.backward();optimizer.step()
    weight=model.conditioner.module.embed_noisyobs.fc1.weight.detach();values=[torch.empty_like(weight) for _ in range(2)];dist.all_gather(values,weight)
    assert torch.equal(values[0],values[1]);dist.destroy_process_group()


@pytest.mark.parametrize('variant',['gem_smpl','gem_x'])
def test_two_process_conditioner_and_denoiser_roots(tmp_path,variant):
    torch.multiprocessing.spawn(_worker,args=(str(tmp_path/'group'),variant),nprocs=2,join=True)


@pytest.mark.parametrize('variant',['gem_smpl','gem_x'])
def test_native_checkpoint_roundtrip_saves_current_neural_weights(tmp_path,variant):
    from motius.models.gem_smpl import GemSmplBundle
    from motius.models.gem_x import GemXBundle
    from motius.pipelines.auto import Pipeline
    torch.set_num_threads(1)
    reference,cfg=graph(variant)
    artifact=tmp_path/'source';checkpoint='gem_smpl.ckpt' if variant=='gem_smpl' else 'gem_soma.ckpt'
    destination=artifact/'inputs/pretrained'/checkpoint;destination.parent.mkdir(parents=True)
    state=dict(reference.conditioner.state_dict())
    state.update({'pipeline.denoiser3d.denoiser.'+key:value for key,value in reference.denoiser.state_dict().items()})
    torch.save({'state_dict':state},destination)
    (artifact/'gem_architecture.json').write_text(json.dumps(OmegaConf.to_container(cfg,resolve=True)))
    cls=GemSmplBundle if variant=='gem_smpl' else GemXBundle
    model=cls(artifact,device='cpu')
    with torch.no_grad():model.denoiser.final_layer.fc1.weight.add_(.25)
    inputs=batch(variant,cfg)
    with torch.no_grad():expected=forward(model,inputs)
    saved=tmp_path/'saved';model.save_pretrained(saved)
    reloaded=cls.from_pretrained(saved,device='cpu')
    with torch.no_grad():actual=forward(reloaded,inputs)
    assert torch.equal(model.denoiser.final_layer.fc1.weight,reloaded.denoiser.final_layer.fc1.weight)
    assert all(torch.equal(expected[key],actual[key]) for key in expected)
    manifest=json.loads((saved/'model_index.json').read_text())
    from motius.pipelines.auto import _pipeline_path_from_manifest
    assert _pipeline_path_from_manifest(manifest)=='motius.pipelines.'+variant+'.'+('GemSmplPipeline' if variant=='gem_smpl' else 'GemXPipeline')
