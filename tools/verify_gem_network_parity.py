"""Compare independent original GEM neural graphs against native neural modules.

Full body postprocessing and public video inference are separate acceptance gates.
"""
from __future__ import annotations
import argparse,ast,copy,importlib,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def source_conditioner(upstream,cfg,variant):
    import torch
    from torch import nn
    from timm.models.vision_transformer import Mlp
    layer=importlib.import_module('gem.network.base_arch.transformer.layer')
    tree=ast.parse((upstream/'gem/gem.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GEM')
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    start=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='f_condition_dim' for t in n.targets))
    end=next(i for i,n in enumerate(init.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='condition_source' for t in n.targets))
    final=next(n for n in init.body if isinstance(n,ast.If) and ast.unparse(n.test)=='self.model_cfg.use_cond_exists_as_input')
    source=ast.parse('def construct(self,model_cfg,obs_num_joints):\n    pass\n').body[0]
    source.body=copy.deepcopy(init.body[start:end+1])+[copy.deepcopy(final)]
    condition=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='create_condition_mask')
    scope=dict(torch=torch,nn=nn,Mlp=Mlp,BasicBlock=layer.BasicBlock,zero_module=layer.zero_module,
               length_to_mask=importlib.import_module('gem.utils.net_utils').length_to_mask)
    if variant=='gem_x':
        dataset=ast.parse((upstream/'gem/datasets/metrosim/metrosim.py').read_text())
        assignment=next(n for n in dataset.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='nvskel77to33_idx' for t in n.targets))
        exec(compile(ast.Module(body=[assignment],type_ignores=[]),str(upstream),'exec'),scope)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[source,condition],type_ignores=[])),str(upstream/'gem/gem.py'),'exec'),scope)
    owner=nn.Module();owner.model_cfg=cfg.model_config
    owner.pipeline=SimpleNamespace(args=cfg.pipeline_args,args_denoiser3d=cfg.denoiser)
    owner.no_exist_keys=['obs','observed_motion_3d','multi_text_embed']
    owner.music_mask_prob=cfg.model_config.get('music_mask_prob',0.);owner.audio_mask_prob=cfg.model_config.get('audio_mask_prob',0.)
    owner.remove_fingers=cfg.pipeline_args.get('remove_fingers',False)
    scope['construct'](owner,cfg.model_config,33 if variant=='gem_x' else 17)
    return owner,scope['create_condition_mask']


def worker(args):
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    root=Path(__file__).resolve().parents[1];variant=args.variant
    cfg=OmegaConf.create(json.loads((root/'motius/configuration'/(variant+'.json')).read_text()))
    checkpoint=args.artifact/'inputs/pretrained'/('gem_smpl.ckpt' if variant=='gem_smpl' else 'gem_soma.ckpt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)['state_dict']
    parameters=OmegaConf.to_container(cfg.denoiser,resolve=True);parameters.pop('_target_')
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream));package='gem.network.gem_denoiser'
        conditioner,prepare=source_conditioner(args.upstream,cfg,variant)
    else:
        package='motius.models.'+variant+'.denoiser'
        module=importlib.import_module('motius.models.'+variant+'.conditioning')
        conditioner=module.ConditionEncoder(cfg.pipeline_args,cfg.model_config,cfg.denoiser,33 if variant=='gem_x' else 17)
        if variant=='gem_x':parameters.update(fixed_scale_mean=state['pipeline.denoiser3d.denoiser.scale_mean'],fixed_scale_components=state['pipeline.denoiser3d.denoiser.scale_comps'])
    network=importlib.import_module(package).NetworkEncoderRoPE(**parameters)
    original_condition={k:v for k,v in state.items() if not k.startswith('pipeline.denoiser3d.denoiser.')}
    conditioner.load_state_dict(original_condition,strict=True)
    network.load_state_dict({k.removeprefix('pipeline.denoiser3d.denoiser.'):v for k,v in state.items() if k.startswith('pipeline.denoiser3d.denoiser.')},strict=True)
    conditioner.to(args.device).eval();network.to(args.device).eval()
    dimensions=151 if variant=='gem_smpl' else 585
    indices=({'body_pose':(0,126),'betas':(126,136),'global_orient':(136,142),'global_orient_gv':(142,148),'local_transl_vel':(148,151)} if variant=='gem_smpl' else {'body_pose':(0,456),'identity_coeffs':(456,501),'scale_params':(501,570),'global_orient':(570,576),'global_orient_gv':(576,582),'local_transl_vel':(582,585)})
    output={}
    for case,(batch_size,frames,training) in enumerate([(1,16,False),(2,21,False),(2,12,True)]):
        conditioner.train(training);network.train(training)
        torch.manual_seed(151+case)
        joints=17 if variant=='gem_smpl' else 77
        lengths=torch.tensor([frames,frames-5][:batch_size],device=args.device)
        obs=torch.randn(batch_size,frames,joints,3,device=args.device);obs[...,2]=torch.rand_like(obs[...,2])
        mask=lambda:torch.rand(batch_size,frames,device=args.device)>.3
        batch=dict(B=batch_size,L=frames,device=torch.device(args.device),length=lengths,has_text=torch.ones(batch_size,dtype=torch.bool,device=args.device),
            condition_mask=dict(has_img_mask=mask(),has_2d_mask=mask(),has_cam_mask=mask(),has_audio_mask=mask(),has_music_mask=mask(),j2d_visible_mask=obs[...,2]>.5),
            mask={'2d_only':torch.zeros(batch_size,dtype=torch.bool,device=args.device)},obs=obs,
            f_cliffcam=torch.randn(batch_size,frames,3,device=args.device),f_cam_angvel=torch.randn(batch_size,frames,6,device=args.device),
            f_imgseq=torch.randn(batch_size,frames,1024,device=args.device),target_x=torch.randn(batch_size,frames,dimensions,device=args.device),
            encoded_text=torch.randn(batch_size,7,cfg.denoiser.encoded_text_dim,device=args.device),music_embed=torch.randn(batch_size,frames,35,device=args.device))
        torch.manual_seed(131+case)
        if args.worker=='upstream':
            result=prepare(conditioner,batch,cfg.model_config.condition_mask,'regression',training)
        else:
            from motius.models.gem_smpl.positions import length_to_mask
            selector=importlib.import_module('motius.motion.conditioning.'+variant).create_condition_mask
            owner=SimpleNamespace(args=cfg.pipeline_args,pipeline=SimpleNamespace(args=cfg.pipeline_args),condition_source=conditioner.condition_source,music_mask_prob=cfg.model_config.get('music_mask_prob',0.),audio_mask_prob=cfg.model_config.get('audio_mask_prob',0.))
            existence,visible=selector(owner,batch,cfg.model_config.condition_mask,'regression',training)
            conditional,unconditional,empty=conditioner(batch,existence,visible)
            result=batch;result.update(f_cond=conditional,f_uncond=unconditional,f_empty=empty)
            if result.get('text_mask') is not None:result['encoded_text']=result['encoded_text']*(1-result['text_mask'][:,None,None].float())
            result['motion']=result['target_x']*length_to_mask(lengths,frames)[...,None]
        noisy=torch.randn_like(result['motion']);timesteps=torch.randint(0,1000,(batch_size,),device=args.device)
        y=dict(f_cond=result['f_cond'],length=lengths,encoded_text=result['encoded_text'])
        prediction=network(noisy,timesteps,y=y,inputs=result,sample_indices_dict=indices)
        output[str(case)]={k:v.detach().cpu() for k,v in prediction.items() if isinstance(v,torch.Tensor)}
        output[str(case)].update({k:result[k].detach().cpu() for k in ['f_cond','f_uncond','f_empty','encoded_text','motion']})
        if training:
            loss=sum(v.square().mean() for v in prediction.values() if isinstance(v,torch.Tensor))
            loss.backward();output['gradient_loss']=loss.detach().cpu()
            output['gradients']={prefix+'.'+k:p.grad.detach().cpu() for prefix,model in [('conditioner',conditioner),('denoiser',network)] for k,p in model.named_parameters() if p.grad is not None}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--variant',choices=['gem_smpl','gem_x'],required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'network_parity.json'
    report=dict(complete=False,comparison='Original GEM network and complete stochastic conditioning graph versus native neural closure',scope='Neural graphs and their gradients only; source training objectives, body postprocessing and public video still require separate certificates',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),checkpoint_sha256=digest(args.artifact/'inputs/pretrained'/('gem_smpl.ckpt' if args.variant=='gem_smpl' else 'gem_soma.ckpt')),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GEM neural comparisons exact',args.variant)

if __name__=='__main__':main()
