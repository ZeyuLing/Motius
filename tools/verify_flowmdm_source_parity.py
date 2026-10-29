"""Compare original FlowMDM source with native HML/BABEL samplers and training."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from argparse import Namespace
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    cfg=json.loads((args.artifact/'args.json').read_text());dataset=cfg['dataset']
    cfg.update(model_path=str(args.artifact/('model001300000.pt' if dataset=='babel' else 'model000500000.pt')),
        seed=42,guidance_param=1.5 if dataset=='babel' else 2.5,bpe_denoising_step=125 if dataset=='babel' else 60,use_chunked_att=False)
    cfg.setdefault('lambda_vel_rcxyz',0.)
    torch.manual_seed(42)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        import clip
        def local_clip(*a,**kw):
            return clip.model.build_model(dict(load_file(str(args.artifact/'clip.safetensors')))).float(),None
        clip.load=local_clip
        import importlib
        flow=importlib.import_module('model.FlowMDM')
        # The original constructor creates an unused SMPL visualizer. It never
        # participates in the released HML/BABEL neural forward or any case here.
        # No neural, diffusion, loss or motion-representation operation is replaced.
        class UnusedBody:
            def __init__(self,*a,**kw):self.smpl_model=torch.nn.Identity()
            def __call__(self,*a,**kw):raise AssertionError('body visualizer entered the numerical comparison')
        flow.Rotation2xyz=UnusedBody
        from utils.model_util import load_model
        from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM
        guided,diffusion=load_model(Namespace(**cfg),torch.device(args.device))
        model=vars(guided)['model'] if 'model' in vars(guided) else guided
        sampler=DiffusionWrapper_FlowMDM(Namespace(**cfg),diffusion,guided)
    else:
        from motius.pipelines.flowmdm import FlowMDMPipeline
        from motius.trainers.flowmdm import FlowMDMTrainer
        pipeline=FlowMDMPipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':args.device})
        model=pipeline.bundle.network;sampler=pipeline.sampler;diffusion=sampler.diffusion
    output={}
    for index,(captions,lengths) in enumerate([(['A person walks forward.'],[40]),(['A person walks forward.','The person raises both arms.'],[40,32])]):
        torch.manual_seed(47+index);random.seed(47+index)
        with torch.no_grad():
            if args.worker=='upstream':
                for key in ['emb_hash','emb_forcemask_hash']:
                    if hasattr(model,key):delattr(model,key)
                normalized=sampler.p_sample_loop(model_kwargs={'y':dict(text=captions,lengths=torch.tensor(lengths,device=args.device))},clip_denoised=False)
                normalized=normalized[0,:,0].T.contiguous()
            else:normalized=pipeline._sample_sequence(captions,lengths)
            if dataset=='babel':
                stats=[torch.load(args.artifact/f'rfeats_{n}.pt',map_location='cpu',weights_only=True).to(args.device) for n in ['mean','std']]
            else:stats=[torch.from_numpy(np.load(args.artifact/f'{n}.npy')).to(args.device) for n in ['Mean','Std']]
            physical=normalized*stats[1]+stats[0]
        output[f'{dataset}_segments{len(captions)}']={'normalized':normalized.cpu(),'physical':physical.cpu()}
    # Use the released architecture, weights, masks, noise and source objective.
    torch.manual_seed(91);random.seed(91)
    motion=torch.randn(2,model.njoints,1,40,device=args.device)*.01
    noise=torch.randn_like(motion);timesteps=torch.tensor([10,950],device=args.device)
    lengths=torch.tensor([40,32],device=args.device);text=['walk forward','raise both arms']
    mask=(torch.arange(40,device=args.device)[None,:]<lengths[:,None])[:,None,None,:]
    for key in ['emb_hash','emb_forcemask_hash']:
        if hasattr(model,key):delattr(model,key)
    model.train();model.clip_model.eval();model.zero_grad(set_to_none=True)
    torch.manual_seed(93);random.seed(93)
    if args.worker=='upstream':
        terms=diffusion.training_losses(model,motion,timesteps,model_kwargs={'y':dict(text=text,mask=mask,lengths=lengths)},noise=noise)
        loss=terms['loss'].mean()
    else:
        trainer=FlowMDMTrainer(pipeline.bundle)
        loss=trainer.train_step(dict(motion=motion,motion_lens=lengths,text=text,noise=noise,timesteps=timesteps))['loss']
    loss.backward()
    output['training']={'loss':loss.detach().cpu(),'gradients':{k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None}}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for k in ['upstream','artifact','output-dir']:parser.add_argument('--'+k,type=Path,required=True)
    parser.add_argument('--device',default='cuda');parser.add_argument('--worker',choices=['upstream','native'])
    args=parser.parse_args()
    for k in ['upstream','artifact','output_dir']:setattr(args,k,getattr(args,k).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='original FlowMDM source, released weights, complete 1000-step sampling and objective gradients',
        excluded_constructor='unused SMPL visualizer; calling it raises an error in the reference worker',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for folder in ['model','diffusion'] for p in (args.upstream/folder).rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.glob('*') if p.is_file() and p.suffix in ['.safetensors','.json','.npy','.pt']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env=os.environ.copy();env['PYTHONPATH']=str(root)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--artifact',str(args.artifact),
            '--output-dir',str(args.output_dir),'--device',args.device,'--worker',kind],cwd=root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p)
        for folder in ['motius/models/flowmdm','motius/pipelines/flowmdm','motius/trainers/flowmdm','motius/diffusion/flowmdm'] for p in (root/folder).rglob('*.py')})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All FlowMDM source comparisons exact')

if __name__=='__main__':main()
