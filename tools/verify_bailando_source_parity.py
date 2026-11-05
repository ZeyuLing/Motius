"""Compare Bailando source networks, token sampling and decoded trajectories."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from verify_intergen_source_parity import digest


def namespace(value):
    if isinstance(value,dict):return SimpleNamespace(**{k:namespace(v) for k,v in value.items()})
    if isinstance(value,list):return [namespace(v) for v in value]
    return value


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    meta=json.loads((args.artifact/'bailando_config.json').read_text());cfg=meta['config']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from models.cross_cond_gpt2_ac import CrossCondGPT2AC
        from models.sep_vqvae_root import SepVQVAER
        model=CrossCondGPT2AC(namespace(cfg['gpt'])).to(args.device).eval()
        vq=SepVQVAER(namespace(cfg['vqvae'])).to(args.device).eval()
        model.load_state_dict(load_file(str(args.artifact/'gpt.safetensors')),strict=True)
        vq.load_state_dict(load_file(str(args.artifact/'vqvae.safetensors')),strict=True)
    else:
        from motius.pipelines.bailando import BailandoPipeline
        pipeline=BailandoPipeline.from_pretrained(str(args.artifact),device=args.device)
        model,vq=pipeline.bundle.gpt,pipeline.bundle.vqvae
    output={}
    for index,(batch,frames) in enumerate([(1,6),(1,35),(2,8)]):
        key=f'batch{batch}_music_frames{frames}'
        features=np.random.default_rng(71+index).normal(size=(batch,frames,438)).astype(np.float32)
        logits={};counter=[0]
        def capture(module,inputs,value):
            step=counter[0];counter[0]+=1
            logits[f'{step}_up']=value[0].detach().cpu()
            logits[f'{step}_down']=value[1].detach().cpu()
        handle=model.gpt_head.register_forward_hook(capture)
        with torch.no_grad():
            if args.worker=='upstream':
                music=torch.from_numpy(features).to(args.device)
                a,b=cfg['default_initial_codes']
                seed=(torch.full((batch,1),a,dtype=torch.long,device=args.device),
                      torch.full((batch,1),b,dtype=torch.long,device=args.device))
                codes=model.sample(seed,music[:,1:])
                motion=vq.decode(codes)
                # Original root velocity integration order (motion_vqvae.py).
                decoded=motion.clone();velocity=decoded[:,:,:3].clone()
                decoded[:,0,:3]=0
                for frame in range(1,decoded.shape[1]):
                    decoded[:,frame,:3]=decoded[:,frame-1,:3]+velocity[:,frame-1,:]
                relative=decoded.reshape(batch,decoded.shape[1],24,3)
                root=relative[:,:,:1].clone()
                joints=relative+root
                joints[:,:,0]=root[:,:,0]
                up,down=codes[0][0],codes[1][0]
            else:
                result=pipeline(music_features=features)
                motion=torch.from_numpy(result.model_motion)
                joints=torch.from_numpy(result.joints)
                up,down=torch.from_numpy(result.codes_up),torch.from_numpy(result.codes_down)
        handle.remove()
        output[key]={'logits':logits,'motion':motion.detach().cpu(),'joints':joints.detach().cpu(),
                     'codes_up':up.cpu(),'codes_down':down.cpu()}
    # Encoding the same physical motion also checks the initial-motion condition path.
    torch.manual_seed(89)
    motion=torch.randn(2,64,72,device=args.device)
    with torch.no_grad():indices=vq.encode(motion)
    output['encoding']={'up':indices[0][0].cpu(),'down':indices[1][0].cpu()}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def compare(a,b,path,rows):
    import torch
    if isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+'/'+k,rows)
    else:
        assert torch.isfinite(a).all() and torch.isfinite(b).all(),path
        rows.append(dict(tensor=path,shape=list(a.shape),bitwise_equal=torch.equal(a,b),
                         max_abs_error=(a.to(torch.float64)-b.to(torch.float64)).abs().max().item() if a.numel() else 0.))
        torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--worker',choices=['upstream','native'])
    args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='original Bailando repository and released checkpoint; precomputed music-feature inputs',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for p in (args.upstream/'models').rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.glob('*')
                        if p.is_file() and p.suffix in ['.safetensors','.json']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env=os.environ.copy();env['PYTHONPATH']=str(root)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),
            '--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--device',args.device,
            '--worker',kind],cwd=root,env=env,check=True)
    import torch
    a=torch.load(args.output_dir/'upstream.pt',weights_only=True);b=torch.load(args.output_dir/'native.pt',weights_only=True)
    try:compare(a,b,'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p)
        for folder in ['motius/models/bailando','motius/pipelines/bailando'] for p in (root/folder).rglob('*.py')})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All Bailando source comparisons exact')


if __name__=='__main__':main()
