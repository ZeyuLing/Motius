"""Compare independent released MoMask source inference and training graphs."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    meta=json.loads((args.artifact/'momask_config.json').read_text());cfg=meta['config']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        import clip
        def local_clip(*a,**kw):
            return clip.model.build_model(dict(load_file(str(args.artifact/'clip.safetensors')))).float(),None
        clip.load=local_clip
        from models.mask_transformer.transformer import MaskTransformer,ResidualTransformer
        from models.vq.model import RVQVAE
        common=dict(code_dim=cfg['vq']['code_dim'],cond_mode='text',clip_version='ViT-B/32',
            opt=SimpleNamespace(num_tokens=cfg['vq']['nb_code'],num_quantizers=cfg['vq']['num_quantizers'],device=torch.device(args.device)))
        def config(c):
            return dict(latent_dim=c['latent_dim'],ff_size=c['ff_size'],num_layers=c['n_layers'],num_heads=c['n_heads'],
                dropout=c['dropout'],clip_dim=c['clip_dim'],cond_drop_prob=c['cond_drop_prob'])
        base=MaskTransformer(**common,**config(cfg['t2m']))
        residual=ResidualTransformer(**common,**config(cfg['res']),shared_codebook=cfg['vq']['shared_codebook'],share_weight=cfg['res']['share_weight'])
        c=cfg['vq'];vq=RVQVAE(SimpleNamespace(**c),263,c['nb_code'],c['code_dim'],c['output_emb_width'],c['down_t'],c['stride_t'],c['width'],c['depth'],c['dilation_growth_rate'],c['vq_act'],c['vq_norm'])
        for model,filename in [(base,'t2m_trans'),(residual,'res_trans'),(vq,'vq')]:
            missing,unexpected=model.load_state_dict(load_file(str(args.artifact/f'{filename}.safetensors')),strict=False)
            assert not unexpected and all(k.startswith('clip_model.') for k in missing),(missing,unexpected)
            model.to(args.device).eval()
    else:
        from motius.models.momask import MoMaskBundle
        from motius.pipelines.momask.decoding import base_generate,residual_generate,base_edit
        from motius.trainers.momask.objective import compute_base_objective,compute_residual_objective
        bundle=MoMaskBundle.from_pretrained(str(args.artifact),device=args.device)
        base,residual,vq=bundle.t2m_transformer,bundle.res_transformer,bundle.vq_model
    output={}
    for index,(lengths,gumbel) in enumerate([([10],False),([15,10],True)]):
        key=f'batch{len(lengths)}_gumbel{gumbel}';lens=torch.tensor(lengths,device=args.device)
        text=['A person walks forward.','A person raises both arms.'][:len(lengths)]
        torch.manual_seed(31+index)
        with torch.no_grad():
            if args.worker=='upstream':
                tokens=base.generate(text,lens,10,4,gsample=gumbel)
                codes=residual.generate(tokens,text,lens,cond_scale=5.)
            else:
                tokens=base_generate(base,text,lens,10,4,gsample=gumbel)
                codes=residual_generate(residual,tokens,text,lens,cond_scale=5.)
            motion=vq.forward_decoder(codes)
            stats=[torch.from_numpy(np.load(args.artifact/f'{n}.npy')).to(args.device) for n in ['Mean','Std']]
            physical=motion*stats[1]+stats[0]
            padded=tokens.clamp_min(0);mask=torch.zeros_like(padded,dtype=torch.bool);mask[:,2:6]=True
            if args.worker=='upstream':edited=base.edit(text,padded,lens,5,4,edit_mask=mask)
            else:edited=base_edit(base,text,padded,lens,5,4,edit_mask=mask)
        output[key]={k:v.cpu() for k,v in dict(tokens=tokens,codes=codes,normalized=motion,physical=physical,edited=edited).items()}
    # Released neural parameters and actual source masking, text drop and dropout.
    for index,(name,model) in enumerate([('base',base),('residual',residual)]):
        torch.manual_seed(93)
        ids=torch.randint(0,cfg['vq']['nb_code'],(2,8) if name=='base' else (2,8,cfg['vq']['num_quantizers']),device=args.device)
        lens=torch.tensor([8,6],device=args.device);text=['walk forward','jump once']
        model.train();model.clip_model.eval();model.zero_grad(set_to_none=True)
        torch.manual_seed(79+index)
        if args.worker=='upstream':loss,pred,accuracy=model(ids,text,lens)
        else:
            fn=compute_base_objective if name=='base' else compute_residual_objective
            loss,pred,accuracy=fn(model,model,ids,text,lens)
        loss.backward()
        output[f'{name}_training']={'loss':loss.detach().cpu(),'prediction':pred.cpu(),
            'accuracy':torch.as_tensor(accuracy),'gradients':{k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None}}
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
    report=dict(complete=False,comparison='original MoMask source and released weights; inference and objective gradients',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for p in (args.upstream/'models').rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.glob('*') if p.is_file() and p.suffix in ['.safetensors','.json','.npy']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env=os.environ.copy();env['PYTHONPATH']=str(root)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--artifact',str(args.artifact),
            '--output-dir',str(args.output_dir),'--device',args.device,'--worker',kind],cwd=root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p)
        for folder in ['motius/models/momask','motius/pipelines/momask','motius/trainers/momask'] for p in (root/folder).rglob('*.py')})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All MoMask source comparisons exact')

if __name__=='__main__':main()
