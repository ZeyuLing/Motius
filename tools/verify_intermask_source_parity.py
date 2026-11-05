"""Independently compare InterMask source token generation and VQ decoding.

The source checkout is imported only by an isolated worker. Its unused CLIP visual
model is omitted during construction; the entire text tower is strictly loaded
from the released artifact. Sampling and decoding call the original source code.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from verify_intergen_source_parity import digest


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    artifact=args.artifact
    if args.worker=='upstream':
        import clip
        from clip.model import LayerNorm,Transformer
        from torch import nn
        from types import SimpleNamespace
        def text_tower(*a,**kw):
            return SimpleNamespace(token_embedding=nn.Embedding(49408,768),
                transformer=Transformer(768,12,12,attn_mask=torch.empty(77,77).fill_(float('-inf')).triu_(1)),
                positional_embedding=nn.Parameter(torch.empty(77,768)),ln_final=LayerNorm(768),dtype=torch.float32),None
        clip.load=text_tower
        sys.path.insert(0,str(args.upstream))
        from utils.get_opt import get_opt
        from models.mask_transformer.transformer import MaskTransformer
        from models.vq.model import RVQVAE
        vq_opt=get_opt(str(artifact/'vq_opt.txt'),torch.device(args.device))
        opt=get_opt(str(artifact/'transformer_opt.txt'),torch.device(args.device))
        opt.num_tokens=vq_opt.nb_code;opt.code_dim=vq_opt.code_dim
        vq=RVQVAE(vq_opt,12,vq_opt.nb_code,vq_opt.code_dim,vq_opt.code_dim,vq_opt.down_t,
            vq_opt.stride_t,vq_opt.width,vq_opt.depth,vq_opt.dilation_growth_rate,vq_opt.vq_act,vq_opt.vq_norm)
        trans=MaskTransformer(code_dim=opt.code_dim,cond_mode='text',latent_dim=opt.latent_dim,
            ff_size=opt.ff_size,num_layers=opt.n_layers,num_heads=opt.n_heads,dropout=opt.dropout,
            clip_dim=768,cond_drop_prob=opt.cond_drop_prob,clip_version='ViT-L/14@336px',opt=opt)
        for model,name in [(vq,'vq_model'),(trans,'transformer')]:
            missing,unexpected=model.load_state_dict(load_file(str(artifact/f'{name}.safetensors')),strict=False)
            allowed=all(key.startswith(('decoder.conv','decoder.resnets')) for key in missing) if name=='vq_model' else not missing
            if not allowed or unexpected: raise RuntimeError((name,missing,unexpected))
            model.to(args.device).eval()
        sample=lambda *a,**kw:trans.generate(*a,**kw)
        reaction=lambda *a,**kw:trans.generate_reaction(*a,**kw)
        decode=lambda tokens:vq.forward_decoder(tokens)
    else:
        from motius.models.intermask import InterMaskBundle
        from motius.pipelines.intermask.sampling import generate,generate_reaction
        from motius.motion.representation.intermask import unpack_motion
        bundle=InterMaskBundle.from_pretrained(str(artifact),device=args.device)
        vq,trans=bundle.load_model()
        sample=lambda *a,**kw:generate(trans,*a,**kw)
        reaction=lambda *a,**kw:generate_reaction(trans,*a,**kw)
        decode=lambda tokens:unpack_motion(bundle.decode_tokens(tokens),'interhuman')
    mean=torch.from_numpy(np.load(artifact/'stats/global_mean.npy')).float()
    std=torch.from_numpy(np.load(artifact/'stats/global_std.npy')).float()
    output={}
    for seed,frames,prompts,gsample in [(13,32,['two people shake hands'],False),
            (19,60,['two people hug','two people kick each other'],False),
            (27,32,['two people shake hands'],True)]:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        key=f'seed{seed}_frames{frames}_batch{len(prompts)}_gumbel{gsample}'
        collected=[]
        handle=trans.output_process.register_forward_hook(lambda m,a,value:collected.append(value.detach().cpu()))
        with torch.no_grad():
            ids=sample(prompts,torch.full((len(prompts),),frames//4,device=args.device),20,2.,
                       topk_filter_thres=.9,temperature=1.,gsample=gsample)
            first,second=ids.chunk(2,dim=1)
            decoded=torch.cat([decode(first.unsqueeze(-1)),decode(second.unsqueeze(-1))],dim=-1)
            decoded=decoded.reshape(len(prompts),frames,2,262).detach().cpu()*std+mean
        handle.remove()
        output[key]={'ids':ids.cpu(),'motion':decoded,'logits':torch.stack(collected)}
    # Reaction generation is a distinct sampling branch and must preserve the fixed person's tokens.
    torch.manual_seed(41)
    prompts=['two people shake hands'];lengths=torch.tensor([8],device=args.device)
    first=torch.randint(0,trans.opt.num_tokens,(1,40),device=args.device)
    with torch.no_grad():ids=reaction(prompts,first,lengths,20,2.)
    output['reaction']={'ids':ids.cpu()}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--artifact',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--worker',choices=['upstream','native'])
    args=parser.parse_args()
    args.upstream=args.upstream.resolve();args.artifact=args.artifact.resolve();args.output_dir=args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='original source with released checkpoint',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for folder in ['models','utils','data']
            for p in (args.upstream/folder).rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.rglob('*')
            if p.is_file() and p.suffix in ['.safetensors','.npy','.txt'] and '.cache' not in p.parts},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env=os.environ.copy();env['PYTHONPATH']=str(root)
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),
            '--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--device',args.device,
            '--worker',kind],cwd=root,env=env,check=True)
    import torch
    a=torch.load(args.output_dir/'upstream.pt',weights_only=True)
    b=torch.load(args.output_dir/'native.pt',weights_only=True)
    for case in a:
        for key in a[case]:
            x,y=a[case][key],b[case][key]
            error=(x-y).abs().max().item()
            report['tests'].append(dict(case=case,tensor=key,shape=list(x.shape),bitwise_equal=torch.equal(x,y),max_abs_error=error))
            path.write_text(json.dumps(report,indent=2)+'\n')
            torch.testing.assert_close(x,y,rtol=0,atol=0)
    report.update(complete=True,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p)
        for folder in ['motius/models/intermask','motius/pipelines/intermask','motius/motion/representation']
        for p in (root/folder).rglob('*.py') if 'representation' not in folder or p.name=='intermask.py'})
    path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['tests'],indent=2))


if __name__=='__main__':main()
