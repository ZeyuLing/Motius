"""Compare original MotionGPT T5/VQ networks and inference with native parts."""
from __future__ import annotations
import argparse,json,os,random,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    def seed(value):
        random.seed(value);np.random.seed(value);torch.manual_seed(value)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from transformers import AutoConfig,T5ForConditionalGeneration
        from mGPT.archs.mgpt_lm import MLM
        from mGPT.archs.mgpt_vq import VQVae
        # The published MotionGPT state contains every T5 parameter. Redirect
        # only the original constructor's base-weight download to its identical
        # config; numerical source code, tokenizer and all forward calls remain original.
        original=T5ForConditionalGeneration.from_pretrained
        def from_config(model_path,*a,**kw):
            config=AutoConfig.from_pretrained(model_path,local_files_only=True);config.tie_word_embeddings=False
            return T5ForConditionalGeneration(config)
        T5ForConditionalGeneration.from_pretrained=from_config
        try:lm=MLM(str(args.artifact/'deps/flan-t5-base'),model_type='t5',stage='lm_instruct',motion_codebook_size=512,max_length=256)
        finally:T5ForConditionalGeneration.from_pretrained=original
        vq=VQVae(nfeats=263,quantizer='ema_reset',code_num=512,code_dim=512,output_emb_width=512,down_t=2,stride_t=2,width=512,depth=3,dilation_growth_rate=3,norm=None,activation='relu')
        state=torch.load(args.artifact/'motiongpt_s3_h3d.tar',map_location='cpu',weights_only=True,mmap=True)['state_dict']
        for prefix,model in [('lm',lm),('vae',vq)]:model.load_state_dict({k[len(prefix)+1:]:v for k,v in state.items() if k.startswith(prefix+'.')},strict=True)
        lm.language_model.encoder.embed_tokens=lm.language_model.shared;lm.language_model.decoder.embed_tokens=lm.language_model.shared
        lm.language_model.config.tie_word_embeddings=False
        lm.to(args.device).eval();vq.to(args.device).eval();decoder=lm
    else:
        from motius.models.motiongpt import MotionGPTBundle
        from motius.pipelines.motiongpt import MotionGPTPipeline
        from motius.trainers.motiongpt import MotionGPTTrainer
        bundle=MotionGPTBundle(artifact_dir=str(args.artifact),device=args.device,local_files_only=True)
        pipeline=MotionGPTPipeline(bundle);lm,vq=bundle.lm,bundle.vae;decoder=pipeline.decoder
    output={}
    mean=torch.from_numpy(np.load(args.artifact/'assets/meta/mean.npy')).to(args.device)
    std=torch.from_numpy(np.load(args.artifact/'assets/meta/std.npy')).to(args.device)
    for index,(lengths,with_len) in enumerate([([40],False),([60,40],True)]):
        text=['A person walks forward.','A person raises both arms.'][:len(lengths)]
        seed(31+index)
        logits=[]
        handle=lm.language_model.register_forward_hook(lambda module,inputs,value:logits.append(value.logits.detach().cpu()))
        with torch.no_grad():
            tokens=decoder.generate_conditional(texts=text,lengths=lengths,task='t2m',with_len=with_len,stage='test')
            motions={};physical={}
            for i,ids in enumerate(tokens):
                if ids.numel()>1:value=vq.decode(ids.clamp(0,511))[0].float()
                else:value=torch.zeros(max(1,lengths[i]),263,device=args.device)
                motions[str(i)]=value.cpu();physical[str(i)]=(value*std+mean).cpu()
        handle.remove()
        output[f'generation_{index}']={'tokens':{str(i):v.cpu() for i,v in enumerate(tokens)},'logits':{str(i):v for i,v in enumerate(logits)},'normalized':motions,'physical':physical}
    seed(113);motion=torch.randn(2,40,263,device=args.device)
    with torch.no_grad():
        codes,_=vq.encode(motion);reconstructed,commit,perplexity=vq(motion)
        captions=decoder.generate_conditional(motion_tokens=codes,lengths=[10,10],task='m2t',with_len=False,stage='test')
    output['reconstruction']={'codes':codes.cpu(),'motion':reconstructed.cpu(),'commitment':commit.cpu(),'perplexity':perplexity.cpu(),
        'captions':{str(i):torch.tensor(list(text.encode('utf8')),dtype=torch.uint8) for i,text in enumerate(captions)}}
    # Complete source instruction objective, including template draws, tokenization,
    # label masking, T5 dropout, and all trainable parameter gradients.
    seed(73);tokens=torch.randint(0,512,(2,8),device=args.device)
    tasks=[{'input':['Generate motion: <Caption_Placeholder>'],'output':['<Motion_Placeholder>']}]*2
    batch=dict(text=['walk forward','jump once'],motion_tokens=tokens,token_lens=[8,6],tasks=tasks)
    lm.train();lm.zero_grad(set_to_none=True);seed(79)
    if args.worker=='upstream':loss=lm(batch['text'],tokens,batch['token_lens'],tasks).loss
    else:
        trainer=MotionGPTTrainer(bundle);loss=trainer.train_step(batch)['loss']
    loss.backward()
    output['language_training']={'loss':loss.detach().cpu(),'gradients':{k:p.grad.detach().cpu() for k,p in lm.named_parameters() if p.grad is not None}}
    # Quantizer EMA initialization/update and VQ reconstruction losses are part of
    # the published stage-one graph, independent of the original Lightning app.
    lm.zero_grad(set_to_none=True);vq.train();vq.zero_grad(set_to_none=True);seed(89)
    if args.worker=='upstream':
        from omegaconf import OmegaConf
        from mGPT.losses.mgpt import GPTLosses
        config=OmegaConf.create({'LOSS':{'ABLATION':{'RECONS_LOSS':'l1_smooth'},'LAMBDA_FEATURE':1.,'LAMBDA_VELOCITY':.5,'LAMBDA_COMMIT':.02}})
        reconstructed,commit,perplexity=vq(motion)
        loss=GPTLosses(config,'vae',22).to(args.device).update({'m_ref':motion,'m_rst':reconstructed,'loss_commit':commit})
    else:
        trainer=MotionGPTTrainer(bundle,stage='vae');loss=trainer.train_step({'motion':motion})['loss']
    loss.backward()
    output['vq_training']={'loss':loss.detach().cpu(),'gradients':{k:p.grad.detach().cpu() for k,p in vq.named_parameters() if p.grad is not None},
        'updated_codebook':vq.quantizer.codebook.detach().cpu()}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='Original MotionGPT source and released FLAN-T5/VQ weights: text generation, motion captions, reconstruction, language/VQ objectives and gradients',
        reference_construction='Base pretrained T5 weight download redirected to identical artifact config; full released MotionGPT state loaded strictly; original math and tokenizer unchanged',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for folder in ['mGPT/archs','mGPT/losses'] for p in (args.upstream/folder).rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.rglob('*') if p.is_file() and '.cache' not in p.parts},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--device',args.device]
        for key in ['upstream','artifact','output_dir']:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    folders=['motius/models/motiongpt','motius/pipelines/motiongpt','motius/trainers/motiongpt','motius/motion/conditioning/motiongpt.py']
    report.update(complete=True,torch_version=torch.__version__,deterministic_cudnn=True,native_files={str(p.relative_to(root)):digest(p)
        for folder in folders for p in (list((root/folder).rglob('*.py')) if (root/folder).is_dir() else [root/folder])})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All MotionGPT source comparisons exact')

if __name__=='__main__':main()
