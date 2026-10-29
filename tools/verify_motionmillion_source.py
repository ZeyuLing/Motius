"""Real released MotionMillion 7B/T5/FSQ graphs against the original repository."""
from pathlib import Path
import argparse,ast,json,os,subprocess,sys
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import torch,numpy as np
    from torch.distributions import Categorical
    from safetensors.torch import load_file
    from verify_motionstreamer_source import hashes
    from verify_prompthmr_video_parity import tensors
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    config=json.loads((args.artifact/'mm_config.json').read_text())['config']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from models.vqvae import HumanVQVAE
        from models.lit_llama.model_hf import LLaMAHF,LLaMAHFConfig
        from transformers import T5EncoderModel,T5Tokenizer
        from utils.losses import ReConsLoss
        c=config['vqvae'];vq=HumanVQVAE(NS(dataname='t2m',causal=False,quantizer='FSQ',nb_code=c['nb_code']),c['nb_code'],c['code_dim'],c['output_emb_width'],c['down_t'],c['stride_t'],c['width'],c['depth'],c['dilation_growth_rate'],c['activation'],c['norm'],c['kernel_size'],c['use_patcher'],c['patch_size'],c['patch_method'])
        vq.load_state_dict(load_file(str(args.artifact/'fsq.safetensors')),strict=True);vq.cuda().eval()
        c=config['ar'];architecture=LLaMAHFConfig.from_name(c['config_name']);architecture.block_size=c['block_size'];architecture.vocab_size=vq.vqvae.quantizer.codebook_size+2;architecture.clip_dim=c['clip_dim'];architecture.tie_weights=c['tie_weights']
        ar=LLaMAHF(architecture);ar.load_state_dict(load_file(str(args.artifact/'ar.safetensors')),strict=True);ar.cuda().eval()
        text=T5EncoderModel.from_pretrained(args.artifact/'text_encoder',local_files_only=True).cuda().eval().requires_grad_(False)
        tokenizer=T5Tokenizer.from_pretrained(args.artifact/'text_encoder',local_files_only=True)
        def encode(captions):
            inputs=tokenizer(captions,padding=True,truncation=True,return_tensors='pt')
            mask=inputs.attention_mask.cuda()
            features=text(input_ids=inputs.input_ids.cuda(),attention_mask=mask,output_hidden_states=False).last_hidden_state
            if features.shape[1]>150:features=features[:,:150];mask=mask[:,:150]
            return features,mask
    else:
        from motius.pipelines.motionmillion import MotionMillionPipeline
        from motius.motion.conditioning.motionmillion_text import prepare_motionmillion_text
        from motius.pipelines.motionmillion.sampling import sample_tokens
        from motius.trainers.motionmillion import MotionMillionTrainer
        pipe=MotionMillionPipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':'cuda'})
        bundle=pipe.bundle;vq,ar,text=bundle.vqvae,bundle.ar,bundle.text_model
        def encode(captions):return bundle.encode_text(prepare_motionmillion_text(bundle.text_model_name,captions))
    output={'weights':{'vq':hashes(vq),'ar':hashes(ar),'text':hashes(text)},'generation':{},'training':{}}
    stats=[torch.from_numpy(np.load(args.artifact/f'{name}.npy')).cuda() for name in ['mean','std']]
    for index,captions in enumerate([['A person walks forward.'],['A person raises both arms.','']]):
        torch.manual_seed(61+index)
        with torch.no_grad():
            features,mask=encode(captions);generated=[];tokens=[]
            for i in range(len(captions)):
                codes=ar.sample(features[i:i+1],mask[i:i+1],False) if args.worker=='upstream' else sample_tokens(ar,features[i:i+1],mask[i:i+1],False)
                normalized=vq.forward_decoder(codes);generated.append(normalized[0]*stats[1]+stats[0]);tokens.append(codes)
            output['generation'][str(index)]=tensors(dict(features=features,mask=mask,tokens=tokens,motion=generated,rng=torch.cuda.get_rng_state()))
            if args.worker=='native':
                torch.manual_seed(61+index)
                public=pipe.infer_t2m(captions)
                for expected,actual in zip(generated,public):torch.testing.assert_close(expected.cpu(),torch.from_numpy(actual),rtol=0,atol=0)
        print(args.worker,'full 50-token generation',index,flush=True)
    # Actual source corruption function and source shifted, padding-masked CE.
    if args.worker=='upstream':
        source=args.upstream/'train_t2m_llama.py';tree=ast.parse(source.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='train_one_iter')
        namespace={'torch':torch,'np':np,'comp_device':torch.device('cuda'),'args':NS(pkeep=.7,nb_code=vq.vqvae.quantizer.codebook_size)}
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    torch.manual_seed(211);np.random.seed(211)
    size=vq.vqvae.quantizer.codebook_size
    target=torch.randint(0,size,(1,6),device='cuda');target[0,-2]=size;target[0,-1]=size+1
    with torch.no_grad():features,mask=encode(['A person walks forward.'])
    ar.train().zero_grad(set_to_none=True)
    if args.worker=='upstream':
        prediction,labels=namespace['train_one_iter'](features,target,torch.tensor([4],device='cuda'),mask,ar)
        prediction=prediction[...,:-1,:].contiguous();labels=labels[...,1:].contiguous().to(torch.int64)
        loss=torch.nn.CrossEntropyLoss(ignore_index=size+1)(prediction.view(-1,prediction.shape[-1]),labels.view(-1))
        selected=Categorical(torch.softmax(prediction.float(),dim=-1)).sample();valid=labels!=size+1;accuracy=((selected==labels)&valid).sum()/valid.sum()
    else:
        trainer=MotionMillionTrainer(bundle,pkeep=.7);bundle.train()
        result=trainer.train_step(dict(tokens=target,text_features=features,text_mask=mask));loss,accuracy=result['loss'],result['motionmillion_accuracy']
    loss.backward()
    output['training']['ar']=tensors(dict(loss=loss,accuracy=accuracy,gradients=hashes(ar,gradients=True),rng=torch.cuda.get_rng_state()))
    ar.zero_grad(set_to_none=True)
    print(args.worker,'7B loss and all gradient hashes saved',flush=True)
    torch.manual_seed(712)
    motion=torch.sin(torch.arange(2*40*272,device='cuda').reshape(2,40,272)*.013)
    vq.train().zero_grad(set_to_none=True)
    if args.worker=='upstream':
        prediction,commitment,*_=vq(motion);objective=ReConsLoss('l1_smooth',22)
        loss=objective(prediction,motion)+.02*commitment+.1*objective.forward_vel(prediction,motion)
        loss=loss+.3*objective.forward_acc(prediction,motion)+.2*objective.forward_acc_vel(prediction,motion)
    else:
        trainer=MotionMillionTrainer(bundle,stage='vqvae',acc_loss=.3,acc_vel_loss=.2);bundle.train()
        loss=trainer.train_step(dict(motion=motion))['loss']
    loss.backward()
    output['training']['vq']=tensors(dict(loss=loss,gradients=hashes(vq,gradients=True),rng=torch.cuda.get_rng_state()))
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--reuse-results',action='store_true')
    args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    import torch
    from verify_prompthmr_video_parity import compare_full
    report={'complete':False,'scope':'Original MotionMillion 7B, real Flan-T5-XL, FSQ and complete default 50-token greedy generation, full neural hashes, original AR corruption/shifted CE and FSQ reconstruction/velocity/acceleration objectives with all gradient hashes and RNG. Native public generation checked against the same original graph. Controlled FP32/math-attention/deterministic-convolution runtime; dataset, optimizer/distributed protocol, other releases and optional KV cache excluded.',
            'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),'tests':[]}
    path=args.output_dir/'source_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.reuse_results else ['upstream','native']):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--worker',kind],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'},check=True)
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n');print('MotionMillion original real 7B/T5/FSQ inference and training exact')

if __name__=='__main__':main()
