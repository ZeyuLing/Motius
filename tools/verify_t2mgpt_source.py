"""Independent T2M-GPT released-weight inference, token CE and VQ objectives."""
from pathlib import Path
import argparse,ast,json,os,subprocess,sys
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import numpy as np,torch,clip
    from safetensors.torch import load_file
    from torch.distributions import Categorical
    from verify_motionstreamer_source import hashes,digest
    from verify_prompthmr_video_parity import tensors
    if os.environ.get("MOTIUS_REFERENCE_HOST"):
        assert os.uname().nodename == os.environ["MOTIUS_REFERENCE_HOST"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    config=json.loads((args.artifact/'t2mgpt_config.json').read_text())['config']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from models.vqvae import HumanVQVAE
        from models.t2m_trans import Text2Motion_Transformer
        from utils.losses import ReConsLoss
        cfg=config['vqvae'];vq=HumanVQVAE(NS(**cfg),cfg['nb_code'],cfg['code_dim'],cfg['output_emb_width'],cfg['down_t'],cfg['stride_t'],cfg['width'],cfg['depth'],cfg['dilation_growth_rate'],cfg['vq_act'],cfg['norm'])
        cfg=config['gpt'];gpt=Text2Motion_Transformer(num_vq=config['vqvae']['nb_code'],embed_dim=cfg['embed_dim_gpt'],clip_dim=cfg['clip_dim'],block_size=cfg['block_size'],num_layers=cfg['num_layers'],n_head=cfg['n_head_gpt'],drop_out_rate=cfg['drop_out_rate'],fc_rate=cfg['ff_rate'])
        for model,name in [(vq,'vq'),(gpt,'gpt')]:model.load_state_dict(load_file(str(args.artifact/f'{name}.safetensors')),strict=True);model.cuda().eval()
        text=clip.model.build_model(dict(load_file(str(args.artifact/'clip.safetensors')))).cuda().eval().requires_grad_(False)
    else:
        from motius.pipelines.t2mgpt import T2MGPTPipeline
        from motius.trainers.t2mgpt import T2MGPTTrainer
        from motius.motion.conditioning.clip_text import encode_clip_text
        from motius.pipelines.t2mgpt.sampling import sample_tokens
        pipe=T2MGPTPipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':'cuda'})
        bundle=pipe.bundle;vq,gpt,text=bundle.vqvae,bundle.gpt,bundle.clip_model
    output={'neural':{'vq':hashes(vq),'gpt':hashes(gpt),'clip':hashes(text)},'inference':{},'training':{}}
    captions=['A person walks forward.','A person raises both arms.']
    stats=[torch.from_numpy(np.load(args.artifact/f'{n}.npy')).cuda() for n in ['Mean','Std']]
    for index,categorical in enumerate([False,True]):
        torch.manual_seed(71+index)
        with torch.no_grad():
            features=text.encode_text(clip.tokenize(captions,truncate=True).cuda()).float() if args.worker=='upstream' else encode_clip_text(bundle,captions)
            motions=[];codes=[]
            for feature in features:
                if args.worker=='upstream':
                    try:tokens=gpt.sample(feature[None],categorical)
                    except UnboundLocalError:tokens=torch.ones(1,1,device='cuda',dtype=torch.long)
                else:tokens=sample_tokens(gpt,feature[None],categorical)
                motion=vq.forward_decoder(tokens)
                codes.append(tokens);motions.append(motion[0]*stats[1]+stats[0])
            output['inference'][str(index)]=tensors({'features':features,'codes':codes,'physical':motions,'rng':torch.cuda.get_rng_state()})
            if args.worker=='native':
                torch.manual_seed(71+index)
                public=pipe.infer_t2m(captions,if_categorial=categorical)
                for expected,actual in zip(motions,public):torch.testing.assert_close(expected.cpu(),torch.from_numpy(actual),rtol=0,atol=0)
    # Original module-level training statements, not a reimplemented reference objective.
    if args.worker=='upstream':
        source=args.upstream/'train_t2m_trans.py';tree=ast.parse(source.read_text())
        loop=next(n for n in tree.body if isinstance(n,ast.While))
        body=loop.body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='input_index' for t in n.targets))
        stop=next(i for i,n in enumerate(body) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='optimizer.zero_grad')
        original_ce=compile(ast.Module(body=body[start:stop],type_ignores=[]),str(source),'exec')
    for index,(pkeep,maximum) in enumerate([(.5,False),(-1.,True)]):
        torch.manual_seed(417+index);np.random.seed(417+index)
        target=torch.randint(0,512,(2,11),device='cuda');lengths=torch.tensor([10,7],device='cuda');target[0,10]=512;target[1,7]=512;target[1,8:]=513
        gpt.train().zero_grad(set_to_none=True)
        if args.worker=='upstream':
            features=text.encode_text(clip.tokenize(captions,truncate=True).cuda()).float()
            namespace={'torch':torch,'np':np,'Categorical':Categorical,'target':target,'m_tokens_len':lengths,'bs':2,'right_num':0,
                       'args':NS(pkeep=pkeep,nb_code=512,if_maxtest=maximum),'feat_clip_text':features,'trans_encoder':gpt,'loss_ce':torch.nn.CrossEntropyLoss()}
            exec(original_ce,namespace);loss=namespace['loss_cls'];accuracy=loss.new_tensor(namespace['right_num'])/(lengths+1).sum()
        else:
            trainer=T2MGPTTrainer(bundle,pkeep=pkeep,if_maxtest=maximum)
            bundle.train();values=trainer.train_step(dict(tokens=target,token_lens=lengths,text=captions));loss,accuracy=values['loss'],values['t2mgpt_accuracy']
        loss.backward()
        output['training']['gpt'+str(index)]={'loss':loss.detach().cpu(),'accuracy':accuracy.detach().cpu(),'gradients':hashes(gpt,gradients=True),'rng':torch.cuda.get_rng_state().cpu()}
    torch.manual_seed(623)
    vq.train().zero_grad(set_to_none=True)
    motion=torch.sin(torch.arange(2*40*263,device='cuda').reshape(2,40,263)*.013)
    if args.worker=='upstream':
        prediction,commitment,perplexity=vq(motion)
        objective=ReConsLoss('l1_smooth',22)
        loss=objective(prediction,motion)+.02*commitment+.1*objective.forward_vel(prediction,motion)
    else:
        trainer=T2MGPTTrainer(bundle,stage='vqvae');bundle.train();loss=trainer.train_step(dict(motion=motion))['loss']
    loss.backward()
    quant=vq.vqvae.quantizer
    output['training']['vq']={'loss':loss.detach().cpu(),'gradients':hashes(vq,gradients=True),'state':hashes(vq),'ema':{k:tensors(v) for k,v in vars(quant).items() if isinstance(v,(torch.Tensor,bool))},'rng':torch.cuda.get_rng_state().cpu()}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--reuse-results',action='store_true')
    args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    from verify_prompthmr_video_parity import compare_full
    import torch
    report={'complete':False,'scope':'Actual released T2M-GPT neural weights, original eval-mode greedy/categorical decoding and VQ output, native public inference, original module-level corrupted-token CE with dropout and VQ reconstruction/commitment/EMA objectives, all gradient hashes and RNG. Original EOS-first evaluator fallback retained. Dataset and full optimizer protocol excluded.',
            'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),'tests':[]}
    path=args.output_dir/'source_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.reuse_results else ['upstream','native']):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--worker',kind],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},check=True)
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n');print('T2M-GPT independent original inference and training exact')

if __name__=='__main__':main()
