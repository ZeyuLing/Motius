"""TM2T released neural weights, original beam decoder and caption objective."""
from pathlib import Path
import argparse,ast,json,os,subprocess,sys,types
from types import SimpleNamespace as NS
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def worker(args):
    import torch,numpy as np
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
    from motius.motion.conditioning.tm2t import TM2TVocabulary
    from motius.configuration.tm2t import _first_existing
    vocabulary=TM2TVocabulary(args.artifact);pad=len(vocabulary)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from networks.modules import VQEncoderV3
        from networks.quantizer import Quantizer
        from networks.transformer import TransformerV2,cal_performance
        from networks.translator import Translator
        vq=VQEncoderV3(259,[1024,1024],2).cuda().eval()
        quant=Quantizer(1024,1024,1.).cuda().eval()
        text=TransformerV2(1027,1026,len(vocabulary)+1,pad,d_src_word_vec=512,d_trg_word_vec=512,d_model=512,d_inner=2048,n_enc_layers=4,n_dec_layers=4,n_head=8,d_k=64,d_v=64,dropout=.1,n_src_position=100,n_trg_position=50,trg_emb_prj_weight_sharing=True).cuda().eval()
        raw=torch.load(args.artifact/'t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar',map_location='cpu',weights_only=False)
        vq.load_state_dict(raw['vq_encoder'],strict=True);quant.load_state_dict(raw['quantizer'],strict=True)
        raw=torch.load(args.artifact/'t2m/M2T_EL4_DL4_NH8_PS/model/finest.tar',map_location='cpu',weights_only=False);text.load_state_dict(raw['m2t_transformer'],strict=True)
        beam=Translator(text,2,30,1026,pad,vocabulary.index('sos'),vocabulary.index('eos')).cuda()
        stats=[torch.from_numpy(np.load(args.artifact/f't2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/{name}.npy')).cuda() for name in ['mean','std']]
    else:
        from motius.pipelines.tm2t import TM2TPipeline
        from motius.trainers.tm2t import TM2TTrainer
        pipe=TM2TPipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':'cuda'})
        vq,quant,text=pipe.bundle.vq_encoder,pipe.bundle.quantizer,pipe.bundle.transformer
        beam=pipe.translator;stats=[pipe.bundle.mean,pipe.bundle.std]
    output={'weights':{'vq':hashes(vq),'quant':hashes(quant),'text':hashes(text)},'inference':{},'training':{}}
    for index,length in enumerate([40,76,196]):
        motion=torch.sin(torch.arange(length*263,device='cuda').reshape(length,263)*.017)*.03
        motion[...,3]=.9
        with torch.no_grad():
            normalized=torch.from_numpy((motion.cpu().numpy()-stats[0].cpu().numpy())/stats[1].cpu().numpy()).float().cuda()
            latent=vq(normalized[None,...,:-4]);codes=quant.map2index(latent).flatten().tolist()[:53]
            tokens=[1024]+codes+[1025];tokens += [1026]*(55-len(tokens))
            tokens=torch.tensor(tokens,device='cuda').unsqueeze(0)
            predicted=beam.translate_sentence(tokens)
            output['inference'][str(index)]=tensors({'latent':latent,'tokens':tokens,'predicted':predicted})
            if args.worker=='native':assert pipe.infer_m2t([motion],lengths=[length])[0]==' '.join(vocabulary.token(word) for word in predicted[1:-1])
    if args.worker=='upstream':
        source=args.upstream/'networks/trainers.py';tree=ast.parse(source.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TransformerM2TTrainer')
        namespace={'torch':torch,'cal_performance':cal_performance,'OrderedDict':__import__('collections').OrderedDict}
        for n in cls.body:
            if isinstance(n,ast.FunctionDef) and n.name in {'forward','backward'}:exec(compile(ast.Module(body=[n],type_ignores=[]),str(source),'exec'),namespace)
    for index,smoothing in enumerate([False,True]):
        torch.manual_seed(327+index)
        motion_tokens=torch.randint(0,1024,(2,9),device='cuda');motion_tokens[:,0]=1024;motion_tokens[:,-1]=1025
        words=torch.randint(0,pad,(2,8),device='cuda');words[:,0]=vocabulary.index('sos');words[:,-1]=vocabulary.index('eos');words[1,-2:]=pad
        text.train().zero_grad(set_to_none=True)
        if args.worker=='upstream':
            original=NS(m2t_transformer=text,device=torch.device('cuda'),opt=NS(m2t_v3=False,txt_pad_idx=pad,label_smoothing=smoothing))
            namespace['forward'](original,(torch.zeros(2,8,300,device='cuda'),words,['a','b'],torch.tensor([8,6]),motion_tokens,None))
            logs=namespace['backward'](original);loss=original.loss
            accuracy=loss.new_tensor(logs['accuracy']);mean=loss.detach()/original.n_word
        else:
            trainer=TM2TTrainer(pipe.bundle,label_smoothing=smoothing);pipe.bundle.train()
            values=trainer.train_step(dict(motion_tokens=motion_tokens,text_tokens=words));loss,accuracy,mean=values['loss'],values['tm2t_accuracy'],values['tm2t_mean_ce']
        loss.backward()
        output['training'][str(index)]={'loss':loss.detach().cpu(),'mean':mean.cpu(),'accuracy':accuracy.cpu(),'gradients':hashes(text,gradients=True),'rng':torch.cuda.get_rng_state().cpu()}
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
    report={'complete':False,'scope':'Original TM2T released VQ encoder/codebook/caption Transformer, three motion-to-token/beam cases and original TransformerM2TTrainer forward/backward with summed CE and label smoothing, all neural/gradient hashes and RNG. Native public caption composition checked against the same original neural/beam composition. Raw-dataset sampling, tokenizer/GAN training and complete optimizer protocol excluded.',
            'upstream_commit':subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),'tests':[]}
    path=args.output_dir/'source_parity.json';path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.reuse_results else ['upstream','native']):
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--output-dir',str(args.output_dir),'--worker',kind],cwd=ROOT,env={**os.environ,'PYTHONPATH':str(ROOT)},check=True)
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;path.write_text(json.dumps(report,indent=2)+'\n');print('TM2T released networks, beam decoder and caption training exact')

if __name__=='__main__':main()
