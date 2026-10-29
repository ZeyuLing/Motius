"""Compare native TMR to independent original TMR using the released Motius weights."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors,compare_full


def worker(args):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cfg=json.loads((args.artifact/'config.json').read_text());architecture=cfg['arch']
    state=load_file(str(args.artifact/cfg['weights_file']))
    if all(key.startswith('tmr.') for key in state):state={key[4:]:value for key,value in state.items()}
    output={}
    for case,(train,sample_mean) in enumerate([(False,True),(False,False),(True,True),(True,False)]):
        if args.worker=='upstream':
            sys.path.insert(0,str(args.upstream))
            from src.model.actor import ACTORStyleEncoder,ACTORStyleDecoder
            from src.model.tmr import TMR
            model=TMR(ACTORStyleEncoder(cfg['motion_nfeats'],cfg['vae'],**architecture),
                ACTORStyleEncoder(cfg['text_nfeats'],cfg['vae'],**architecture),
                ACTORStyleDecoder(cfg['motion_nfeats'],**architecture),vae=cfg['vae'],sample_mean=sample_mean,temperature=.1)
            model.load_state_dict(state,strict=True);model.cuda().train(train);network=model
        else:
            from motius.models.tmr import TMRBundle
            from motius.trainers.tmr import TMRTrainer
            bundle=TMRBundle(motion_nfeats=cfg['motion_nfeats'],text_nfeats=cfg['text_nfeats'],vae=cfg['vae'],arch=architecture,sample_mean=sample_mean)
            bundle.tmr.load_state_dict(state,strict=True);bundle.cuda().train(train);network=bundle.tmr;model=TMRTrainer(bundle)
        torch.manual_seed(221+case);batch=3;text_length=7;frames=24
        text=dict(x=torch.randn(batch,text_length,cfg['text_nfeats'],device='cuda'),mask=torch.arange(text_length,device='cuda')[None]<torch.tensor([7,4,6],device='cuda')[:,None])
        motion=dict(x=torch.randn(batch,frames,cfg['motion_nfeats'],device='cuda'),mask=torch.arange(frames,device='cuda')[None]<torch.tensor([24,18,20],device='cuda')[:,None])
        sentence=torch.nn.functional.normalize(torch.randn(batch,768,device='cuda'),dim=-1);sentence[1]=sentence[0]
        inputs=dict(text_x_dict=text,motion_x_dict=motion,sent_emb=sentence)
        torch.manual_seed(329+case)
        losses=model.compute_loss(inputs)
        diagnostic={}
        if args.diagnostic_losses:
            first=next(network.parameters())
            diagnostic={key:torch.autograd.grad(losses[key],first,retain_graph=True)[0] for key in ['recons','kl','latent','contrastive','loss']}
        losses['loss'].backward()
        wanted=['recons','kl','latent','contrastive','loss']
        result=dict(losses={name:losses[name] for name in wanted},gradients={name:None if param.grad is None else param.grad for name,param in network.named_parameters()})
        if args.diagnostic_losses:result['per_loss_token_gradients']=diagnostic
        network.eval()
        with torch.no_grad():
            if args.worker=='upstream':
                result['text_embedding']=model.encode(text,modality='text',sample_mean=True)
                result['motion_embedding']=model.encode(motion,modality='motion',sample_mean=True)
                result['reconstruction']=model.decode(result['text_embedding'],mask=motion['mask'])
            else:
                result['text_embedding']=network.encode(text,modality='text',sample_mean=True)
                result['motion_embedding']=network.encode(motion,modality='motion',sample_mean=True)
                result['reconstruction']=network.decode(result['text_embedding'],motion['mask'])
        output[str(case)]=tensors(result)
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--diagnostic-losses',action='store_true');args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'source_parity.json';report=dict(complete=False,comparison='Independent Mathux/TMR original networks and objective versus native Bundle/Trainer, with identical released Motius evaluator weights',scope='Prepared text/motion inputs, eval/train dropout, mean/stochastic VAE sampling, reconstruction/contrastive/KL objectives and all gradients. This is not an original TMR pretrained checkpoint or full raw-text/dataset protocol.',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All independent original TMR comparisons exact')

if __name__=='__main__':main()
