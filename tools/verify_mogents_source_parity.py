"""Compare independent released MoGenTS source graphs with native composition."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cfg=json.loads((args.artifact/'mogents_config.json').read_text())['config'];c=cfg['vq']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        import clip
        def local_clip(*a,**kw):
            return clip.model.build_model(dict(load_file(str(args.artifact/'clip.safetensors')))).float(),None
        clip.load=local_clip
        from models.transformer.transformer_aux import MaskTransformer,ResidualTransformer
        from models.transformer.transformer_ts import MaskTransformer2D,ResidualTransformer2D
        from models.vq.model import RVQVAE,LengthEstimator
        opt=argparse.Namespace(num_tokens1d=c['nb_code1d'],num_tokens2d=c['nb_code2d'],num_quantizers=c['num_quantizers'],
            device=torch.device(args.device),attnj=cfg['mask']['attnj'],attnt=cfg['mask']['attnt'])
        def options(config,spatial=False,residual=False):
            value=dict(code_dim=c['code_dim2d' if spatial else 'code_dim1d'],cond_mode='text',clip_version='ViT-B/32',opt=opt,
                latent_dim=config['latent_dim'],ff_size=config['ff_size'],num_layers=config['n_layers'],num_heads=config['n_heads'],
                dropout=config['dropout'],clip_dim=config['clip_dim'],cond_drop_prob=config['cond_drop_prob'])
            if residual:value.update(shared_codebook=c['shared_codebook'],share_weight=config['share_weight'])
            return value
        models=dict(aux_base=MaskTransformer(**options(cfg['mask'])),ts_base=MaskTransformer2D(**options(cfg['mask'],True)),
            aux_residual=ResidualTransformer(**options(cfg['res'],False,True)),ts_residual=ResidualTransformer2D(**options(cfg['res'],True,True)))
        vq=RVQVAE(SimpleNamespace(**c),263,c['down_t'],c['stride_t'],c['width'],c['depth'],c['dilation_growth_rate'],c['vq_act'],c['vq_norm'])
        length=LengthEstimator(**cfg['length'])
        for name,model in list(models.items())+[('vq',vq),('length_est',length)]:
            filename=dict(aux_base='mask_aux',ts_base='mask_ts',aux_residual='res_aux',ts_residual='res_ts').get(name,name)
            missing,unexpected=model.load_state_dict(load_file(str(args.artifact/f'{filename}.safetensors')),strict=False)
            assert not unexpected and all(k.startswith('clip_model.') for k in missing),(name,missing,unexpected)
            model.to(args.device).eval()
    else:
        from motius.models.mogents import MoGenTSBundle
        from motius.pipelines.mogents import decoding
        from motius.trainers.mogents import objective
        from motius.trainers.mogents.trainer import MoGenTSTrainer
        from motius.motion.representation.mogents import pack_motion,unpack_motion
        bundle=MoGenTSBundle.from_pretrained(str(args.artifact),device=args.device)
        models={stage:getattr(bundle,name) for stage,name in MoGenTSTrainer.MODULES.items()}
        vq,length=bundle.vq_model,bundle.length_estimator
    from motius.motion.conditioning.clip_text import encode_clip_text
    output={}
    stats=[torch.from_numpy(np.load(args.artifact/f'{n}.npy')).to(args.device) for n in ['Mean','Std']]
    for index,(lengths,gumbel) in enumerate([([10],False),([15,10],True)]):
        text=['A person walks forward.','A person raises both arms.'][:len(lengths)]
        lens=torch.tensor(lengths,device=args.device)
        torch.manual_seed(31+index)
        with torch.no_grad():
            generate=lambda stage,*a,**kw: (models[stage].generate(*a,**kw) if args.worker=='upstream' else getattr(decoding,stage+'_generate')(models[stage],*a,**kw))
            base=generate('aux_base',text,lens,10,4,gsample=gumbel)
            base_ts=generate('ts_base',text,lens,10,4,gsample=gumbel,n_j=6)
            residual=generate('aux_residual',base,text,lens,cond_scale=5.)
            residual_ts=generate('ts_residual',base_ts,text,lens,cond_scale=5.)
            aux,spatial=vq.forward_decoder(residual,residual_ts)
            motion=spatial if args.worker=='upstream' else bundle.fuse_motion(aux,unpack_motion(spatial,vq.J0))
            length_logits=length(models['aux_base'].encode_text(text) if args.worker=='upstream' else encode_clip_text(models['aux_base'], text))
        output[f'batch{len(lengths)}_gumbel{gumbel}']={k:v.cpu() for k,v in dict(base=base,base_ts=base_ts,residual=residual,
            residual_ts=residual_ts,aux=aux,motion=motion,physical=motion*stats[1]+stats[0],length_logits=length_logits).items()}
    # Independently reproduce the original dataset's joint-field layout, then
    # compare encoding and both decoder branches with real released codebooks.
    torch.manual_seed(113);motion=torch.randn(2,40,263,device=args.device)
    if args.worker=='upstream':
        fields=torch.zeros(2,40,22,12,device=args.device)
        fields[:,:,1:,:3]=motion[:,:,4:67].reshape(2,40,21,3)
        fields[:,:,1:,3:9]=motion[:,:,67:193].reshape(2,40,21,6)
        fields[:,:,:,9:12]=motion[:,:,193:259].reshape(2,40,22,3)
    else:fields=pack_motion(motion)
    with torch.no_grad():
        encoded=vq.encode(motion,fields)
        reconstruction=vq(motion,fields)
        aux,commit_a,perplex_a,spatial,commit_s,perplex_s=reconstruction[:6]
        if args.worker=='upstream':fused=reconstruction[6]
        else:
            spatial=unpack_motion(spatial,vq.J0);fused=bundle.fuse_motion(aux,spatial)
    output['autoencoder']={k:v.cpu() for k,v in dict(codes_aux=encoded[0],codes_ts=encoded[2],aux=aux,spatial=spatial,fused=fused,
        commitment_aux=commit_a,commitment_ts=commit_s,perplexity_aux=perplex_a,perplexity_ts=perplex_s).items()}
    for index,(stage,model) in enumerate(models.items()):
        spatial=stage.startswith('ts_');residual=stage.endswith('residual')
        shape=(2,8,6) if spatial else (2,8)
        if residual:shape+= (c['num_quantizers'],)
        torch.manual_seed(93)
        tokens=torch.randint(0,c['nb_code2d' if spatial else 'nb_code1d'],shape,device=args.device)
        lens=torch.tensor([8,6],device=args.device);text=['walk forward','jump once']
        model.train();model.clip_model.eval();model.zero_grad(set_to_none=True)
        torch.manual_seed(79+index)
        if args.worker=='upstream':loss,pred,accuracy=model(tokens,text,lens)
        else:loss,pred,accuracy=getattr(objective,'compute_'+stage+'_objective')(model,model,tokens,text,lens)
        loss.backward()
        output[stage+'_training']={'loss':loss.detach().cpu(),'prediction':pred.cpu(),'accuracy':torch.as_tensor(accuracy),
            'gradients':{k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None}}
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda')
    args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'source_parity.json'
    report=dict(complete=False,comparison='Original MoGenTS source and released HumanML3D weights: dual-stream generation, VQ encoding/decoding, four token-stage objectives and gradients',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        source_files={str(p.relative_to(args.upstream)):digest(p) for p in (args.upstream/'models').rglob('*.py')},
        artifact_files={str(p.relative_to(args.artifact)):digest(p) for p in args.artifact.iterdir() if p.is_file() and p.suffix in ['.safetensors','.json','.npy']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind,'--device',args.device]
        for key in ['upstream','artifact','output_dir']:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    folders=['motius/models/mogents','motius/models/momask','motius/pipelines/mogents','motius/trainers/mogents','motius/motion/representation/mogents.py']
    report.update(complete=True,torch_version=torch.__version__,deterministic_cudnn=True,native_files={str(p.relative_to(root)):digest(p)
        for folder in folders for p in (list((root/folder).rglob('*.py')) if (root/folder).is_dir() else [root/folder])})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All MoGenTS source comparisons exact')

if __name__=='__main__':main()
