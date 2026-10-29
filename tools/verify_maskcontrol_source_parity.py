"""Independent released MaskControl control-optimization and training comparison."""
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
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    cfg=json.loads((args.artifact/'maskcontrol_config.json').read_text())['config']
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        import clip
        def local_clip(*a,**kw):
            return clip.model.build_model(dict(load_file(str(args.artifact/'clip.safetensors')))).float(),None
        clip.load=local_clip
        from models.mask_transformer.control_transformer import ControlTransformer
        from models.mask_transformer.transformer import ResidualTransformer
        from models.vq.model import RVQVAE
        from utils.motion_process import recover_from_ric
        c=cfg['vq'];vq=RVQVAE(SimpleNamespace(**c),263,c['nb_code'],c['code_dim'],c['output_emb_width'],c['down_t'],c['stride_t'],c['width'],c['depth'],c['dilation_growth_rate'],c['vq_act'],c['vq_norm'])
        vq.load_state_dict(load_file(str(args.artifact/'vq.safetensors')),strict=True)
        vq.to(args.device).eval().requires_grad_(False)
        state=load_file(str(args.artifact/'control.safetensors'))
        mean=state.pop('mean',torch.from_numpy(np.load(args.artifact/'Mean.npy'))).to(args.device)
        std=state.pop('std',torch.from_numpy(np.load(args.artifact/'Std.npy'))).to(args.device)
        mask_emb=state.pop('mask_emb_vq',None)
        common=dict(code_dim=cfg['vq']['code_dim'],cond_mode='text',clip_version='ViT-B/32',
            opt=SimpleNamespace(num_tokens=cfg['vq']['nb_code'],num_quantizers=cfg['vq']['num_quantizers'],joints_num=22,device=torch.device(args.device)))
        def config(c):
            return dict(latent_dim=c['latent_dim'],ff_size=c['ff_size'],num_layers=c['n_layers'],num_heads=c['n_heads'],dropout=c['dropout'],clip_dim=c['clip_dim'],cond_drop_prob=c['cond_drop_prob'])
        model=ControlTransformer(**common,**config(cfg['control']),vq_model=vq,mean=mean,std=std,control='random')
        if mask_emb is not None:model.mask_emb_vq=mask_emb.to(args.device)
        missing,unexpected=model.load_state_dict(state,strict=False)
        assert not unexpected and all(k.startswith(('clip_model.','vq_model.')) for k in missing),(missing,unexpected)
        residual=ResidualTransformer(**common,**config(cfg['residual']),shared_codebook=c['shared_codebook'],share_weight=cfg['residual']['share_weight'])
        missing,unexpected=residual.load_state_dict(load_file(str(args.artifact/'residual.safetensors')),strict=False)
        assert not unexpected and all(k.startswith('clip_model.') for k in missing),(missing,unexpected)
        model.to(args.device).eval().requires_grad_(False)
        residual.to(args.device).eval().requires_grad_(False)
    else:
        from motius.pipelines.maskcontrol import MaskControlPipeline
        from motius.pipelines.maskcontrol.sampling import MaskControlSampler
        from motius.motion.representation.humanml import recover_from_ric
        from motius.trainers.maskcontrol import MaskControlTrainer
        pipeline=MaskControlPipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':args.device})
        bundle=pipeline.bundle;model=bundle.control_model;vq=bundle.vq_model;residual=bundle.residual_model
    output={}
    for index,(batch,each,final,relative) in enumerate([(1,0,0,False),(2,2,3,False),(1,2,3,True)]):
        lengths=torch.tensor([40,32][:batch],device=args.device);text=['walk slowly','raise both arms'][:batch]
        targets=torch.zeros(batch,196,22,3,device=args.device);mask=torch.zeros(batch,196,22,dtype=torch.bool,device=args.device)
        targets[:,::8,20]=torch.tensor([.3,1.2,.2],device=args.device);mask[:,::8,20]=True
        torch.manual_seed(37+index)
        if args.worker=='upstream':
            joints,motion=model.generate_with_control(text,lengths,4,4.,1.,.9,False,vq,targets,mask,mean,std,5.,residual,
                control_opt=dict(each_iter=each,each_lr=.06,iter=final,lr=.06),is_relative=relative)
            physical=motion*std+mean
        else:
            motion,physical=pipeline.generate(text,lengths,targets,mask,time_steps=4,each_iterations=each,final_iterations=final,relative_control=relative)
            joints=recover_from_ric(physical.float(),22)
        output[f'batch{batch}_each{each}_final{final}_relative{relative}']={k:v.detach().cpu() for k,v in dict(normalized=motion,physical=physical,joints=joints).items()}
    for index,control in enumerate(['random','cross']):
        torch.manual_seed(81)
        pose=torch.randn(2,196,263,device=args.device)*.01;lengths=torch.tensor([196,160],device=args.device);text=['walk slowly','raise both arms']
        model.zero_grad(set_to_none=True)
        torch.manual_seed(83+index)
        if args.worker=='upstream':
            model.control=control;model.ctrl_train()
            _,ce,pred,accuracy,control_loss=model(text,lengths,pose)
            loss=ce+control_loss
        else:
            trainer=MaskControlTrainer(bundle,control=control)
            result=trainer.train_step(dict(text=text,motion=pose,motion_lens=lengths))
            ce,accuracy,control_loss,loss=(result[k] for k in ['maskcontrol_ce','maskcontrol_accuracy','maskcontrol_control','loss'])
        loss.backward()
        output[control+'_training']={'ce':ce.detach().cpu(),'control':control_loss.detach().cpu(),'accuracy':torch.as_tensor(accuracy).cpu(),
            'gradients':{k:p.grad.detach().cpu() for k,p in model.named_parameters() if p.grad is not None}}
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
    report=dict(complete=False,comparison='original MaskControl source and released weights; inference-time optimization and training gradients',
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
    report.update(complete=True,cudnn_deterministic=True,cudnn_benchmark=False,torch_version=torch.__version__,native_files={str(p.relative_to(root)):digest(p)
        for folder in ['motius/models/maskcontrol','motius/pipelines/maskcontrol','motius/trainers/maskcontrol'] for p in (root/folder).rglob('*.py')})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All MaskControl source comparisons exact')

if __name__=='__main__':main()
