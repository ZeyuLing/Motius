"""Compare all three original MotionBricks training stages with native Trainers."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
from verify_bailando_source_parity import compare
from verify_motionbricks_rollout_parity import reference


def worker(args):
    import torch,numpy as np,random
    torch.set_num_threads(4);torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if args.worker=='upstream':
        models,_,_,_=reference(args)
        from motionbricks.vqvae.models.motion_vqvae import MotionVQVAEModel
        from omegaconf import OmegaConf
        cfg=OmegaConf.load(args.artifact/'motionbricks_checkpoint/motionbricks_vqvae/version_1/hparams.yaml')
        rep=models['pose'].motion_rep
        models['vqvae']=MotionVQVAEModel(models['pose'].supporting_nets['pose_net'],None,rep,optimizer=None,scheduler=None,args=cfg.model.args)
    else:
        from motius.models.motionbricks import MotionBricksBundle
        from motius.trainers.motionbricks import MotionBricksPoseTrainer,MotionBricksRootTrainer,MotionBricksVQTrainer
        from motius.configuration.motionbricks import build_representation
        bundle=MotionBricksBundle(args.artifact,device='cuda')
        rep=build_representation(args.artifact)
        trainers=dict(pose=MotionBricksPoseTrainer,root=MotionBricksRootTrainer,vqvae=MotionBricksVQTrainer)
    torch.set_float32_matmul_precision('high')
    output={}
    for case,stage in enumerate(['pose','root','vqvae']):
        torch.manual_seed(21+case);np.random.seed(21+case);random.seed(21+case)
        batch_size,frames=2,512
        motion=torch.randn(batch_size,frames,len(rep.indices['all']),device='cuda')*.05
        batch=dict(motion=motion,motion_len=torch.tensor([frames,frames-4],device='cuda'),motion_pad_mask=torch.zeros(batch_size,frames,device='cuda',dtype=torch.bool),batch_size=batch_size,
            text=['',''],text_len=torch.zeros(batch_size,device='cuda',dtype=torch.long),text_pad_mask=torch.zeros(batch_size,1,device='cuda',dtype=torch.bool),text_embeddings=torch.randn(batch_size,4096,device='cuda'))
        if args.worker=='upstream':
            model=models[stage].cuda().train();model.zero_grad(set_to_none=True)
            model._trainer=SimpleNamespace(global_step=0)
            if stage=='pose':model.supporting_nets['pose_net'].eval().cuda()
            metrics={}
            def log(key,value,**kwargs):metrics[key.removeprefix('loss/train_')]=value
            model.log=log
            active=model.backbone_net if stage!='vqvae' else model.pose_net
        else:
            trainer=trainers[stage](bundle)
            bundle.zero_grad(set_to_none=True)
            active=getattr(bundle,stage) if stage!='vqvae' else torch.nn.ModuleDict({name:getattr(bundle,name) for name in ['encoder','decoder','quantizer']})
        # Initializers may draw random values; stochastic training starts here in both workers.
        torch.manual_seed(51+case);np.random.seed(51+case);random.seed(51+case)
        if args.worker=='upstream':loss=model.training_step(batch,0)
        else:metrics=trainer.train_step(batch);loss=metrics['loss']
        assert loss is not None and torch.isfinite(loss),stage
        loss.backward()
        output[stage]={key:value.detach().cpu() for key,value in metrics.items() if isinstance(value,torch.Tensor)}
        output[stage]['gradients']={key:parameter.grad.detach().cpu() for key,parameter in active.named_parameters() if parameter.grad is not None}
        if stage=='vqvae':output[stage]['ema']={key:value.detach().cpu() for key,value in active.state_dict().items() if key.startswith('quantizer.')}
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'training_parity.json';report=dict(complete=False,comparison='Actual original MotionBricks pose/root/VQ data augmentation, full original weighted objectives, active neural gradients and VQ EMA updates versus native Trainers',scope='Synthetic normalized raw G1 supervision and released weights; full datasets/optimizer protocols require separate validation',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream/'motionbricks' if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All MotionBricks training comparisons exact')

if __name__=='__main__':main()
