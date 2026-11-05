"""Compare the published PromptHMR temporal objective and all neural gradients."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors,compare_full


def worker(args):
    import torch,numpy as np
    from omegaconf import OmegaConf
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1234);np.random.seed(1234)
    body=args.body_models_root/'smplx/SMPLX_NEUTRAL.npz'
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream));sys.path.insert(0,str(args.upstream/'pipeline/gvhmr'))
        from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import Pipeline
        config=OmegaConf.load(args.artifact/'video/prhmr_release_002.yaml')
        trainer=Pipeline(config.pipeline.args,config.network,str(body)).cuda().train()
        state=torch.load(args.artifact/'video/phmr_b1b2.ckpt',map_location='cpu',weights_only=False)['state_dict']
        prefix='pipeline.denoiser3d.'
        network=trainer.denoiser3d
        network.load_state_dict({key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)},strict=True)
    else:
        from motius.models.prompthmr import PromptHMRBundle
        from motius.trainers.prompthmr import PromptHMRVideoTrainer
        bundle=PromptHMRBundle.from_pretrained(args.artifact)
        trainer=PromptHMRVideoTrainer(bundle,body_model_path=body).train();network=bundle.video
    output={}
    for case in range(2):
        torch.manual_seed(291+case);batch,length=2,12
        def rand(*shape):return torch.randn(*shape,device='cuda')
        params=dict(body_pose=rand(batch,length,63)*.03,betas=rand(batch,length,10)*.2,
            global_orient=rand(batch,length,3)*.1,transl=rand(batch,length,3)*.01)
        params['transl'][...,2]+=3
        world={name:value.clone() for name,value in params.items()}
        world['transl'][...,0]+=torch.arange(length,device='cuda')*.02
        intrinsic=torch.eye(3,device='cuda').expand(batch,length,3,3).clone()
        intrinsic[...,0,0]=intrinsic[...,1,1]=640;intrinsic[...,0,2]=180;intrinsic[...,1,2]=320
        bbox=torch.tensor([180.,320.,400.],device='cuda').expand(batch,length,3).clone()
        valid=torch.ones(batch,length,device='cuda',dtype=torch.bool);valid[1,-2:]=False
        data=dict(length=torch.tensor([length,length-2],device='cuda'),obs=rand(batch,length,17,3),bbx_xys=bbox,K_fullimg=intrinsic,
            cam_angvel=rand(batch,length,6)*.01,f_imgseq=rand(batch,length,2,1024),smpl_params_c=params,smpl_params_w=world,
            R_c2gv=torch.eye(3,device='cuda').expand(batch,length,3,3).clone(),gt_j3d=rand(batch,length,22,3),
            gt_c_verts437=rand(batch,length,437,3)*.1,gt_cr_verts437=rand(batch,length,437,3)*.1,
            mask=dict(valid=valid,spv_incam_only=torch.tensor([False,bool(case)],device='cuda')),nb_epoch=6)
        data['gt_j3d'][...,2]=3;data['gt_c_verts437'][...,2]+=3
        network.zero_grad(set_to_none=True);torch.manual_seed(571+case)
        capture={}
        handle=network.register_forward_hook(lambda module,inputs,result:capture.update(model_output=result))
        result=trainer(data,train=True,nb_epoch=6) if args.worker=='upstream' else trainer.train_step(data)
        handle.remove();result['loss'].backward()
        names=['loss','simple_loss','cr_j3d_loss','transl_c_loss','j2d_loss','cr_vert_loss','verts2d_loss','transl_w_loss','static_conf_loss']
        output[str(case)]=tensors(dict(losses={name:result[name] for name in names},**capture,
            gradients={name:None if param.grad is None else param.grad for name,param in network.named_parameters()}))
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','body-models-root','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for name in ['upstream','artifact','body_models_root','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'training_parity.json';report=dict(complete=False,comparison='Original published temporal Pipeline(train=True) weighted camera/world/vertex/contact objectives, condition dropout and all temporal neural gradients versus native BaseTrainer',scope='Two synthetic supervised batches, released BEDLAM1+2 temporal weights, full original objective and gradients. Image training, dataset and full optimizer/schedule protocol are not certified.',tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All original temporal training comparisons exact')

if __name__=='__main__':main()
