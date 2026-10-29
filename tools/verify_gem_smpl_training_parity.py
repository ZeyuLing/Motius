"""Compare original GEM-SMPL augmentation, regression/diffusion losses and gradients."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_bailando_source_parity import compare


def worker(args):
    import torch,numpy as np
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    variant=args.variant
    if variant=='gem_x':
        from motius.utils.deterministic_geometry import DeterministicSparseMatmul
        runtime=DeterministicSparseMatmul();runtime.__enter__()
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        if variant=='gem_x':sys.path.insert(0,str(args.upstream.parent/'soma-original-e0f'))
        from hydra import initialize_config_dir,compose
        from hydra.utils import instantiate
        from types import SimpleNamespace
        with initialize_config_dir(config_dir=str(args.upstream/'configs'),version_base='1.3'):
            cfg=compose(config_name='demo' if variant=='gem_smpl' else 'demo_soma',overrides=['exp='+('gem_smpl' if variant=='gem_smpl' else 'gem_soma_regression'),'video_name=demo','ckpt_path=null'])
        model=instantiate(cfg.model,_recursive_=False)
        model.load_pretrained_model(str(args.artifact/'inputs/pretrained'/('gem_smpl.ckpt' if variant=='gem_smpl' else 'gem_soma.ckpt')))
        model.to('cuda').train();model._trainer=SimpleNamespace(global_step=0)
    else:
        import importlib
        prefix='GemSmpl' if variant=='gem_smpl' else 'GemX'
        Bundle=getattr(importlib.import_module('motius.models.'+variant),prefix+'Bundle')
        Processor=getattr(importlib.import_module('motius.motion.conditioning.'+variant+'_processor'),prefix+'Processor')
        Trainer=getattr(importlib.import_module('motius.trainers.'+variant),prefix+'Trainer')
        bundle=Bundle.from_pretrained(args.artifact,body_models_root=args.body_models_root,device='cuda')
        processor=Processor(bundle);trainer=Trainer(bundle,processor)
    results={}
    for case,stage in enumerate(['regression','diffusion','joint'] if variant=='gem_smpl' else ['regression','joint']):
        torch.manual_seed(151+case);np.random.seed(151+case)
        batch_size,frames=2,120
        device='cuda';lengths=torch.tensor([frames,frames-3],device=device)
        valid=torch.arange(frames,device=device)[None]<lengths[:,None]
        def random(*shape):return torch.randn(*shape,device=device)
        pose=dict(body_pose=random(batch_size,frames,63)*.03,global_orient=random(batch_size,frames,3)*.03,
                  betas=random(batch_size,frames,10)*.1,transl=random(batch_size,frames,3)*.03+torch.tensor([0.,0.,3.],device=device))
        if variant=='gem_x':
            pose=dict(body_pose=random(batch_size,frames,228)*.03,global_orient=pose['global_orient'],
                identity_coeffs=random(batch_size,frames,45)*.03,scale_params=random(batch_size,frames,69)*.03,transl=pose['transl'])
        parameter_key='smpl_params' if variant=='gem_smpl' else 'soma_params'
        batch=dict(length=lengths,**{parameter_key+'_c':pose,parameter_key+'_w':{k:v.clone() for k,v in pose.items()}},
            R_c2gv=torch.eye(3,device=device).repeat(batch_size,frames,1,1),
            kp2d=random(batch_size,frames,17 if variant=='gem_smpl' else 77,3)*25+128,bbx_xys=torch.tensor([128.,128.,220.],device=device).repeat(batch_size,frames,1),
            K_fullimg=torch.tensor([[900.,0.,128.],[0.,900.,128.],[0.,0.,1.]],device=device).repeat(batch_size,frames,1,1),
            cam_angvel=torch.tensor([1.,0.,0.,0.,1.,0.],device=device).repeat(batch_size,frames,1),
            f_imgseq=random(batch_size,frames,1024),caption=['',''],has_text=torch.zeros(batch_size,dtype=torch.bool,device=device))
        batch['kp2d'][...,2]=.9
        batch['mask']={key:valid.clone() for key in ['valid','bbx_xys','vitpose','has_img_mask','has_2d_mask','has_cam_mask']}
        batch['mask']['vitpose']=torch.ones(batch_size,dtype=torch.bool,device=device)
        batch['mask'].update({key:torch.zeros_like(valid) for key in ['has_audio_mask','has_music_mask']})
        batch['mask'].update({key:torch.zeros(batch_size,dtype=torch.bool,device=device) for key in ['2d_only','spv_incam_only','invalid_contact']})
        if variant=='gem_x':
            # The published Metrosim dataset supplies static_gt directly. Build
            # equivalent labels from this synthetic world's actual body motion.
            if args.worker=='upstream':
                from gem.utils.motion_utils import get_static_joint_mask
                body_model=model.body_model
            else:
                from motius.motion.representation.gem_x.motion import get_static_joint_mask
                body_model=processor.body_model
            with torch.no_grad():
                joints=body_model(**batch['soma_params_w'])['joints']
                batch['static_gt']=get_static_joint_mask(joints,vel_thr=.15,repeat_last=True)[:,:,[69,70,74,75,14,42]].float()
        if stage=='diffusion':batch['regression_outputs']={'model_output':{'pred_x_start':random(batch_size,frames,151)}}
        torch.manual_seed(131+case);np.random.seed(131+case)
        if args.worker=='upstream':
            model.zero_grad(set_to_none=True)
            if stage=='joint':
                # Logging is the only omitted side effect; all training math is the original method.
                model.log=lambda *args,**kwargs:None
                outputs=model.training_step(batch,0)
            else:
                model.prepare_batch(batch,stage)
                outputs=model.train_step(batch,0,stage)
            parameters={('denoiser.'+k.removeprefix('pipeline.denoiser3d.denoiser.') if k.startswith('pipeline.denoiser3d.denoiser.') else 'conditioner.'+k):v for k,v in model.named_parameters()}
        else:
            bundle.zero_grad(set_to_none=True);trainer.stage=stage
            outputs=trainer.train_step(batch)
            parameters=dict(bundle.named_parameters())
        outputs['loss'].backward()
        results[stage]={key:value.detach().cpu() for key,value in outputs.items() if isinstance(value,torch.Tensor)}
        results[stage]['gradients']={key:value.grad.detach().cpu() for key,value in parameters.items() if value.grad is not None}
    torch.save(results,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','body-models-root','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--variant',choices=['gem_smpl','gem_x'],default='gem_smpl')
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','body_models_root','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'training_parity.json'
    report=dict(complete=False,comparison='Original '+args.variant+' raw body supervision augmentation, default weighted stage objectives and all active network gradients versus native Trainer',runtime_policy='PyTorch deterministic algorithms; GEM-X additionally uses the same constant sparse-map dispatch in both workers',scope='Synthetic raw SMPL supervision with actual licensed geometry and released model; dataset loading and optimizer protocol require separate validation',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All '+args.variant+' training comparisons exact')

if __name__=='__main__':main()
