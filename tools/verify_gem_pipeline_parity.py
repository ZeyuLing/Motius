"""Compare complete original GEM feature-to-body inference with native composition."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch
    if args.deterministic_geometry:
        from motius.utils.deterministic_geometry import DeterministicSparseMatmul
        runtime=DeterministicSparseMatmul();runtime.__enter__()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    variant=args.variant
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        if variant=='gem_x':sys.path.insert(0,str(args.upstream.parent/'soma-original-e0f'))
        from hydra import initialize_config_dir,compose
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=str(args.upstream/'configs'),version_base='1.3'):
            cfg=compose(config_name='demo' if variant=='gem_smpl' else 'demo_soma',overrides=['exp='+('gem_smpl' if variant=='gem_smpl' else 'gem_soma_regression'),'video_name=demo','ckpt_path=null'])
        model=instantiate(cfg.model,_recursive_=False)
        model.load_pretrained_model(str(args.artifact/'inputs/pretrained'/('gem_smpl.ckpt' if variant=='gem_smpl' else 'gem_soma.ckpt')))
        model.to(args.device).eval()
    else:
        from motius.models.gem_bundle import GEMModelBundle
        import importlib
        cls=type('NativeGemBundle',(GEMModelBundle,),dict(VARIANT=variant,CHECKPOINT='gem_smpl.ckpt' if variant=='gem_smpl' else 'gem_soma.ckpt'))
        bundle=cls(args.artifact,device=args.device,body_models_root=args.body_models_root)
        model=getattr(importlib.import_module('motius.pipelines.'+variant+'.composition'),'GemSmplComposition' if variant=='gem_smpl' else 'GemXComposition')(bundle)
    output={}
    for case,(frames,static_camera,postproc) in enumerate([(16,True,False),(21,False,True)]):
        torch.manual_seed(71+case)
        joints=17 if variant=='gem_smpl' else 77
        keypoints=torch.randn(frames,joints,3)*30+128;keypoints[...,2]=torch.rand(frames,joints)*.3+.7
        boxes=torch.tensor([128.,128.,220.]).repeat(frames,1)
        intrinsics=torch.tensor([[900.,0.,128.],[0.,900.,128.],[0.,0.,1.]]).repeat(frames,1,1)
        rotation=torch.eye(3).repeat(frames,1,1)
        angular=torch.tensor([1.,0.,0.,0.,1.,0.]).repeat(frames,1)
        if not static_camera:
            from scipy.spatial.transform import Rotation
            rotation=torch.from_numpy(Rotation.from_euler('y',torch.linspace(0,.01,frames).numpy()).as_matrix()).float()
            # Use original representation's angular convention for both workers.
            angular[1:,1]=.001
        data=dict(kp2d=keypoints,bbx_xys=boxes,K_fullimg=intrinsics,cam_angvel=angular,cam_tvel=torch.zeros(frames,3),R_w2c=rotation,
                  f_imgseq=torch.randn(frames,1024),has_text=torch.tensor([False]),length=torch.tensor(frames),meta=[{'mode':'default'}],
                  mask={k:torch.full((frames,),flag,dtype=torch.bool) for k,flag in dict(has_img_mask=True,has_2d_mask=True,has_cam_mask=not static_camera,has_audio_mask=False,has_music_mask=False).items()})
        torch.manual_seed(311+case)
        with torch.no_grad():prediction=model.predict(data,static_cam=static_camera,postproc=postproc)
        def cpu(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu()
            if isinstance(value,dict):return {k:cpu(v) for k,v in value.items() if isinstance(v,(torch.Tensor,dict,list,tuple))}
            if isinstance(value,(list,tuple)):return [cpu(x) for x in value]
            return value
        output[str(case)]=cpu(prediction)
    torch.save(output,args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','body-models-root','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--variant',choices=['gem_smpl','gem_x'],required=True)
    parser.add_argument('--deterministic-geometry',action='store_true')
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    for key in ['upstream','artifact','body_models_root','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'pipeline_parity.json'
    report=dict(complete=False,deterministic_geometry=args.deterministic_geometry,comparison='Complete unchanged original GEM model/pipeline from real released weights versus native feature-to-body composition',scope='Input features fixed; video frontend and source training objectives require separate certificates',soma_revision='e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5' if args.variant=='gem_x' else None,upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        environment={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':environment['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=environment,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GEM pipeline comparisons exact',args.variant)

if __name__=='__main__':main()
