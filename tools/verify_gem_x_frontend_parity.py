"""Compare GEM-X source video preprocessing against native neural/processing components."""
from __future__ import annotations
import argparse,ast,json,os,subprocess,sys,uuid
from pathlib import Path
from types import SimpleNamespace
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
    path=args.output_dir/args.worker;path.mkdir(parents=True,exist_ok=True)
    diagnostics={}
    def instrument(cls):
        original=cls.__init__
        def initialize(self,*values,**options):
            original(self,*values,**options)
            model=self.estimator.model
            diagnostics['dtypes']={key:str(value.dtype) for key,value in model.named_parameters()}
            def first(value):
                if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
                if isinstance(value,(list,tuple)):
                    for item in value:
                        result=first(item)
                        if result is not None:return result
                if isinstance(value,dict):
                    for item in value.values():
                        result=first(item)
                        if result is not None:return result
            for name in ['backbone','decoder','head_pose']:
                module=getattr(model,name)
                def hook(module,inputs,output,key=name):
                    diagnostics.setdefault(key,[]).append(dict(input=first(inputs),output=first(output)))
                module.register_forward_hook(hook)
        cls.__init__=initialize

    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        # Resolve the unchanged source torch.hub request from an isolated cache
        # containing the independently checked-out required DINOv3 revision.
        cache=args.output_dir/'torch_hub';cache.mkdir(exist_ok=True)
        alias=cache/'facebookresearch_dinov3_main'
        if not alias.exists():alias.symlink_to(args.dinov3,target_is_directory=True)
        (cache/'trusted_list').write_text('facebookresearch_dinov3\n')
        torch.hub.set_dir(str(cache))
        from gem.utils.sam3db_extractor import SAM3DBExtractor
        instrument(SAM3DBExtractor)
        from gem.utils.vitpose_extractor import VitPoseExtractor
        from gem.utils.video_io_utils import get_video_lwh,read_video_np
        from gem.utils.geo_transform import get_bbx_xys_from_xyxy,compute_cam_angvel,compute_cam_tvel,normalize_T_w2c
        from gem.utils.cam_utils import estimate_K,get_a_pred_cam
        from gem.utils.kp2d_utils import smooth_bbx_xyxy
        from gem.utils.pylogger import Log
        # Extract the complete original preprocessing functions without importing
        # unrelated optional rendering applications (Open3D/PyTorch3D).
        names={'_run_human_detection','run_preprocess','load_data_dict'}
        tree=ast.parse((args.upstream/'scripts/demo/demo_soma.py').read_text())
        scope=dict(torch=torch,Path=Path,Log=Log,SAM3DBExtractor=SAM3DBExtractor,VitPoseExtractor=VitPoseExtractor,
            get_video_lwh=get_video_lwh,read_video_np=read_video_np,get_bbx_xys_from_xyxy=get_bbx_xys_from_xyxy,
            compute_cam_angvel=compute_cam_angvel,compute_cam_tvel=compute_cam_tvel,normalize_T_w2c=normalize_T_w2c,
            estimate_K=estimate_K,get_a_pred_cam=get_a_pred_cam,smooth_bbx_xyxy=smooth_bbx_xyxy)
        exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[]),str(args.upstream/'scripts/demo/demo_soma.py'),'exec'),scope)
        from omegaconf import OmegaConf
        paths={key:str(path/name) for key,name in dict(bbx='bbx.pt',vitpose='vitpose.pt',slam='slam_results.pt',vit_features='vit_features.pt').items()}
        cfg=OmegaConf.create(dict(video_path=str(args.video),paths=paths,verbose=False,static_cam=False,render_mhr=False,
            sam3d_ckpt_path=str(args.artifact/'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt'),sam3d_mhr_path=str(args.artifact/'inputs/mhr_data/mhr_model.pt')))
        scope['run_preprocess'](cfg);data=scope['load_data_dict'](cfg)
    else:
        from motius.pipelines.gem_x.features import SAM3DBExtractor
        instrument(SAM3DBExtractor)
        from motius.pipelines.gem_x.frontend import prepare_video
        data=prepare_video(args.video,path,args.artifact,'cuda',False)
    data.pop('meta');torch.save(data,args.output_dir/(args.worker+'.pt'))
    torch.save(diagnostics,args.output_dir/(args.worker+'-sam-debug.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','dinov3','video','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--deterministic-geometry',action='store_true')
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','dinov3','video','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    run=args.output_dir/('run-'+uuid.uuid4().hex[:12]);run.mkdir()
    path=args.output_dir/'frontend_parity.json'
    report=dict(complete=False,deterministic_geometry=args.deterministic_geometry,comparison='Original unchanged GEM-X video detection/ByteTrack, DINOv3 SOMA-77 heatmaps, SAM3D primary body-token features and camera preparation versus native components',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),dinov3_commit=subprocess.check_output(['git','-C',str(args.dinov3),'rev-parse','HEAD'],text=True).strip(),video_sha256=digest(args.video),run_directory=str(run),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        env['TORCH_HOME']=str(args.upstream/'inputs/torch_home')
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*(['--deterministic-geometry'] if args.deterministic_geometry else []),'--worker',kind,'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--dinov3',str(args.dinov3),'--video',str(args.video),'--output-dir',str(run)],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(run/'upstream.pt',weights_only=True),torch.load(run/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GEM-X frontend comparisons exact')

if __name__=='__main__':main()
