"""Compare the complete native GEM-X public video API with independent original source."""
from __future__ import annotations
import argparse,ast,json,os,subprocess,sys,uuid
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch,numpy as np
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    from motius.utils.deterministic_geometry import DeterministicSparseMatmul
    policy=DeterministicSparseMatmul();policy.__enter__()
    destination=args.output_dir/args.worker;destination.mkdir(parents=True,exist_ok=True)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream));sys.path.insert(0,str(args.upstream.parent/'soma-original-e0f'))
        cache=args.output_dir/'torch_hub';cache.mkdir(exist_ok=True)
        alias=cache/'facebookresearch_dinov3_main'
        if not alias.exists():alias.symlink_to(args.dinov3,target_is_directory=True)
        (cache/'trusted_list').write_text('facebookresearch_dinov3\n');torch.hub.set_dir(str(cache))
        from gem.utils.sam3db_extractor import SAM3DBExtractor
        from gem.utils.vitpose_extractor import VitPoseExtractor
        from gem.utils.video_io_utils import get_video_lwh,read_video_np
        from gem.utils.geo_transform import get_bbx_xys_from_xyxy,compute_cam_angvel,compute_cam_tvel,normalize_T_w2c
        from gem.utils.cam_utils import estimate_K,get_a_pred_cam
        from gem.utils.kp2d_utils import smooth_bbx_xyxy
        from gem.utils.pylogger import Log
        names={'_run_human_detection','run_preprocess','load_data_dict'}
        tree=ast.parse((args.upstream/'scripts/demo/demo_soma.py').read_text())
        scope=dict(torch=torch,Path=Path,Log=Log,SAM3DBExtractor=SAM3DBExtractor,VitPoseExtractor=VitPoseExtractor,
            get_video_lwh=get_video_lwh,read_video_np=read_video_np,get_bbx_xys_from_xyxy=get_bbx_xys_from_xyxy,
            compute_cam_angvel=compute_cam_angvel,compute_cam_tvel=compute_cam_tvel,normalize_T_w2c=normalize_T_w2c,
            estimate_K=estimate_K,get_a_pred_cam=get_a_pred_cam,smooth_bbx_xyxy=smooth_bbx_xyxy)
        exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names],type_ignores=[]),str(args.upstream/'scripts/demo/demo_soma.py'),'exec'),scope)
        from omegaconf import OmegaConf
        paths={key:str(destination/name) for key,name in dict(bbx='bbx.pt',vitpose='vitpose.pt',slam='slam_results.pt',vit_features='vit_features.pt').items()}
        cfg=OmegaConf.create(dict(video_path=str(args.video),paths=paths,verbose=False,static_cam=args.static_camera,render_mhr=False,
            sam3d_ckpt_path=str(args.artifact/'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt'),sam3d_mhr_path=str(args.artifact/'inputs/mhr_data/mhr_model.pt')))
        scope['run_preprocess'](cfg);data=scope['load_data_dict'](cfg)
        from hydra import initialize_config_dir,compose
        from hydra.utils import instantiate
        with initialize_config_dir(config_dir=str(args.upstream/'configs'),version_base='1.3'):
            config=compose(config_name='demo_soma',overrides=['exp=gem_soma_regression','video_name=demo','ckpt_path=null'])
        model=instantiate(config.model,_recursive_=False)
        model.load_pretrained_model(str(args.artifact/'inputs/pretrained/gem_soma.ckpt'));model.cuda().eval()
        torch.manual_seed(311)
        with torch.no_grad():prediction=model.predict(data,static_cam=args.static_camera,postproc=True)
        value={group:{key:t.detach().cpu() for key,t in prediction[group].items()} for group in ('body_params_incam','body_params_global')}
        value['K_fullimg']=prediction['K_fullimg'].cpu()
        from gem.utils.soma_utils.soma_layer import SomaLayer
        body=SomaLayer(str(args.artifact/'inputs/soma_assets'),low_lod=True,device='cuda',identity_model_type='mhr',mode='warp').cuda().eval()
        with torch.no_grad():
            for group,space in [('body_params_incam','camera'),('body_params_global','world')]:
                result=body(**{key:t.cuda() for key,t in value[group].items()})
                value['joints_'+space]=result['joints'].cpu();value['vertices_'+space]=result['vertices'].cpu()
    else:
        from motius import Pipeline
        pipeline=Pipeline.from_pretrained(str(args.artifact),bundle_kwargs={'device':'cuda'})
        result=pipeline.infer_monocular_motion_capture(args.video,destination,seed=311,deterministic=True,static_camera=args.static_camera,materialize_geometry=True)
        from motius.pipelines.gem_x.parser import load_gem_x_payload
        payload=load_gem_x_payload(destination/args.video.stem/'hpe_results.pt')
        value={key:payload[key] for key in ['body_params_incam','body_params_global','K_fullimg']}
        track=result.tracks[0]
        for space in ['camera','world']:
            value['joints_'+space]=torch.from_numpy(getattr(track,'joints_'+space))
            value['vertices_'+space]=torch.from_numpy(getattr(track,'vertices_'+space))
        assert result.original_fps==30 and result.output_fps==30
    torch.save(value,args.output_dir/(args.worker+'.pt'))
    (args.output_dir/(args.worker+'-runtime.json')).write_text(json.dumps({'sparse_map_calls':policy.calls}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','dinov3','video','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--static-camera',action='store_true');parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','dinov3','video','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    run=args.output_dir/('run-'+uuid.uuid4().hex[:12]);run.mkdir();path=args.output_dir/'video_parity.json'
    report=dict(complete=False,comparison='Original full GEM-X cold-video preprocessing, source GEM prediction and source SOMA materialization versus the public native Auto Pipeline',runtime_policy='Both execute unchanged original matrix products with PyTorch deterministic algorithms plus the same frozen sparse-map dispatch; unconstrained CUDA sparse/index-add kernels are nondeterministic even in original source',static_camera=args.static_camera,video_sha256=digest(args.video),upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),run_directory=str(run),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_HOME':str(args.upstream/'inputs/torch_home')}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        command=[sys.executable,__file__,'--worker',kind,'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--dinov3',str(args.dinov3),'--video',str(args.video),'--output-dir',str(run)]
        if args.static_camera:command.append('--static-camera')
        subprocess.run(command,cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(run/'upstream.pt',weights_only=True),torch.load(run/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__,runtime={kind:json.loads((run/(kind+'-runtime.json')).read_text()) for kind in ['upstream','native']})
    path.write_text(json.dumps(report,indent=2)+'\n');print('All public GEM-X video comparisons exact under the declared reproducible runtime')

if __name__=='__main__':main()
