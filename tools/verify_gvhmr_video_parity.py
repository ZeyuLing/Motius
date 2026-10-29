"""Compare the public native GVHMR video pipeline with independent source results."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,uuid
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import cv2,numpy as np,torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.deterministic=True
    import pycolmap
    if hasattr(pycolmap,'set_random_seed'):pycolmap.set_random_seed(17)
    # Hydra's original unquoted video_name override rejects Unicode filenames.
    # An ASCII symlink preserves the exact input bytes for both independent runs.
    alias=args.output_dir/'input.mp4'
    if not alias.exists():alias.symlink_to(args.video)
    args.video=alias
    output={}
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from omegaconf import OmegaConf
        from hmr4d.model.gvhmr.pipeline.gvhmr_pipeline import Pipeline
        import importlib.util
        spec=importlib.util.spec_from_file_location('original_gvhmr_demo',args.upstream/'tools/demo/demo.py')
        demo=importlib.util.module_from_spec(spec);spec.loader.exec_module(demo)
        from hmr4d.utils.geo.hmr_cam import normalize_kp2d,estimate_K
        from hmr4d.utils.geo_transform import compute_cam_angvel
        from hmr4d.utils.video_io_utils import get_video_lwh
        from hmr4d.utils.smplx_utils import make_smplx
        config=OmegaConf.create(dict(endecoder_opt={'_target_':'hmr4d.model.gvhmr.utils.endecoder.EnDecoder','stats_name':'MM_V1_AMASS_LOCAL_BEDLAM_CAM'},normalize_cam_angvel=True,weights=None,static_conf=None))
        pipeline=Pipeline(config,OmegaConf.create({'_target_':'hmr4d.network.gvhmr.relative_transformer.NetworkEncoderRoPE'})).cuda().eval()
        state=torch.load(args.weights/'gvhmr/gvhmr_siga24_release.ckpt',map_location='cpu',weights_only=True)['state_dict']
        pipeline.load_state_dict({k.removeprefix('pipeline.'):v for k,v in state.items()},strict=True)
        body=make_smplx('supermotion').cuda().eval()
        asset=args.upstream/'hmr4d/utils/body_model'
        mapping=torch.load(asset/'smplx2smpl_sparse.pt',map_location='cuda',weights_only=True)
        regressor=torch.load(asset/'smpl_neutral_J_regressor.pt',map_location='cuda',weights_only=True)
    else:
        from motius.models.gvhmr import GVHMRBundle
        from motius.pipelines.gvhmr import GVHMRPipeline
        bundle=GVHMRBundle(args.weights,device='cuda')
        pipeline=GVHMRPipeline(bundle,body_model_path=args.body_model)
    for static in [args.static_camera]:
        cv2.setRNGSeed(17)
        if hasattr(pycolmap,'set_random_seed'):pycolmap.set_random_seed(17)
        if args.worker=='upstream':
            previous_argv=sys.argv
            sys.argv=['demo.py','--video',str(args.video),'--output_root',str(args.output_dir/f'upstream_static_{static}')]
            if static:sys.argv.append('--static_cam')
            try:cfg=demo.parse_args_to_cfg()
            finally:sys.argv=previous_argv
            demo.run_preprocess(cfg)
            data=demo.load_data_dict(cfg)
            frames=int(data['length']);intrinsic=data['K_fullimg'];angular=data['cam_angvel']
            batch=dict(length=data['length'][None].cuda(),obs=normalize_kp2d(data['kp2d'],data['bbx_xys'])[None].cuda(),
                bbx_xys=data['bbx_xys'][None].cuda(),K_fullimg=intrinsic[None].cuda(),cam_angvel=angular[None].cuda(),f_imgseq=data['f_imgseq'][None].cuda())
            with torch.no_grad():
                result=pipeline(batch,train=False,postproc=True,static_cam=static)
                value={}
                for name,group in [('camera','pred_smpl_params_incam'),('world','pred_smpl_params_global')]:
                    parameters={k:v[0] for k,v in result[group].items()}
                    vertices=body(**parameters).vertices
                    vertices=torch.stack([torch.matmul(mapping,frame) for frame in vertices])
                    joints=torch.einsum('jv,fvc->fjc',regressor,vertices)
                    value[name]={'parameters':{k:v.cpu() for k,v in parameters.items()},'vertices':vertices.cpu(),'joints':joints.cpu()}
                value['intrinsics']=intrinsic.cpu();value['cam_angvel']=angular.cpu()
                value['preprocess']={k:data[k].cpu() for k in ['bbx_xys','kp2d','f_imgseq']}
                value['pixels']=torch.from_numpy(demo.read_video_np(cfg.video_path))
        else:
            result=pipeline.infer_video(args.video,args.output_dir/f'native_static_{static}',static_camera=static,materialize_geometry=True)
            track=result.tracks[0]
            value={}
            for name in ['camera','world']:
                parameters=dict(body_pose=track.native_parameters['body_pose'],betas=track.native_parameters['betas'],
                    global_orient=track.native_parameters[f'global_orient_{name}'],transl=track.native_parameters[f'transl_{name}'])
                value[name]={'parameters':{k:torch.from_numpy(v) for k,v in parameters.items()},
                    'vertices':torch.from_numpy(getattr(track,f'vertices_{name}')),'joints':torch.from_numpy(getattr(track,f'joints_{name}'))}
            value['intrinsics']=torch.from_numpy(result.camera_intrinsics)
            preprocessed=args.output_dir/f'native_static_{static}'/args.video.stem/'preprocess'
            value['preprocess']={'bbx_xys':torch.load(preprocessed/'bbx.pt',weights_only=True)['bbx_xys'],
                'kp2d':torch.load(preprocessed/'vitpose.pt',weights_only=True),'f_imgseq':torch.load(preprocessed/'vit_features.pt',weights_only=True)}
            from motius.pipelines.gvhmr.video import read_video_np
            value['pixels']=torch.from_numpy(read_video_np(str(preprocessed.parent/'0_input_video.mp4')))
            # Native preprocessed odometry is independently available on disk.
            if static:value['cam_angvel']=compute_native_angular_identity(len(track.frame_ids))
            else:
                from motius.motion.representation.gvhmr.geometry import compute_cam_angvel
                trajectory=torch.load(args.output_dir/f'native_static_{static}'/args.video.stem/'preprocess/slam_results.pt',weights_only=False)
                value['cam_angvel']=compute_cam_angvel(torch.from_numpy(trajectory[:,:3,:3])).cpu()
        output[f'static_camera_{static}']=value
    torch.save(output,args.output_dir/f'{args.worker}_static_{args.static_camera}.pt')


def compute_native_angular_identity(frames):
    import torch
    from motius.motion.representation.gvhmr.geometry import compute_cam_angvel
    return compute_cam_angvel(torch.eye(3).repeat(frames,1,1))


def main():
    parser=argparse.ArgumentParser()
    for key in ['upstream','weights','video','frontend-dir','body-model','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native'])
    parser.add_argument('--static-camera',action='store_true');args=parser.parse_args()
    keys=['upstream','weights','video','frontend_dir','body_model','output_dir']
    for key in keys:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'video_parity.json'
    # Every run owns fresh case directories: source demo caches solely by frame count.
    run_directory=args.output_dir/('run-'+uuid.uuid4().hex[:12]);run_directory.mkdir()
    report=dict(complete=False,comparison='Public native GVHMR infer_video including independent video preprocessing, static/dynamic camera and full SMPL-X-to-SMPL materialization',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        video_sha256=digest(args.video),body_model_sha256=digest(args.body_model),run_directory=str(run_directory),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    import torch
    for static in [True,False]:
        for kind in ['upstream','native']:
            command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind]
            if static:command.append('--static-camera')
            for key in keys:command+=['--'+key.replace('_','-'),str(run_directory if key=='output_dir' else getattr(args,key))]
            subprocess.run(command,cwd=args.upstream if kind=='upstream' else root,
                env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1' if kind=='upstream' else '0'},check=True)
        try:compare(torch.load(run_directory/f'upstream_static_{static}.pt',weights_only=True),torch.load(run_directory/f'native_static_{static}.pt',weights_only=True),'',report['tests'])
        finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GVHMR public video pipeline comparisons exact')

if __name__=='__main__':main()
