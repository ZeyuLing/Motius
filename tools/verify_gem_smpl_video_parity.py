"""Verify public native GEM-SMPL video inference against independent pinned source."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,uuid
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch,numpy as np
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    destination=args.output_dir/args.worker;destination.mkdir(parents=True,exist_ok=True)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream));sys.path.insert(0,str(args.upstream/'scripts/demo'))
        from demo_utils import detect_and_track,CocoPoseExtractor,get_image_features,get_camera_static,load_model
        from demo_smpl_hpe import assemble_data
        from gem.utils.video_io_utils import get_video_lwh,read_video_np
        from gem.utils.smplx_utils import make_smplx
        # The source's original video decode/crop order is preserved.
        length,width,height=get_video_lwh(str(args.video));boxes=detect_and_track(str(args.video),str(destination))
        frames=read_video_np(str(args.video));pose=CocoPoseExtractor(device='cuda')
        keypoints=pose.extract(frames,boxes,batch_size=32);del frames,pose
        features,mask=get_image_features(str(args.video),boxes,str(args.artifact/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt'),str(destination))
        rotation,angular,translation,intrinsics=get_camera_static(length,width,height)
        data=assemble_data(keypoints,boxes,intrinsics,angular,translation,rotation,features,mask,args.static_camera)
        model=load_model(str(args.artifact/'inputs/pretrained/gem_smpl.ckpt'))
        torch.manual_seed(311)
        with torch.no_grad():pred=model.predict(data,static_cam=args.static_camera)
        value={group:{k:v.cpu() for k,v in pred[group].items()} for group in ('body_params_incam','body_params_global')}
        value['K_fullimg']=pred['K_fullimg'].cpu()
        from gem.utils.body_model import __file__ as asset_module
        asset=Path(asset_module).parent
        mapping=torch.load(asset/'smplx2smpl_sparse.pt',map_location='cuda',weights_only=True)
        regressor=torch.load(asset/'smpl_neutral_J_regressor.pt',map_location='cuda',weights_only=True)
        body=make_smplx('supermotion').cuda().eval()
        with torch.no_grad():
            for group,space in [('body_params_incam','camera'),('body_params_global','world')]:
                vertices=body(**{k:v.cuda() for k,v in value[group].items()}).vertices
                vertices=torch.stack([torch.matmul(mapping,frame) for frame in vertices])
                value['vertices_'+space]=vertices.cpu()
                value['joints_'+space]=(regressor@vertices).cpu()
    else:
        from motius.models.gem_smpl import GemSmplBundle
        from motius.pipelines.gem_smpl import GemSmplPipeline
        bundle=GemSmplBundle.from_pretrained(args.artifact,body_models_root=args.body_models_root,device='cuda')
        pipeline=GemSmplPipeline(bundle)
        result=pipeline.infer_monocular_motion_capture(args.video,destination,static_camera=args.static_camera,seed=311,deterministic=True)
        from motius.pipelines.gem_smpl.parser import load_gem_smpl_payload
        value=load_gem_smpl_payload(destination/args.video.stem/'smpl_params.pt')
        track=result.tracks[0]
        for space in ['camera','world']:
            value['vertices_'+space]=torch.from_numpy(getattr(track,'vertices_'+space))
            value['joints_'+space]=torch.from_numpy(getattr(track,'joints_'+space))
        assert result.original_fps==30 and result.output_fps==30
    torch.save(value,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','body-models-root','video','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--static-camera',action='store_true')
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','body_models_root','video','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    run=args.output_dir/('run-'+uuid.uuid4().hex[:12]);run.mkdir()
    path=args.output_dir/'video_parity.json'
    report=dict(complete=False,comparison='Full original cold video preprocessing, released GEM-SMPL diffusion/global/contact IK and original SMPL mesh materialization versus public native Pipeline',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),video_sha256=digest(args.video),static_camera=args.static_camera,run_directory=str(run),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        cmd=[sys.executable,__file__,'--worker',kind,'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--body-models-root',str(args.body_models_root),'--video',str(args.video),'--output-dir',str(run)]
        if args.static_camera:cmd.append('--static-camera')
        subprocess.run(cmd,cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(run/'upstream.pt',weights_only=True),torch.load(run/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All public GEM-SMPL video comparisons exact')

if __name__=='__main__':main()
