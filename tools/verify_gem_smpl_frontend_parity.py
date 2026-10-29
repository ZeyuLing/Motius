"""Compare original GEM-SMPL video preprocessing with native vision/processing."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,uuid
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    path=args.output_dir/args.worker;path.mkdir(parents=True,exist_ok=True)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        sys.path.insert(0,str(args.upstream/'scripts/demo'))
        from demo_utils import detect_and_track,CocoPoseExtractor,get_image_features,get_camera_static
        from demo_smpl_hpe import assemble_data
        from gem.utils.video_io_utils import get_video_lwh,read_video_np
        length,width,height=get_video_lwh(str(args.video))
        boxes=detect_and_track(str(args.video),str(path))
        frames=read_video_np(str(args.video))
        extractor=CocoPoseExtractor(device='cuda');keypoints=extractor.extract(frames,boxes,batch_size=32)
        del frames,extractor
        features,mask=get_image_features(str(args.video),boxes,str(args.artifact/'inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt'),str(path))
        rotation,angular,translation,intrinsics=get_camera_static(length,width,height)
        data=assemble_data(keypoints,boxes,intrinsics,angular,translation,rotation,features,mask,False)
    else:
        from motius.pipelines.gem_smpl.frontend import prepare_video
        data=prepare_video(args.video,path,args.artifact,'cuda',False)
    data.pop('meta');torch.save(data,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','video','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','video','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    run=args.output_dir/('run-'+uuid.uuid4().hex[:12]);run.mkdir()
    path=args.output_dir/'frontend_parity.json'
    report=dict(complete=False,comparison='Original GEM-SMPL cold video detection, exact COCO-17 heatmaps/crops, HMR2 features and camera data',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),video_sha256=digest(args.video),run_directory=str(run),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,'--worker',kind,'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--video',str(args.video),'--output-dir',str(run)],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(run/'upstream.pt',weights_only=True),torch.load(run/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GEM-SMPL frontend comparisons exact')

if __name__=='__main__':main()
