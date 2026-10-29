"""Compare complete original masked DROID mapping, BA and trajectory filling."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors,compare_full


def worker(args):
    import torch,numpy as np,cv2
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True);torch.manual_seed(1234);np.random.seed(1234)
    frames=[];reader=cv2.VideoCapture(str(args.video))
    while True:
        ok,image=reader.read()
        if not ok:break
        frames.append(image[...,::-1].copy())
    reader.release();images=np.stack(frames)
    masks=torch.load(args.masks,weights_only=True)['masks'].bool()
    if args.worker=='upstream':
        from reference_prompthmr import definitions
        import argparse as arglib
        sys.path.insert(0,str(args.upstream/'pipeline/droidcalib/droid_slam'))
        from droid import Droid
        scope=dict(np=np,torch=torch,cv2=cv2)
        from torchvision.transforms import Resize
        scope.update(Resize=Resize)
        definitions(args.upstream/'pipeline/camera/slam_utils.py',{'est_calib','image_stream','preprocess_masks','get_dimention'},scope)
        defaults=json.loads((Path(__file__).resolve().parents[1]/'motius/configuration/prompthmr/droid.json').read_text())
        defaults['weights']=str(args.artifact/'data/pretrain/droidcalib.pth')
        scope.update(Droid=Droid,slam_args=arglib.Namespace(**defaults))
        definitions(args.upstream/'pipeline/camera/masked_droid_slam.py',{'run_slam'},scope)
        droid,trajectory,intrinsics,timestamps=scope['run_slam'](images,masks=masks,opt_intr=True)
    else:
        from motius.models.vision.prompthmr.droid.bundle import PromptDroidBundle
        from motius.pipelines.prompthmr.odometry.slam import run_slam
        model=PromptDroidBundle(args.artifact/'data/pretrain/droidcalib.pth')
        droid,trajectory,intrinsics,timestamps=run_slam(model,images,masks=masks,opt_intr=True)
    count=droid.video.counter.value
    output=dict(trajectory=trajectory,intrinsics=intrinsics,timestamps=timestamps,
        poses=droid.video.poses[:count],disparities=droid.video.disps[:count],upsampled_disparities=droid.video.disps_up[:count])
    torch.save(tensors(output),args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','video','masks','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for name in ['upstream','artifact','video','masks','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'odometry_parity.json';report=dict(complete=False,comparison='Complete original masked DROID mapping, intrinsics optimization, bundle adjustment and trajectory filling versus native bundle/composition',scope='Public 30-frame video, actual person masks, focal intrinsics optimization; metric scale requires a separate depth/scale comparison',tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All complete masked DROID comparisons exact')

if __name__=='__main__':main()
