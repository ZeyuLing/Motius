"""Compare the complete original numeric static-video graph with native capture."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,uuid
from pathlib import Path
from verify_prompthmr_network_parity import compare


def tensors(value):
    import torch,numpy as np
    if isinstance(value,dict):return {str(key):tensors(item) for key,item in value.items()}
    if isinstance(value,np.ndarray):return torch.from_numpy(value.copy())
    if isinstance(value,(np.number,bool,int,float)):return torch.as_tensor(value)
    if isinstance(value,(list,tuple)):return [tensors(item) for item in value]
    if value is None:return None
    if isinstance(value,str):return value
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    raise TypeError(type(value))


def compare_full(original,native,path,rows):
    import torch
    if isinstance(original,dict):
        assert original.keys()==native.keys(),path
        for key in original:compare_full(original[key],native[key],path+'/'+key,rows)
    elif isinstance(original,(list,tuple)):
        assert len(original)==len(native),path
        for index,(left,right) in enumerate(zip(original,native)):compare_full(left,right,path+'/'+str(index),rows)
    elif isinstance(original,torch.Tensor):compare(original,native,path,rows)
    else:assert original==native,path


def worker(args):
    import torch,numpy as np,random
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    if args.dynamic_camera:torch.use_deterministic_algorithms(True)
    random.seed(1234);np.random.seed(1234);torch.manual_seed(1234)
    if args.worker=='upstream':
        from reference_prompthmr import load_reference
        Pipeline=load_reference(args)
        pipeline=Pipeline(static_cam=not args.dynamic_camera)
        result=pipeline(str(args.video),str(args.output_dir/'upstream-cache'),static_cam=not args.dynamic_camera,save_only_essential=False)
    else:
        from motius.pipelines.prompthmr import PromptHMRPipeline
        pipeline=PromptHMRPipeline.from_pretrained(args.artifact,body_models_root=args.body_models_root)
        result=pipeline(args.video,output_dir=args.output_dir/'native-cache',static_camera=not args.dynamic_camera,return_raw=True,deterministic=args.dynamic_camera)
    torch.save(tensors(result),args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','body-models-root','video','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--compare-only',action='store_true')
    parser.add_argument('--dynamic-camera',action='store_true');args=parser.parse_args()
    for name in ['upstream','artifact','body_models_root','video','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    run=args.output_dir if args.compare_only else args.output_dir/('run-'+uuid.uuid4().hex[:12]);run.mkdir(exist_ok=True)
    mode='dynamic' if args.dynamic_camera else 'static'
    path=args.output_dir/'video_parity.json';report=dict(complete=False,comparison=f'Complete original {mode}-camera numeric video reconstruction versus native public Pipeline, including real frontends, ROI mutation/UDP keypoints, image/temporal fusion, gravity/floor fitting and original post-optimization',reference_boundary='Pristine original Pipeline method ordering and numeric stages; only import/model-factory locations and checkpoint paths are supplied explicitly; optional exporters omitted after authoritative results.pkl',scope=f'{mode.capitalize()} camera, published SAM 2 path and default fitting/optimization on one public video; full-dataset metrics and artifact export require separate validation',runtime_policy='Same cuDNN deterministic settings in both workers; dynamic camera additionally enables torch deterministic algorithms for SLAM scatter operations',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),run_directory=str(run),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.compare_only else ['upstream','native']):
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_HOME':str(args.upstream.parent/'prompthmr-torchhub')}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        command=[sys.executable,__file__,'--worker',kind,'--upstream',str(args.upstream),'--artifact',str(args.artifact),'--body-models-root',str(args.body_models_root),'--video',str(args.video),'--output-dir',str(run)]
        if args.dynamic_camera:command.append('--dynamic-camera')
        subprocess.run(command,cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare_full(torch.load(run/'upstream.pt',weights_only=True),torch.load(run/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print(f'All complete PromptHMR {mode}-video numeric comparisons exact')

if __name__=='__main__':main()
