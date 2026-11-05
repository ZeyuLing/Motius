"""Compare Metric3D's original published ViT-Large graph with native networks."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_video_parity import tensors,compare_full


def worker(args):
    import torch
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(1234)
    checkpoint=args.artifact/'data/pretrain/metric_depth_vit_large_800k.pth'
    if args.worker=='upstream':
        source=args.upstream/'pipeline/yvanyin_metric3d_main'
        sys.path.insert(0,str(source))
        from mmengine import Config
        from mono.model.monodepth_model import get_configured_monodepth_model
        cfg=Config.fromfile(str(source/'mono/configs/HourglassDecoder/vit.raft5.large.py'))
        model=get_configured_monodepth_model(cfg)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)['model_state_dict']
        missing=model.load_state_dict(state,strict=False)
        assert missing.missing_keys==['depth_model.encoder.mask_token'] and not missing.unexpected_keys,missing
        model=model.cuda().half().eval();predict=model.inference
    else:
        from motius.models.vision.prompthmr.metric3d.bundle import PromptMetricDepthBundle
        model=PromptMetricDepthBundle(checkpoint);predict=model
    output={}
    for batch in [1,2]:
        torch.manual_seed(781+batch)
        image=torch.randn(batch,3,616,1064,device='cuda',dtype=torch.float16)
        with torch.inference_mode():result=predict({'input':image})
        output[str(batch)]=tensors(result)
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'network_parity.json'
    report=dict(complete=False,comparison='Original Metric3D ViT-Large released checkpoint, all recurrent depth/confidence/normal outputs versus native neural graph',scope='FP16 neural inference, full 616x1064 input, batches 1/2; metric scale fitting and odometry require separate validation',checkpoint_missing_constant='Original checkpoint omits only the unused pretraining mask token, whose source initialization is exact zeros',tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare_full(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All Metric3D neural comparisons exact')

if __name__=='__main__':main()
