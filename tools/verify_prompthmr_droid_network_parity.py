"""Verify original DROID-Calib encoders and recurrent/graph update versus native roots."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_prompthmr_network_parity import compare


def worker(args):
    import torch
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream/'pipeline/droidcalib/droid_slam'))
        from droid_net import DroidNet
        model=DroidNet()
        state=torch.load(args.artifact/'data/pretrain/droidcalib.pth',map_location='cpu',weights_only=False)
        state={key.replace('module.',''):value for key,value in state.items()}
        for key in ['update.weight.2.weight','update.weight.2.bias','update.delta.2.weight','update.delta.2.bias']:state[key]=state[key][:2]
        model.load_state_dict(state,strict=True);model.cuda().eval()
    else:
        from motius.models.vision.prompthmr.droid.bundle import PromptDroidBundle
        model=PromptDroidBundle(args.artifact/'data/pretrain/droidcalib.pth')
    output={}
    for case,batch_size in enumerate([1,2]):
        torch.manual_seed(151+case);images=torch.randn(batch_size,4,3,64,80,device='cuda')
        with torch.no_grad(),torch.amp.autocast('cuda'):
            features=model.fnet(images);context=model.cnet(images)
            net,inp=context.split([128,128],dim=2);net=net.tanh();inp=inp.relu()
            corr=torch.randn(batch_size,4,196,8,10,device='cuda')
            flow=torch.randn(batch_size,4,4,8,10,device='cuda')
            one=model.update(net,inp,corr)
            graph=model.update(net,inp,corr,flow=flow,ii=torch.tensor([0,1,0,1],device='cuda'),jj=torch.tensor([1,0,1,0],device='cuda'))
        output[str(case)]=dict(features=features.cpu(),context=context.cpu(),one={str(i):value.cpu() for i,value in enumerate(one)},graph={str(i):value.cpu() for i,value in enumerate(graph)})
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for name in ['upstream','artifact','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'network_parity.json';report=dict(complete=False,comparison='Original DROID-Calib released feature/context encoders, GRU update and graph aggregation versus native neural roots',runtime_policy='Same PyTorch deterministic algorithms and cuDNN settings in both workers',scope='Neural components in FP16 autocast, batches 1/2; full camera odometry/optimization requires a separate certificate',tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All DROID-Calib neural comparisons exact')

if __name__=='__main__':main()
