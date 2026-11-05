"""Compare independent original PromptHMR image/body and temporal graphs."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_bailando_source_parity import compare as compare_tensors


def compare(original,native,path,rows):
    if isinstance(original,dict):
        assert original.keys()==native.keys(),path
        for key in original:compare(original[key],native[key],path+'/'+str(key),rows)
    elif isinstance(original,(list,tuple)):
        assert len(original)==len(native),path
        for index,(left,right) in enumerate(zip(original,native)):compare(left,right,path+'/'+str(index),rows)
    else:compare_tensors(original,native,path,rows)


def worker(args):
    import torch,numpy as np
    from omegaconf import OmegaConf
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream));sys.path.insert(0,str(args.upstream/'pipeline/gvhmr'))
        from prompt_hmr import load_model
        from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
        image=load_model(str(args.artifact/'checkpoint.ckpt'))
        cfg=OmegaConf.load(args.artifact/'video/prhmr_release_002.yaml')
        video=DemoPL(cfg.model.pipeline,str(args.body_models_root/'smplx/SMPLX_NEUTRAL.npz')).cuda().eval()
        video.load_pretrained_model(str(args.artifact/'video/phmr_b1b2.ckpt'))
    else:
        from motius.models.prompthmr.bundle import PromptHMRBundle
        from motius.motion.body_models.prompthmr.image import PromptHMRBodyGeometry
        from motius.pipelines.prompthmr.image import PromptHMRImagePrediction
        from motius.pipelines.prompthmr.video import PromptHMRVideoPrediction
        bundle=PromptHMRBundle(args.artifact,device='cuda')
        geometry=PromptHMRBodyGeometry(args.body_models_root,mapping_path=args.artifact/'data/body_models/smplx2smpl.pkl',joint_mapping_path=args.artifact/'data/body_models/smplx2smpl_joints.npy')
        state=torch.load(args.artifact/'checkpoint.ckpt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
        geometry.load_state_dict({key:value for key,value in state.items() if key.startswith(('smpl.','smplx.','smplx2smpl'))},strict=True)
        image=PromptHMRImagePrediction(bundle,geometry)
        video=PromptHMRVideoPrediction(bundle,str(args.body_models_root/'smplx/SMPLX_NEUTRAL.npz'))
    def cpu(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu()
        if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [cpu(v) for v in value]
        return value
    results={}
    for case,(batch_size,people,text,mask) in enumerate([(1,1,False,False),(1,2,True,True),(2,1,False,True)]):
        torch.manual_seed(117+case);batch=[]
        for index in range(batch_size):
            item=dict(image=torch.randn(3,896,896),cam_int=torch.tensor([[[900.,0.,448.],[0.,900.,448.],[0.,0.,1.]]]),
                boxes=torch.tensor([[100.,120.,600.,800.,1.]]).repeat(people,1),interaction=people>1)
            if text:item['text']=['a person standing','NULL'][:people]
            if mask:item['masks']=(torch.rand(people,1,256,256)>.4).float()
            batch.append(item)
        with torch.no_grad(),torch.amp.autocast('cuda'):outputs=image(batch)
        results['image_'+str(case)]=cpu(outputs)
    for case,(batch_size,frames) in enumerate([(1,16),(2,24)]):
        torch.manual_seed(83+case);device='cuda'
        inputs=dict(length=torch.tensor([frames,frames-3][:batch_size],device=device),obs=torch.randn(batch_size,frames,17,3,device=device),
            bbx_xys=torch.tensor([448.,448.,700.],device=device).repeat(batch_size,frames,1),
            K_fullimg=torch.tensor([[900.,0.,448.],[0.,900.,448.],[0.,0.,1.]],device=device).repeat(batch_size,frames,1,1),
            cam_angvel=torch.tensor([1.,0.,0.,0.,1.,0.],device=device).repeat(batch_size,frames,1),f_imgseq=torch.randn(batch_size,frames,2,1024,device=device))
        with torch.no_grad(),torch.amp.autocast('cuda'):
            output=video.pipeline.forward(inputs,train=False,postproc=False,static_cam=True) if args.worker=='upstream' else video.forward(inputs,static_cam=True)
        results['video_'+str(case)]=cpu(output)
    torch.save(results,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['upstream','artifact','body-models-root','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--compare-only',action='store_true')
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','artifact','body_models_root','output_dir']:setattr(args,key,getattr(args,key).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'network_parity.json';report=dict(complete=False,comparison='Independent original PromptHMR with released image and official SHA-verified BEDLAM1+2 temporal weights versus native neural/geometry composition',scope='Prepared normalized image/prompt tensors and temporal features; full video frontend, world fitting and training require separate verification',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.compare_only else ['upstream','native']):
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All PromptHMR prepared network and body comparisons exact')

if __name__=='__main__':main()
