"""Independent GVHMR tracking, affine crops, heatmaps and feature-token parity."""
from __future__ import annotations
import argparse,json,os,subprocess,sys
from pathlib import Path
from verify_intergen_source_parity import digest
from verify_bailando_source_parity import compare


def worker(args):
    import torch,numpy as np,cv2
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.deterministic=True
    cv2.setRNGSeed(17)
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream))
        from hmr4d.utils.preproc.tracker import Tracker
        from hmr4d.utils.preproc.vitfeat_extractor import Extractor,get_batch
        from hmr4d.utils.preproc.vitpose import VitPoseExtractor
        from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy
    else:
        from motius.pipelines.gvhmr.tracking import Tracker
        from motius.pipelines.gvhmr.features import Extractor,get_batch
        from motius.pipelines.gvhmr.pose import VitPoseExtractor
        from motius.motion.representation.gvhmr.camera import get_bbx_xys_from_xyxy
    if args.worker=='upstream':tracker=Tracker()
    else:tracker=Tracker(args.weights/'yolo/yolov8x.pt')
    boxes=tracker.get_one_track(str(args.video)).float();del tracker
    bbox=get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float()
    crops,crop_bbox=get_batch(str(args.video),bbox)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    if args.worker=='upstream':pose=VitPoseExtractor(tqdm_leave=False)
    else:pose=VitPoseExtractor(args.weights/'vitpose/vitpose-h-multi-coco.pth',tqdm_leave=False)
    keypoints=pose.extract(str(args.video),bbox);del pose
    if args.worker=='upstream':features=Extractor(tqdm_leave=False)
    else:features=Extractor(args.weights/'hmr2/epoch=10-step=25000.ckpt',tqdm_leave=False)
    intermediates={"backbone":[],"token_embedding":[],"decoder":[]}
    network=features.extractor if args.worker=='upstream' else features.extractor.model
    handles=[]
    for name,module in [("backbone",network.backbone),("token_embedding",network.smpl_head.transformer.to_token_embedding),
                        ("decoder",network.smpl_head.transformer)]:
        handles.append(module.register_forward_hook(lambda m,i,o,key=name:intermediates[key].append(o.detach().cpu())))
    layers={}
    for name,module in network.smpl_head.transformer.named_modules():
        if isinstance(module,(torch.nn.Linear,torch.nn.LayerNorm)):
            def capture_layer(m,inputs,value,key=name):
                if key not in layers:layers[key]=value[:2].detach().cpu().clone()
            handles.append(module.register_forward_hook(capture_layer))
    tokens=features.extract_video_features(str(args.video),bbox)
    for handle in handles:handle.remove()
    intermediates={key:torch.cat(value) for key,value in intermediates.items()}
    del features
    torch.save(dict(decoder_layers=layers,intermediates=intermediates,boxes=boxes,bbox=bbox,crops=crops,crop_bbox=crop_bbox,keypoints=keypoints,tokens=tokens),args.output_dir/f'{args.worker}.pt')


def main():
    parser=argparse.ArgumentParser()
    for key in ['upstream','weights','video','output-dir']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);args=parser.parse_args()
    for key in ['upstream','weights','video','output_dir']:setattr(args,key,getattr(args,key).resolve())
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.worker:worker(args);return
    for name in ['hmr2/epoch=10-step=25000.ckpt','vitpose/vitpose-h-multi-coco.pth','yolo/yolov8x.pt']:
        link=args.upstream/'inputs/checkpoints'/name;link.parent.mkdir(exist_ok=True,parents=True)
        if not link.exists():link.symlink_to(args.weights/name)
    root=Path(__file__).resolve().parents[1];path=args.output_dir/'frontend_parity.json'
    report=dict(complete=False,comparison='Original GVHMR frontends versus native tracker, shared ViT backbone, heatmap decoder and HMR2 feature-token path',
        upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),
        video_sha256=digest(args.video),weight_files={n:digest(args.weights/n) for n in ['hmr2/epoch=10-step=25000.ckpt','vitpose/vitpose-h-multi-coco.pth','yolo/yolov8x.pt']},tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ['upstream','native']:
        command=[sys.executable,str(Path(__file__).resolve()),'--worker',kind]
        for key in ['upstream','weights','video','output_dir']:command+=['--'+key.replace('_','-'),str(getattr(args,key))]
        subprocess.run(command,cwd=args.upstream if kind=='upstream' else root,env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8','TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD':'1' if kind=='upstream' else '0'},check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__)
    path.write_text(json.dumps(report,indent=2)+'\n');print('All GVHMR frontend tensors exact')

if __name__=='__main__':main()
