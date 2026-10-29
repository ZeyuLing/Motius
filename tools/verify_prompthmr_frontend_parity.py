"""Compare original PromptHMR camera, detection, pose, segmentation and SAM tracking."""
from __future__ import annotations
import argparse,json,os,subprocess,sys,importlib.util
from pathlib import Path
from verify_prompthmr_network_parity import compare


def source_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


def worker(args):
    import torch,numpy as np,cv2
    from PIL import Image
    from torchvision.transforms import Compose,ToTensor,Normalize
    from omegaconf import OmegaConf
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    cap=cv2.VideoCapture(str(args.video));images=[]
    for _ in range(8):
        ok,frame=cap.read()
        if not ok:break
        images.append(frame[...,::-1].copy())
    cap.release();images=np.stack(images);assert len(images)==8
    transform=Compose([ToTensor(),Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    device='cuda';weights=args.artifact/'data/pretrain'
    if args.worker=='upstream':
        sys.path.insert(0,str(args.upstream/'pipeline'))
        from spec.cam_calib import CameraRegressorNetwork,convert_preds_to_angles
        camera=CameraRegressorNetwork().cuda().eval()
        state=torch.load(weights/'camcalib_sa_biased_l2.ckpt',map_location='cpu',weights_only=False)['state_dict']
        camera.load_state_dict({key.removeprefix('model.'):value for key,value in state.items()},strict=True)
        from torchvision.models.segmentation import deeplabv3_resnet50
        segmenter=deeplabv3_resnet50(weights=None,weights_backbone=None,aux_loss=True).cuda().eval()
        segmenter.load_state_dict(torch.load(weights/'deeplabv3_resnet50.pth',map_location='cpu',weights_only=False),strict=True)
        from detectron2.config import get_cfg
        cfg=get_cfg();cfg.merge_from_file(str(args.upstream/'pipeline/detectron2/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml'))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=.25;cfg.MODEL.WEIGHTS=str(weights/'sam2_ckpts/keypoint_rcnn_5ad38f.pkl')
        detection_module=source_module('original_prompt_detection',args.upstream/'pipeline/utils_detectron2.py')
        detector=detection_module.DefaultPredictor(cfg)
        vit=args.upstream/'pipeline/detector/ViTPose/easy_vitpose';sys.path.insert(0,str(vit))
        from vit_models.model import ViTPose
        import runpy
        pose_cfg=runpy.run_path(str(vit/'configs/ViTPose_common.py'))['model_huge'];pose_cfg['keypoint_head']['out_channels']=25
        pose=ViTPose(pose_cfg).cuda().eval()
        state=torch.load(weights/'vitpose-h-coco_25.pth',map_location='cpu',weights_only=False);pose.load_state_dict(state.get('state_dict',state),strict=True)
        sam_module=source_module('original_prompt_sam_tracker',args.upstream/'pipeline/detector/sam2_video_predictor.py')
        from hydra.utils import instantiate
        cfg=OmegaConf.load(args.upstream/'pipeline/sam2/sam2_hiera_t.yaml').model
        cfg._target_='original_prompt_sam_tracker.SAM2VideoPredictor'
        cfg.sam_mask_decoder_extra_args=dict(dynamic_multimask_via_stability=True,dynamic_multimask_stability_delta=.05,dynamic_multimask_stability_thresh=.98)
        cfg.binarize_mask_from_pts_for_mem_enc=True;cfg.fill_hole_area=8
        tracker=instantiate(cfg,_recursive_=True).cuda().eval()
        tracker.load_state_dict(torch.load(weights/'sam2_ckpts/sam2_hiera_tiny.pt',map_location='cpu',weights_only=False)['model'],strict=True)
    else:
        from motius.models.vision.prompthmr.bundle import PromptCameraBundle,PromptPoseBundle,PromptSegmentationBundle,PromptDetectorBundle,PromptMaskBundle
        from motius.pipelines.prompthmr.calibration import convert_preds_to_angles
        from motius.pipelines.prompthmr.detection import PromptDetector
        from motius.pipelines.prompthmr.tracking import PromptMaskTracker
        camera=PromptCameraBundle(weights/'camcalib_sa_biased_l2.ckpt')
        segmenter=PromptSegmentationBundle(weights/'deeplabv3_resnet50.pth')
        detector=PromptDetector(PromptDetectorBundle(weights/'sam2_ckpts/keypoint_rcnn_5ad38f.pkl'))
        pose=PromptPoseBundle(weights/'vitpose-h-coco_25.pth')
        tracker=PromptMaskTracker(PromptMaskBundle(weights/'sam2_ckpts/sam2_hiera_tiny.pt'))
    output={}
    with torch.no_grad():
        batch=torch.stack([transform(Image.fromarray(image)) for image in images[:2]]).to(device)
        distributions=camera(batch);angles=convert_preds_to_angles(*distributions,loss_type='softargmax_l2')
        output['camera']={str(i):value.cpu() for i,value in enumerate([*distributions,*angles])}
        batch=torch.stack([transform(Image.fromarray(image)) for image in images]).to(device)
        mask=segmenter(batch)['out'];output['segmentation']={'logits':mask.cpu(),'labels':mask.argmax(1).cpu()}
        instances=detector(images[0][...,::-1])['instances'].to('cpu')
        output['detection']=dict(boxes=instances.pred_boxes.tensor,scores=instances.scores,keypoints=instances.pred_keypoints)
        batch=torch.stack([transform(Image.fromarray(cv2.resize(image,(192,256)))) for image in images[:2]]).to(device)
        output['pose']=pose(batch).cpu()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        state=tracker.init_state(video_frames=images,async_loading_frames=False,offload_video_to_cpu=True)
        height,width=images.shape[1:3]
        points=np.array([[width*.3,height*.15],[width*.7,height*.95]],dtype=np.float32)
        _,_,masks=tracker.add_new_points(state,frame_idx=0,obj_id=1,points=points,labels=np.array([2,3]))
        output['tracking']={'initial':masks.float().cpu()}
        for frame,ids,masks in tracker.propagate_in_video(state):
            output['tracking'][str(frame)]=masks.float().cpu()
    torch.save(output,args.output_dir/(args.worker+'.pt'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['upstream','artifact','video','output-dir']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--worker',choices=['upstream','native']);parser.add_argument('--compare-only',action='store_true');args=parser.parse_args()
    for name in ['upstream','artifact','video','output_dir']:setattr(args,name,getattr(args,name).resolve())
    if args.worker:worker(args);return
    root=Path(__file__).resolve().parents[1];args.output_dir.mkdir(parents=True,exist_ok=True)
    path=args.output_dir/'frontend_parity.json';report=dict(complete=False,comparison='Original camera distributions/angles, person segmentation, Detectron2 predictions, COCO25 heatmaps and eight-frame actual SAM 2 state propagation versus native neural bundles and tracker',scope='Network frontends and tracking on public video; complete crop/heatmap-coordinate processing, camera SLAM and end-to-end public video require separate certificates',reference_boundary='Original standalone modules and exact original YAML architecture loaded directly to avoid importing unused application/export dependencies; no neural substitution',upstream_commit=subprocess.check_output(['git','-C',str(args.upstream),'rev-parse','HEAD'],text=True).strip(),tests=[])
    path.write_text(json.dumps(report,indent=2)+'\n')
    for kind in ([] if args.compare_only else ['upstream','native']):
        env={**os.environ,'PYTHONPATH':str(root),'CUBLAS_WORKSPACE_CONFIG':':4096:8'}
        if kind=='upstream':env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        subprocess.run([sys.executable,__file__,*sys.argv[1:],'--worker',kind],cwd=args.upstream if kind=='upstream' else root,env=env,check=True)
    import torch
    try:compare(torch.load(args.output_dir/'upstream.pt',weights_only=True),torch.load(args.output_dir/'native.pt',weights_only=True),'',report['tests'])
    finally:path.write_text(json.dumps(report,indent=2)+'\n')
    report.update(complete=True,torch_version=torch.__version__);path.write_text(json.dumps(report,indent=2)+'\n')
    print('All PromptHMR frontend comparisons exact')

if __name__=='__main__':main()
