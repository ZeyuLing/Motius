"""GEM-X video preprocessing using native detector, pose and body-token networks."""
from pathlib import Path
import numpy as np
import torch
from motius.pipelines.gem_video import get_video_lwh,read_video_np
from motius.motion.representation.gem_x.geometry import get_bbx_xys_from_xyxy,compute_cam_angvel,compute_cam_tvel,normalize_T_w2c
from motius.motion.representation.gem_x.camera import estimate_K,get_a_pred_cam
def smooth_bbx_xyxy(bbx_xyxy, window=5):
    """Apply moving-average smoothing to a (L, 4) bounding-box sequence."""
    if bbx_xyxy.shape[0] <= window:
        return bbx_xyxy
    kernel = torch.ones(1, 1, window, dtype=bbx_xyxy.dtype) / window
    padded = bbx_xyxy.T.unsqueeze(1)
    pad_size = window // 2
    padded = torch.nn.functional.pad(padded, (pad_size, pad_size), mode='replicate')
    smoothed = torch.nn.functional.conv1d(padded, kernel).squeeze(1).T
    return smoothed

@torch.no_grad()
def prepare_video(video,preprocess_dir,artifact_root,device='cuda',static_camera=False):
    root=Path(artifact_root);path=Path(preprocess_dir);path.mkdir(parents=True,exist_ok=True)
    video=str(video);length,width,height=get_video_lwh(video)
    cache=path/'bbx.pt'
    if not cache.exists():
        from .tracking import YOLOXDetector,detect_and_track
        frames=read_video_np(video)
        detector=YOLOXDetector(str(root/'inputs/checkpoints/yolox/yolox_x_8xb8-300e_humanart-a39d44ed.onnx'),device=str(device))
        xyxy,_=detect_and_track(frames,detector)
        xyxy=smooth_bbx_xyxy(torch.from_numpy(xyxy).float(),window=5)
        xyxy[:,[0,2]]=xyxy[:,[0,2]].clamp(0,width-1)
        xyxy[:,[1,3]]=xyxy[:,[1,3]].clamp(0,height-1)
        boxes=get_bbx_xys_from_xyxy(xyxy,base_enlarge=1.2).float()
        torch.save(dict(bbx_xyxy=xyxy,bbx_xys=boxes),cache)
        del frames,detector
    boxes=torch.load(cache,map_location='cpu',weights_only=True)['bbx_xys']
    cache=path/'vitpose.pt'
    if not cache.exists():
        from .pose import VitPoseExtractor
        pose=VitPoseExtractor(root/'inputs/checkpoints/vitpose/vitpose.pth',device=str(device))
        frames=read_video_np(video)
        keypoints=pose.extract(frames,boxes,img_ds=1.,path_type='np')
        torch.save(keypoints,cache)
        del pose,frames
    keypoints=torch.load(cache,map_location='cpu',weights_only=True)
    if isinstance(keypoints,tuple):keypoints=keypoints[0]
    cache=path/'slam_results.pt'
    if static_camera or not cache.exists():
        length,_,_=get_video_lwh(video)
        torch.save(np.repeat(np.eye(4,dtype=np.float32)[None],length,axis=0),cache)
    if static_camera:
        rotation=torch.eye(3).repeat(length,1,1);translation=torch.zeros(length,3);transform=torch.eye(4).repeat(length,1,1)
    else:
        trajectory=torch.as_tensor(torch.load(cache,map_location='cpu',weights_only=False)).float()
        transform=normalize_T_w2c(trajectory);rotation=transform[:,:3,:3];translation=transform[:,:3,3]
    intrinsics=estimate_K(width,height).repeat(length,1,1)
    cache=path/'vit_features.pt'
    if not cache.exists():
        from .features import SAM3DBExtractor
        extractor=SAM3DBExtractor(root/'inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt',root/'inputs/mhr_data/mhr_model.pt',device=str(device))
        features=extractor.extract_video_features(video,boxes)
        camera=get_a_pred_cam(features['transls'],boxes,intrinsics)
        torch.save(dict(pose_tokens=features['pose_tokens'],pred_cam=camera),cache)
    features=torch.load(cache,map_location='cpu',weights_only=True)
    return dict(meta=[{'vid':Path(video).stem}],length=torch.tensor(length),bbx_xys=boxes.clone(),kp2d=keypoints,K_fullimg=intrinsics,
        cam_angvel=compute_cam_angvel(rotation),cam_tvel=compute_cam_tvel(translation),R_w2c=rotation,T_w2c=transform,
        f_imgseq=features['pose_tokens'],noisy_pred_cam=features.get('pred_cam'),has_text=torch.tensor([False]),
        mask={key:torch.full((length,),value,dtype=torch.bool) for key,value in dict(valid=True,has_img_mask=True,has_2d_mask=True,has_cam_mask=True,has_audio_mask=False,has_music_mask=False).items()})
