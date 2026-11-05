"""Native GVHMR video orchestration with released preprocessing arithmetic."""
from pathlib import Path
import numpy as np
import torch
from .video import get_video_lwh,get_video_reader,get_writer
from .preprocessing import _bbox_xys_from_xyxy
from motius.motion.representation.gvhmr.camera import estimate_K,create_camera_sensor,get_bbx_xys_from_xyxy
from motius.motion.representation.gvhmr.geometry import compute_cam_angvel


class GVHMRFrontend:
    def __init__(self,artifact_root,device):
        self.artifact_root=Path(artifact_root)
        self.device=torch.device(device)

    def checkpoint(self,name):
        for root in [self.artifact_root/"inputs/checkpoints",self.artifact_root]:
            path=root/name
            if path.is_file():return path
        raise FileNotFoundError(f"GVHMR frontend checkpoint missing: {name}")

    def prepare(self,video,output_root,*,static_camera=False,focal_length_mm=None,
                bbox_xyxy=None,use_dpvo=False,verbose=False):
        output=Path(output_root).expanduser().resolve()/Path(video).stem
        preprocess=output/"preprocess";preprocess.mkdir(parents=True,exist_ok=True)
        copied=output/"0_input_video.mp4"
        if not copied.exists() or get_video_lwh(video)[0]!=get_video_lwh(copied)[0]:
            reader=get_video_reader(str(video));writer=get_writer(str(copied),fps=30,crf=23)
            try:
                for frame in reader:writer.write_frame(frame)
            finally:writer.close();reader.close()
        frames,width,height=get_video_lwh(str(copied))
        if bbox_xyxy is None:
            from .tracking import Tracker
            tracker=Tracker(self.checkpoint("yolo/yolov8x.pt"),device=self.device)
            boxes=tracker.get_one_track(str(copied)).float();del tracker
        else:
            boxes=np.asarray(bbox_xyxy,dtype=np.float32)
            _bbox_xys_from_xyxy(boxes)  # validate the caller's dense track
            if boxes.shape!=(frames,4):raise ValueError("bbox_xyxy must match the video frame count")
            boxes=torch.from_numpy(boxes)
        bbox=get_bbx_xys_from_xyxy(boxes,base_enlarge=1.2).float()
        torch.save(dict(bbx_xyxy=boxes,bbx_xys=bbox),preprocess/"bbx.pt")
        from .pose import VitPoseExtractor
        pose=VitPoseExtractor(self.checkpoint("vitpose/vitpose-h-multi-coco.pth"),device=self.device,tqdm_leave=verbose)
        keypoints=pose.extract(str(copied),bbox);del pose
        torch.save(keypoints,preprocess/"vitpose.pt")
        from .features import Extractor
        encoder=Extractor(self.checkpoint("hmr2/epoch=10-step=25000.ckpt"),device=self.device,tqdm_leave=verbose)
        features=encoder.extract_video_features(str(copied),bbox);del encoder
        torch.save(features,preprocess/"vit_features.pt")
        if static_camera:
            rotations=torch.eye(3).repeat(frames,1,1)
        elif use_dpvo:
            from .dpvo import estimate_rotations
            rotations=estimate_rotations(str(copied),self,preprocess)
        else:
            from .odometry import SimpleVO
            trajectory=SimpleVO(str(copied),scale=.5,step=8,method="sift",f_mm=focal_length_mm).compute()
            torch.save(trajectory,preprocess/"slam_results.pt")
            rotations=torch.from_numpy(trajectory[:,:3,:3])
        intrinsics=(estimate_K(width,height) if focal_length_mm is None else
                    create_camera_sensor(width,height,focal_length_mm)[2]).repeat(frames,1,1)
        data=dict(length=torch.tensor(frames),bbx_xys=bbox,kp2d=keypoints,K_fullimg=intrinsics,
                  cam_angvel=compute_cam_angvel(rotations),f_imgseq=features)
        return data,dict(bbx_xyxy=boxes,bbx_xys=bbox),output,copied
