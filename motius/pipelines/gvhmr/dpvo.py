"""Optional DPVO odometry using its installed neural implementation."""
from pathlib import Path
import cv2
import torch
from pytorch3d.transforms import quaternion_to_matrix
from motius.motion.representation.gvhmr.camera import estimate_K,convert_K_to_K4
from .video import get_video_lwh


def estimate_rotations(video,frontend,output):
    from dpvo.dpvo import DPVO
    from dpvo.config import cfg
    frames,width,height=get_video_lwh(video)
    settings=cfg.clone()
    config=Path(__file__).resolve().parents[2]/"configuration/gvhmr/dpvo.yaml"
    settings.merge_from_file(str(config));settings.BUFFER_SIZE=4000
    weights=frontend.checkpoint("dpvo/dpvo.pth")
    intrinsics=convert_K_to_K4(estimate_K(width,height)).to(frontend.device)
    capture=cv2.VideoCapture(video);solver=None;index=0
    try:
        while True:
            valid,image=capture.read()
            if not valid:break
            image=cv2.resize(image,None,fx=.5,fy=.5,interpolation=cv2.INTER_AREA)
            h,w=image.shape[:2];image=image[:h-h%16,:w-w%16]
            tensor=torch.from_numpy(image).permute(2,0,1).to(frontend.device)
            if solver is None:solver=DPVO(settings,str(weights),ht=tensor.shape[1],wd=tensor.shape[2],viz=False)
            solver(index,tensor,intrinsics*.5);index+=1
        if solver is None:raise ValueError("Video contains no frames")
        for _ in range(12):solver.update()
        trajectory=solver.terminate()[0]
    finally:capture.release()
    torch.save(trajectory,Path(output)/"slam_results.pt")
    quaternion=torch.from_numpy(trajectory[:,[6,3,4,5]])
    return quaternion_to_matrix(quaternion).mT
