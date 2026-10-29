"""Published ROI padding, COCO25 heatmap decoding and keypoint layout conversion."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from .heatmaps import pad_image,keypoints_from_heatmaps


def convert_keypoints(values,source,target):
    names=json.loads((Path(__file__).resolve().parents[2]/'configuration/prompthmr/joint_names.json').read_text())
    input_names=names[source];output_names=names[target]
    output=np.zeros((values.shape[0],len(output_names),values.shape[-1]))
    for index,name in enumerate(output_names):
        if name in input_names:output[:,index]=values[:,input_names.index(name)]
    return output


class PromptPosePrediction:
    def __init__(self,bundle):
        self.bundle=bundle
        self.device=next(bundle.parameters()).device

    @torch.no_grad()
    def __call__(self,image,bbox,padding=10):
        # The source mutates the tracked ROI before constructing image prompts.
        bbox[[0,2]]=np.clip(bbox[[0,2]]+[-padding,padding],0,image.shape[1])
        bbox[[1,3]]=np.clip(bbox[[1,3]]+[-padding,padding],0,image.shape[0])
        bounds=bbox.astype(np.int32)
        crop=image[bounds[1]:bounds[3],bounds[0]:bounds[2]]
        crop,(left,top)=pad_image(crop,3/4)
        height,width=crop.shape[:2]
        tensor=cv2.resize(crop,[192,256],interpolation=cv2.INTER_LINEAR)/255
        tensor=((tensor-np.array([.485,.456,.406]))/np.array([.229,.224,.225])).transpose(2,0,1)[None].astype(np.float32)
        heatmaps=self.bundle(torch.from_numpy(tensor).to(self.device)).detach().cpu().numpy()
        coordinates,probability=keypoints_from_heatmaps(heatmaps,center=np.array([[width//2,height//2]]),scale=np.array([[width,height]]),unbiased=True,use_udp=True)
        keypoints=np.concatenate([coordinates[:,:,::-1],probability],axis=2)[0]
        keypoints[:,:2]+=bounds[:2][::-1]-[top,left]
        keypoints=keypoints.copy()
        keypoints[:,:2]=keypoints[:,:2][:,::-1]
        return keypoints

    def tracks(self,images,tracks):
        for track in tracks.values():
            points=np.array([self(images[index],track['bboxes'][index]) for index,_ in enumerate(track['frames'])])
            track['keypoints_2d']=convert_keypoints(points,'vitpose25','openpose')
            track['vitpose']=convert_keypoints(track['keypoints_2d'],'ophandface','cocoophf')
        return tracks
