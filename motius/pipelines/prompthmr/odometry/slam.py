"""Native masked SLAM orchestration using the shared DROID neural bundle."""
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from .droid import Droid
from .processing import est_calib,image_stream,preprocess_masks


def run_slam(bundle,images,masks=None,calib=None,depths=None,opt_intr=False,stride=1,depth_keyframes=None,options=None):
    defaults=json.loads((Path(__file__).resolve().parents[3]/'configuration/prompthmr/droid.json').read_text())
    if options:defaults.update(options)
    args=SimpleNamespace(**defaults);args.opt_intr=opt_intr
    droid=None
    if calib is None:calib=est_calib(images)
    if masks is not None:
        masks=masks[::stride]
        image_masks,confidence_masks=preprocess_masks(images,masks)
    for timestamp,image,intrinsic,depth,size_factor in image_stream(images,calib,depths=depths,stride=stride,depth_keyframes=depth_keyframes):
        if droid is None:
            args.image_size=[image.shape[2],image.shape[3]]
            droid=Droid(bundle,args)
        if masks is not None:
            image=image*(image_masks[timestamp]<.5)
            droid.track(timestamp,image,intrinsics=intrinsic,depth=depth,mask=confidence_masks[timestamp])
        else:droid.track(timestamp,image,intrinsics=intrinsic,depth=depth,mask=None)
    if droid is None:raise ValueError('SLAM has no input frames')
    trajectory,estimated=droid.terminate(image_stream(images,calib))
    intrinsics=estimated.copy();intrinsics[0:4:2]/=size_factor[0];intrinsics[1:4:2]/=size_factor[1]
    count=droid.video.counter.value
    timestamps=droid.video.tstamp.cpu().int().numpy()[:count]
    return droid,trajectory,intrinsics,timestamps
