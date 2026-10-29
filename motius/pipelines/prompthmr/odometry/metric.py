"""Metric camera reconstruction: neural depth, masked SLAM and robust scaling."""
import numpy as np
import torch
from motius.motion.representation.prompthmr.rotation import quaternion_to_matrix
from .processing import est_calib,get_dimention
from .depth_utils import prep_metric3d,prep_metric3d_img,post_metric3d_batch
from .est_scale import est_scale_hybrid
from .slam import run_slam


def predict_depth(bundle,images,calib,resize_shape=None):
    _,intrinsic,padding,origin=prep_metric3d(images[0],calib,'metric3d_vit_large')
    result=[]
    for start in range(0,len(images),8):
        batch=torch.stack([prep_metric3d_img(image) for image in images[start:start+8]]).to(bundle.device).half()
        with torch.inference_mode():depth,confidence,_=bundle({'input':batch})
        depth=post_metric3d_batch(depth,confidence,padding,origin,intrinsic,resize_shape=resize_shape)
        if batch.shape[0]==1:depth=depth.unsqueeze(0)
        result.extend(depth.cpu().numpy())
    return result


def run_metric_slam(depth_bundle,droid_bundle,images,masks=None,calib=None,*,use_depth_inp=False,stride=1,opt_intr=False,keyframes=None,depth_stride=1):
    if keyframes is None:keyframes=np.arange(0,images.shape[0],depth_stride)
    else:keyframes=np.unique(np.concatenate([keyframes,np.arange(0,images.shape[0],10)]))
    depth_input=predict_depth(depth_bundle,images[keyframes],calib) if use_depth_inp else None
    droid,trajectory,estimated,_=run_slam(droid_bundle,images,masks=masks,calib=calib,depths=depth_input,
        opt_intr=opt_intr,depth_keyframes=keyframes,stride=stride)
    if opt_intr:calib=estimated
    count=droid.video.counter.value
    timestamps=droid.video.tstamp.cpu().int().numpy()[:count]
    disparities=droid.video.disps_up.cpu().numpy()[:count]
    del droid
    torch.cuda.empty_cache()
    if use_depth_inp:scale=1.0
    else:
        depths=predict_depth(depth_bundle,[images[t] for t in timestamps],calib,resize_shape=get_dimention(images[0]))
        scales=[]
        for timestamp,disparity,depth in zip(timestamps,disparities,depths):
            mask=None if masks is None else masks[timestamp].numpy().astype(float)
            scales.append(est_scale_hybrid(1/disparity,depth,msk=mask))
        scales=np.array(scales);scale=np.median(scales[~np.isnan(scales)])
    translation=torch.tensor(trajectory[:,:3])*scale
    quaternion=torch.tensor(trajectory[:,3:])
    rotation=quaternion_to_matrix(quaternion[:,[3,0,1,2]])
    return rotation,translation,calib


def estimate_camera(images,masks,cfg,*,weights,device='cuda'):
    from motius.models.vision.prompthmr.droid.bundle import PromptDroidBundle
    from motius.models.vision.prompthmr.metric3d.bundle import PromptMetricDepthBundle
    if cfg.depth_method!='metric3d_vit_large':
        raise ValueError('The native released PromptHMR graph requires Metric3D ViT-Large')
    masks=torch.as_tensor(masks)
    optimize_intrinsics=not cfg.use_depth
    keyframes=None
    intrinsics=est_calib(images)
    if cfg.calib is not None:
        intrinsics=np.loadtxt(cfg.calib);optimize_intrinsics=False
    elif cfg.focal is not None:
        intrinsics[0]=intrinsics[1]=cfg.focal;optimize_intrinsics=False
    elif cfg.use_depth:
        if cfg.calib_method!='ba':raise ValueError('Depth-input calibration requires the published BA calibration path')
        droid=PromptDroidBundle(weights/'droidcalib.pth',device=device)
        _,_,intrinsics,keyframes=run_slam(droid,images,masks=masks,opt_intr=True,stride=cfg.get('calib_stride',1))
    # Original ordering initializes depth before DROID, then releases both after
    # metric camera recovery. All frame selection and scale fitting stay here.
    depth=PromptMetricDepthBundle(weights/'metric_depth_vit_large_800k.pth',device=device)
    droid=PromptDroidBundle(weights/'droidcalib.pth',device=device)
    try:
        rotation,translation,intrinsics=run_metric_slam(depth,droid,images,masks=masks,calib=intrinsics,
            use_depth_inp=cfg.use_depth,stride=cfg.stride,opt_intr=optimize_intrinsics,keyframes=keyframes)
    except ValueError as error:
        if not str(error).startswith('not enough values to unpack'):raise
        rotation=torch.eye(3)[None].repeat_interleave(len(masks),0);translation=torch.zeros(len(masks),3)
    return dict(pred_cam_R=rotation.numpy(),pred_cam_T=translation.numpy(),img_focal=intrinsics[0],img_center=intrinsics[2:])
