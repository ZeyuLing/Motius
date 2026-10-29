"""Native video reconstruction graph composed from neural bundles and geometry."""
from pathlib import Path
import numpy as np
import torch
from omegaconf import OmegaConf
from motius.utils.torch_checkpoint import load_torch_checkpoint
from motius.models.vision.prompthmr.bundle import (
    PromptCameraBundle,PromptDetectorBundle,PromptMaskBundle,PromptPoseBundle,PromptSegmentationBundle,
)
from motius.motion.body_models.prompthmr.image import PromptHMRBodyGeometry
from .calibration import calibrate_video
from .detection import PromptDetector
from .frontend import detect_and_track
from .tracking import PromptMaskTracker
from .keypoints import PromptPosePrediction
from .prepared import PromptHMRPreparedVideo
from .video_io import load_video_frames
from .world import world_hps_estimation
from .optimization import post_optimization


def numpy_results(values):
    for key,value in values.items():
        if isinstance(value,dict):numpy_results(value)
        elif isinstance(value,torch.Tensor):values[key]=value.detach().cpu().numpy()
    return values


class PromptHMRCapture:
    def __init__(self,bundle,body_models_root,*,options=None):
        from smplx import SMPLX
        self.bundle=bundle;self.body_models_root=Path(body_models_root).expanduser().resolve()
        self.config=OmegaConf.load(Path(__file__).resolve().parents[2]/'configuration/prompthmr/pipeline.yaml')
        if options:self.config=OmegaConf.merge(self.config,options)
        root=bundle.artifact_root
        if root is None:raise ValueError('Video capture requires pretrained networks and pipeline assets')
        self.weights=root/'data/pretrain'
        geometry=PromptHMRBodyGeometry(self.body_models_root,mapping_path=root/'data/body_models/smplx2smpl.pkl',
            joint_mapping_path=root/'data/body_models/smplx2smpl_joints.npy',transl_type=bundle.image_config.get('TRANSL','root'))
        geometry_file=root/'prompthmr_geometry.pt'
        if geometry_file.is_file():state=load_torch_checkpoint(geometry_file)
        else:
            state=load_torch_checkpoint(root/'checkpoint.ckpt')['state_dict']
            state={key:value for key,value in state.items() if key.startswith(('smpl.','smplx.','smplx2smpl'))}
        geometry.load_state_dict(state,strict=True)
        self.geometry=geometry
        self.prepared=PromptHMRPreparedVideo(bundle,geometry,self.body_models_root/'smplx/SMPLX_NEUTRAL.npz')
        self.world_body=SMPLX(str(self.body_models_root/'smplx/SMPLX_NEUTRAL.npz'),use_pca=False,flat_hand_mean=True,num_betas=10)
        self.frontends={}

    def frontend(self,name):
        if name not in self.frontends:
            classes=dict(camera=PromptCameraBundle,detector=PromptDetectorBundle,mask=PromptMaskBundle,pose=PromptPoseBundle,segmenter=PromptSegmentationBundle)
            files=dict(camera='camcalib_sa_biased_l2.ckpt',detector='sam2_ckpts/keypoint_rcnn_5ad38f.pkl',mask='sam2_ckpts/sam2_hiera_tiny.pt',pose='vitpose-h-coco_25.pth',segmenter='deeplabv3_resnet50.pth')
            checkpoint=self.weights/files[name]
            if name=='detector' and (self.weights/'detector_native.pt').is_file():checkpoint=self.weights/'detector_native.pt'
            self.frontends[name]=classes[name](checkpoint,device=self.bundle.device)
        return self.frontends[name]

    def infer_video(self,video_path,output_dir,*,static_camera=False,max_frames=None):
        output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
        images,_,fps=load_video_frames(str(video_path),str(output_dir),max_height=896,max_fps=60)
        images=images[:max_frames]
        if not len(images):raise ValueError('Video has no readable frames')
        cfg=OmegaConf.create(OmegaConf.to_container(self.config,resolve=True));cfg.static_cam=static_camera;cfg.seq_folder=str(output_dir)
        results=dict(camera={},people={},timings={},masks=None,has_tracks=False,has_hps_cam=False,has_hps_world=False,
            has_slam=False,has_hands=False,has_2d_kpts=False,has_post_opt=False)
        height,width=images[0].shape[:2]
        results['camera']=dict(img_focal=np.max([height,width]),img_center=np.array([width/2,height/2]))
        results['spec_calib']=calibrate_video(self.frontend('camera'),images,stride=max(1,len(images)//30))
        if cfg.tracker!='sam2':raise NotImplementedError('The native capture graph currently validates the published SAM 2 tracker')
        results['people'],results['masks']=detect_and_track(images,PromptDetector(self.frontend('detector')),
            PromptMaskTracker(self.frontend('mask')),self.frontend('segmenter'),num_max_people=cfg.num_max_people,
            height_thresh=cfg.det_height_thresh,score_thresh=cfg.det_score_thresh,det_thresh=cfg.det_thresh,bbox_interp=cfg.bbox_interp)
        results['has_tracks']=True
        if static_camera:
            if cfg.calib is None:intrinsics=[np.max([height,width]),np.max([height,width]),width/2.,height/2.]
            else:intrinsics=np.loadtxt(cfg.calib)
            results['camera']=dict(pred_cam_R=torch.eye(3)[None].repeat_interleave(len(images),0).numpy(),
                pred_cam_T=torch.zeros(len(images),3).numpy(),img_focal=intrinsics[0],img_center=intrinsics[2:])
        else:
            from .odometry import estimate_camera
            results['camera']=estimate_camera(images,results['masks'],cfg,weights=self.weights,device=self.bundle.device)
        results['has_slam']=True
        PromptPosePrediction(self.frontend('pose')).tracks(images,results['people'])
        results['has_2d_kpts']=True
        results=self.prepared.run(images,results,mask_prompt=True)
        results['contact_joint_ids']=[7,10,8,11,20,21];results['has_hps_cam']=True
        self.world_body.cpu()
        results=world_hps_estimation(cfg,results,self.world_body);results['has_hps_world']=True
        numpy_results(results)
        if cfg.run_post_opt:
            results=post_optimization(cfg,results,images,self.world_body,opt_contact=True)
            results['has_post_opt']=True
        return results,images,fps
