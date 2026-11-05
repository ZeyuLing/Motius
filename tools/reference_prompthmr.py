"""Load pristine PromptHMR numeric stages without its optional exporters.

This reference is deliberately separate from the migrated implementation.
Only import/model-factory locations and the SPEC checkpoint filename are
supplied explicitly. Model classes and stage arithmetic come from the pinned
original files, including the complete original Pipeline method ordering.
"""
import ast,copy,os,sys
from pathlib import Path
from collections import defaultdict
from typing import Any,Dict,List,Tuple,Optional
from verify_prompthmr_frontend_parity import source_module


def definitions(path,names,scope,transform=None):
    tree=ast.parse(path.read_text())
    available={node.name:node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    names=set(names)
    while True:
        dependencies={node.id for name in names for node in ast.walk(available[name]) if isinstance(node,ast.Name) and node.id in available}
        if dependencies<=names:break
        names|=dependencies
    nodes=[node for node in tree.body if getattr(node,'name',None) in names]
    module=ast.Module(body=nodes,type_ignores=[])
    if transform:module=transform.visit(module)
    exec(compile(ast.fix_missing_locations(module),str(path),'exec'),scope)
    return scope


def camera_reference(upstream,artifact):
    """Actual original camera solvers with checkpoint locations made explicit."""
    import torch,numpy as np,cv2
    from PIL import Image
    from glob import glob
    from torchvision.transforms import Resize
    sys.path.insert(0,str(upstream/'pipeline/droidcalib/droid_slam'))
    from droid import Droid
    utilities=source_module('original_prompt_slam_utilities',upstream/'pipeline/camera/slam_utils.py')
    from torchmin import minimize
    scale=dict(np=np,torch=torch,cv2=cv2,minimize=minimize)
    definitions(upstream/'pipeline/camera/est_scale.py',{'est_scale_hybrid'},scale)
    depth=dict(np=np,torch=torch,cv2=cv2)
    definitions(upstream/'pipeline/camera/depth_utils.py',{'prep_metric3d','prep_metric3d_img','post_metric3d','post_metric3d_batch'},depth)
    source=upstream/'pipeline/yvanyin_metric3d_main';sys.path.insert(0,str(source))
    from mmengine import Config
    from mono.model.monodepth_model import get_configured_monodepth_model
    def metric_model(pretrain=True):
        cfg=Config.fromfile(str(source/'mono/configs/HourglassDecoder/vit.raft5.large.py'))
        model=get_configured_monodepth_model(cfg)
        if pretrain:
            model.load_state_dict(torch.load(artifact/'data/pretrain/metric_depth_vit_large_800k.pth',map_location='cpu',weights_only=False)['model_state_dict'],strict=False)
        return model
    image_folder=source_module('original_prompt_camera_image_dataset',upstream/'pipeline/image_folder.py')
    from prompt_hmr.utils.rotation_conversions import quaternion_to_matrix
    scope=dict(depth,os=os,Image=Image,glob=glob,Resize=Resize,Droid=Droid,
        slam_args=utilities.slam_args,ImageFolder=image_folder.ImageFolder,quaternion_to_matrix=quaternion_to_matrix,
        metric3d_vit_large=metric_model,est_scale_hybrid=scale['est_scale_hybrid'])
    scope.update({name:getattr(utilities,name) for name in ['est_calib','get_dimention','image_stream','preprocess_masks']})
    class ExplicitProcessingImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module=='depth_utils':return None
            return node
    definitions(upstream/'pipeline/camera/masked_droid_slam.py',{'run_slam','run_metric_slam'},scope,ExplicitProcessingImports())
    return scope


def load_reference(args):
    import cv2,joblib,torch,numpy as np,scipy.signal as signal
    from scipy.ndimage import gaussian_filter1d
    from scipy.interpolate import make_interp_spline,interp1d
    from PIL import Image,ImageOps
    from omegaconf import OmegaConf
    from torchvision.transforms import Compose,Normalize,ToTensor
    from torch.utils.data import Dataset,DataLoader
    from torch.amp import autocast
    from hydra.utils import instantiate
    upstream=args.upstream;artifact=args.artifact
    sys.path.insert(0,str(upstream));sys.path.insert(0,str(upstream/'pipeline'));sys.path.insert(0,str(upstream/'pipeline/gvhmr'))
    from prompt_hmr import load_model as load_phmr
    from prompt_hmr.utils import rotation_conversions as rotation
    from prompt_hmr.utils.one_euro_filter import smooth_one_euro
    from hmr4d.model.gvhmr.gvhmr_pl_demo import DemoPL
    from hmr4d.utils.geo_transform import compute_cam_angvel
    from hmr4d.utils.geo.hmr_cam import get_bbx_xys_from_xyxy,normalize_kp2d
    from smplx import SMPLX
    common=dict(torch=torch,np=np,cv2=cv2,os=os,copy=copy,OmegaConf=OmegaConf,Any=Any,Dict=Dict,List=List,Tuple=Tuple,Optional=Optional)
    common.update({name:getattr(rotation,name) for name in dir(rotation) if not name.startswith('__')})
    image_folder=source_module('original_prompt_image_dataset',upstream/'pipeline/image_folder.py')
    detection=source_module('original_prompt_detectron_predictor',upstream/'pipeline/utils_detectron2.py')
    sam=source_module('original_prompt_sam_tracker',upstream/'pipeline/detector/sam2_video_predictor.py')
    def build_tracker(config_file,checkpoint,device='cuda',**kwargs):
        cfg=OmegaConf.load(config_file).model
        cfg._target_='original_prompt_sam_tracker.SAM2VideoPredictor'
        cfg.sam_mask_decoder_extra_args=dict(dynamic_multimask_via_stability=True,dynamic_multimask_stability_delta=.05,dynamic_multimask_stability_thresh=.98)
        cfg.binarize_mask_from_pts_for_mem_enc=True;cfg.fill_hole_area=8
        network=instantiate(cfg,_recursive_=True)
        sam._load_checkpoint(network,checkpoint)
        return network.to(device).eval()
    class OriginalImports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            if node.module in {'detector.sam2_video_predictor','utils_detectron2'}:return None
            return node
    tools=dict(common,defaultdict=defaultdict,make_interp_spline=make_interp_spline,interp1d=interp1d,
        ImageFolder=image_folder.ImageFolder,DefaultPredictor=detection.DefaultPredictor,build_sam2_video_predictor=build_tracker)
    names={'detect_segment_track_sam','batched_mask_to_box','nms_oks','interpolate_bboxes','est_camera','est_calib'}
    definitions(upstream/'pipeline/tools.py',names,tools,OriginalImports())
    # Original ViTPose numeric wrapper, with its unused detector/SORT constructor
    # omitted. The real original ViTPose neural class/checkpoint is restored.
    vit=upstream/'pipeline/detector/ViTPose/easy_vitpose';sys.path.insert(0,str(vit))
    from vit_models.model import ViTPose
    from vit_utils.top_down_eval import keypoints_from_heatmaps
    from vit_utils.inference import pad_image,draw_bboxes
    from vit_utils.visualization import draw_points_and_skeleton,joints_dict
    tree=ast.parse((vit/'inference.py').read_text());original=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='VitInference')
    methods=[node for node in original.body if isinstance(node,ast.FunctionDef) and node.name in {'inference_wbbox','pre_img','postprocess','_inference_torch'}]
    wrapper=ast.ClassDef(name='OriginalPosePrediction',bases=[],keywords=[],body=methods,decorator_list=[])
    pose_scope=dict(common,MEAN=[.485,.456,.406],STD=[.229,.224,.225],keypoints_from_heatmaps=keypoints_from_heatmaps,
        pad_image=pad_image,draw_bboxes=draw_bboxes,draw_points_and_skeleton=draw_points_and_skeleton,joints_dict=joints_dict)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper],type_ignores=[])),str(vit/'inference.py'),'exec'),pose_scope)
    def load_pose(model_path=None,**kwargs):
        import runpy
        cfg=runpy.run_path(str(vit/'configs/ViTPose_common.py'))['model_huge'];cfg['keypoint_head']['out_channels']=25
        network=ViTPose(cfg)
        state=torch.load(model_path,map_location='cpu',weights_only=False);network.load_state_dict(state.get('state_dict',state),strict=True)
        prediction=pose_scope['OriginalPosePrediction']();prediction._vit_pose=network.cuda().eval();prediction.device='cuda';prediction.dataset='coco_25';prediction.target_size=[192,256]
        prediction._inference=prediction._inference_torch
        return prediction
    pose_estimator=dict(common,load_vit_model=load_pose)
    definitions(upstream/'pipeline/detector/vitpose_estimator.py',{'estimate_kp2ds_from_bbox_vitpose'},pose_estimator)
    kps=source_module('original_prompt_keypoint_layouts',upstream/'pipeline/kp_utils.py')
    hps=dict(common,load_phmr=load_phmr,DemoPL=DemoPL,Image=Image,ImageOps=ImageOps,Dataset=Dataset,DataLoader=DataLoader,
        Compose=Compose,ToTensor=ToTensor,Normalize=Normalize,signal=signal,gaussian_filter1d=gaussian_filter1d,autocast=autocast,
        compute_cam_angvel=compute_cam_angvel,get_bbx_xys_from_xyxy=get_bbx_xys_from_xyxy,normalize_kp2d=normalize_kp2d)
    definitions(upstream/'pipeline/phmr_vid.py',{'load_video_head','pad_image','PromptHMRVideoDataset','PromptHMR_Video'},hps)
    import spec.cam_calib as camera
    class CheckpointLocation(ast.NodeTransformer):
        def visit_Constant(self,node):
            if node.value=='data/pretrain/camcalib_sa_biased_l2.ckpt':return ast.copy_location(ast.Constant(str(artifact/'data/pretrain/camcalib_sa_biased_l2.ckpt')),node)
            return node
        def visit_If(self,node):
            if 'dockerenv' in ast.unparse(node.test):return None
            return self.generic_visit(node)
    camera_scope=dict(camera.__dict__)
    definitions(upstream/'pipeline/spec/cam_calib.py',{'run_spec_calib'},camera_scope,CheckpointLocation())
    def run_calibration(images,**kwargs):
        kwargs.pop('method',None)
        return camera_scope['run_spec_calib'](images,**kwargs)
    floor=dict(common)
    definitions(upstream/'prompt_hmr/vis/traj.py',{'fit_floor_height'},floor)
    world=dict(common,fit_floor_height=floor['fit_floor_height'],smooth_one_euro=smooth_one_euro)
    definitions(upstream/'pipeline/world.py',{'transform_smpl_params','world_hps_estimation'},world)
    optimization=dict(common,F=torch.nn.functional)
    definitions(upstream/'pipeline/postprocessing.py',{'gmof','post_optimization'},optimization)
    video_io=source_module('original_prompt_video_io',upstream/'pipeline/utils.py')
    # Keep the original class and complete numeric __call__; stop immediately
    # after authoritative results.pkl, before unrelated MCS/GLB export.
    tree=ast.parse((upstream/'pipeline/pipeline.py').read_text());pipeline=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Pipeline')
    method=next(node for node in pipeline.body if isinstance(node,ast.FunctionDef) and node.name=='__call__')
    end=next(index for index,node in enumerate(method.body) if isinstance(node,ast.Expr) and isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='dump')
    method.body=method.body[:end+1]+[ast.Return(ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='results',ctx=ast.Load()))]
    scope=dict(common,joblib=joblib,SMPLX=SMPLX,load_video_frames=video_io.load_video_frames,
        est_camera=tools['est_camera'],est_calib=tools['est_calib'],detect_segment_track_sam=tools['detect_segment_track_sam'],
        load_vit_model=load_pose,estimate_kp2ds_from_bbox_vitpose=pose_estimator['estimate_kp2ds_from_bbox_vitpose'],
        convert_kps=kps.convert_kps,PromptHMR_Video=hps['PromptHMR_Video'],run_cam_calib=run_calibration,
        world_hps_estimation=world['world_hps_estimation'],post_optimization=optimization['post_optimization'])
    if getattr(args,'dynamic_camera',False):
        camera=camera_reference(upstream,artifact)
        scope.update(run_slam=camera['run_slam'],run_metric_slam=camera['run_metric_slam'])
    exec(compile(ast.fix_missing_locations(ast.Module(body=[pipeline],type_ignores=[])),str(upstream/'pipeline/pipeline.py'),'exec'),scope)
    return scope['Pipeline']
