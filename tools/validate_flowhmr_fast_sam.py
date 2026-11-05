"""Independent upstream vs native Fast SAM full-video validation worker.

Reference checkouts are explicit external inputs and never part of the package.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

PROCESS_STARTED=time.perf_counter()

import numpy as np
import torch
import yaml


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','checkpoint','body-model','j-regressor','assets','video','output'):
        p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--variant',choices=('upstream','fast'),required=True)
    p.add_argument('--frames',type=int,default=750)
    p.add_argument('--vggt-memory-mode', choices=('reference','streamed'), default='reference')
    p.add_argument('--memory-budget-gib', type=float)
    p.add_argument('--sam-frame-batch-size', type=int, default=1)
    p.add_argument('--vggt-attention-backend', choices=('auto','torch','cudnn'), default='auto')
    p.add_argument('--sam-conditioning-baseline', action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    if a.memory_budget_gib is not None:
        torch.cuda.set_per_process_memory_fraction(a.memory_budget_gib*2**30/torch.cuda.get_device_properties(0).total_memory)
    torch.cuda.init();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    paths=json.loads(a.assets.read_text());report={'variant':a.variant,'frames':a.frames,'stages':[],
        'sam_conditioning_optimized':not a.sam_conditioning_baseline,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'seed':0,'cfg_scale':1,
        'camera_interval':1,'postprocess':False,'frontend_cache':False,'complete':False,
        'sampling_steps':20,'vggt_memory_mode':a.vggt_memory_mode,'sam_frame_batch_size':a.sam_frame_batch_size,'vggt_attention_backend':a.vggt_attention_backend,'memory_budget_gib':a.memory_budget_gib,'cpu_threads':4,'motion_dtype':'float32','motion_tf32':False,
        'video_sha256':sha256(a.video),'checkpoint_sha256':sha256(a.checkpoint/'model.safetensors')}
    sam_times=[]
    @contextmanager
    def stage(name):
        torch.cuda.synchronize();start=time.perf_counter();print('START',name,flush=True)
        yield
        torch.cuda.synchronize()
        event={'stage':name,'seconds':time.perf_counter()-start,
               'cumulative_peak_allocated_bytes':torch.cuda.max_memory_allocated(),
               'allocated_bytes':torch.cuda.memory_allocated()}
        report['stages'].append(event);print(json.dumps(event),flush=True)
    def time_extractor(extractor):
        original=extractor.extract_frame
        def measured(*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            value=original(*args,**kwargs)
            torch.cuda.synchronize();sam_times.append(time.perf_counter()-start)
            return value
        extractor.extract_frame=measured
        if not hasattr(extractor, 'extract_frames'):return extractor
        original_frames=extractor.extract_frames
        def measured_frames(images,*args,**kwargs):
            torch.cuda.synchronize();start=time.perf_counter()
            value=original_frames(images,*args,**kwargs)
            torch.cuda.synchronize();sam_times.extend([(time.perf_counter()-start)/len(images)]*len(images))
            return value
        extractor.extract_frames=measured_frames
        return extractor
    started=time.perf_counter()
    if a.variant=='upstream':
        for directory in (a.upstream,a.upstream/'third_party/YOLOX',a.upstream/'third_party/sam-3d-body',a.upstream/'third_party/vggt-omega'):
            sys.path.insert(0,str(directory.resolve()))
        from safetensors.torch import load_file
        from flowhmr.pipeline.pipeline_v2m import V2MPipeline
        from flowhmr.utils.runtime_video_processing import YOLOXHumanDetector,VideoProcessor
        from flowhmr.utils.runtime_vggt_camera import extract_vggt_camera
        from flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor,_maybe_convert_backbone_to_fp16
        from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
        from flowhmr.datasets.v2m_generation.base_dataset import compute_camera_features
        with stage('motion_loading'):
            config=yaml.safe_load((a.checkpoint/'config.yml').read_text())
            args=config['train_pipeline_args'].copy()
            args.update(smpl_model_path=str(a.body_model),j_regressor_path=str(a.j_regressor),mean_std=str(a.checkpoint/'motion_stats.json'))
            model=V2MPipeline(network_module=config['network_module'],network_module_args=config['network_module_args'],**args)
            native=load_file(str(a.checkpoint/'model.safetensors'))
            state=model.state_dict()
            if set(native)-state.keys():raise RuntimeError('Checkpoint contains unexpected upstream keys')
            missing=state.keys()-native.keys()
            if any(not (k=='J_regressor' or k.startswith(('body_model.','mesh_model.','vertex_loss.body_model.'))) for k in missing):
                raise RuntimeError('Checkpoint is missing upstream learned parameters/statistics')
            state.update(native);model.load_state_dict(state,strict=True);del state,native
            model=model.to('cuda').eval()
        with stage('transcode'):
            used=VideoProcessor.transcode(str(a.video),str(a.output/'video'),duration=a.frames/30,force_fps=30)
        with stage('tracking'):
            detector=YOLOXHumanDetector(paths['preprocessor/yolox_l.pth'],torch.device('cuda'))
            boxes=VideoProcessor(detector).compute_bboxes(used)[:a.frames]
        with stage('camera'):
            camera=extract_vggt_camera(used,frame_range=(0,len(boxes)-1),expected_len=len(boxes),
                checkpoint_path=paths['preprocessor/vggt_omega_1b_512.pt'],device='cuda',frame_interval=1)
            rt=torch.as_tensor(camera.camera_RT);intrinsics=torch.as_tensor(camera.K)
        np.savez(a.output/'inputs.npz',bbox=boxes.cpu().numpy(),camera_RT=rt.cpu().numpy(),camera_K=intrinsics.cpu().numpy())
        with stage('sam_loading'):
            sam,cfg=load_sam_3d_body(checkpoint_path=paths['preprocessor/sam/model.ckpt'],
                mhr_path=paths['preprocessor/sam/assets/mhr_model.pt'],device='cuda')
            _maybe_convert_backbone_to_fp16(sam,torch.device('cuda'))
            extractor=time_extractor(Sam3DTokenExtractor(SAM3DBodyEstimator(sam,cfg)))
        with stage('sam_extract'):
            features=extractor.extract_video_tokens(used,bbox_xyxy=boxes,K_all=intrinsics,token_dim=3072,max_frames=a.frames)
        with stage('sampling_and_decode'),torch.no_grad():
            relative,_=compute_camera_features(rt)
            result=model.generate({'feature':features[None].cuda(),'camera_R':relative.reshape(1,len(features),9).cuda()},[0],len(features),cfg_scale=1)
            raw_trans=result['trans'].clone()
            first=model.mesh_model(dict(rot6d=result['rot6d'][:,0],shapes=result['shapes'][:,0],trans=result['trans'][:,0]))
            floor=first['vertices'][...,1].min(-1).values
            result['trans']=result['trans'].clone();result['trans'][...,1]-=floor[:,None]
            n=len(features)
            joints=model.body_model(dict(rot6d=result['rot6d'].reshape(n,52,6),
                shapes=result['shapes'].expand(1,n,-1).reshape(n,16),trans=result['trans'].reshape(n,3)))['keypoints3d'].reshape(1,n,52,3)
            result.update(joints_world=joints,raw_trans=raw_trans)
    else:
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        from motius.pipelines.flowhmr import FlowHMRPipeline
        from motius.pipelines.flowhmr.preprocess import FlowHMRVideoPreprocessor
        with stage('motion_loading'):
            pipeline=FlowHMRPipeline.from_pretrained(str(a.checkpoint),bundle_kwargs=dict(body_model_path=str(a.body_model),device='cuda'),
                lightweight=True,profile=True,vggt_memory_mode=a.vggt_memory_mode,sam_frame_batch_size=a.sam_frame_batch_size,vggt_attention_backend=a.vggt_attention_backend,vggt_checkpoint=paths['preprocessor/vggt_omega_1b_512.pt'])
        class Preprocessor(FlowHMRVideoPreprocessor):
            def extract_features(self,video,boxes,K,**kwargs):
                with stage('sam_loading'):
                    extractor=time_extractor(self.build_sam_extractor())
                with stage('sam_extract'):
                    value=extractor.extract_video_tokens(video,bbox_xyxy=boxes,K_all=K,
                        frame_batch_size=self.sam_frame_batch_size,
                        progress_cb=lambda stage,current,total,message: self.progress_callback('Extract body and hand features',current,total),**kwargs)
                saved.update(features=value,boxes=boxes,K=K)
                return value
        pre=Preprocessor(device='cuda',sam_token_only=True,sam_fast=True,
            sam_optimize_conditioning=not a.sam_conditioning_baseline,sam_frame_batch_size=a.sam_frame_batch_size,
            sam3d_ckpt=paths['preprocessor/sam/model.ckpt'],sam3d_config=paths['preprocessor/sam/model_config.yaml'],
            sam3d_mhr=paths['preprocessor/sam/assets/mhr_model.pt'],yolox_ckpt=paths['preprocessor/yolox_l.pth'])
        saved={}
        progress_events=[]
        pipeline.progress_callback=lambda phase,current=0,total=0: progress_events.append({'phase':phase,'current':current,'total':total,'seconds':time.perf_counter()-started})
        original=pipeline.infer_from_feature
        def capture(feature,camera_RT,**kwargs):
            saved['rt']=torch.as_tensor(camera_RT)
            return original(feature,camera_RT,**kwargs)
        pipeline.infer_from_feature=capture
        result=pipeline.infer_v2m(str(a.video),work_dir=str(a.output),max_frames=a.frames,preprocessor=pre,vggt_frame_interval=1,seeds=[0],postprocess=False)
        features,boxes,intrinsics,rt=saved['features'],saved['boxes'],saved['K'],saved['rt']
        np.savez(a.output/'inputs.npz',bbox=torch.as_tensor(boxes).cpu().numpy(),camera_RT=rt.cpu().numpy(),camera_K=torch.as_tensor(intrinsics).cpu().numpy())
        report['pipeline_stages']=pipeline.stage_timings
        report['inference_settings']=result['inference_settings']
        (a.output/'progress.json').write_text(json.dumps(progress_events,indent=2))
        report['sam_frame_timing_basis']='batch duration divided by frame count' if a.sam_frame_batch_size>1 else 'individual frame'
        used=result['video_path']
    torch.cuda.synchronize()
    report.update(total_seconds=time.perf_counter()-started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        sam_frame_seconds=sam_times,transcoded_video_sha256=sha256(used),
        process_seconds=time.perf_counter()-PROCESS_STARTED,
        complete=True,output_frames=len(features),hand_features_per_frame=True)
    np.savez(a.output/'output.npz',feature=features.cpu().numpy(),
        **{key:result[key].detach().cpu().numpy() for key in ('rot6d','trans','shapes','joints_world','raw_trans')})
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VALIDATION_COMPLETE',a.variant,report['total_seconds'],flush=True)

if __name__=='__main__':main()
