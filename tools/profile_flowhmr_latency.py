"""Cold-process end-to-end FlowHMR profiling with an explicit prior-path control."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
PROCESS_STARTED=time.perf_counter()
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--body-model',required=True)
    p.add_argument('--video',required=True);p.add_argument('--assets',required=True)
    p.add_argument('--output',required=True);p.add_argument('--frames',type=int,default=30)
    p.add_argument('--variant',choices=['standard-prior','standard-optimized','lightweight-prior','lightweight-optimized'],required=True)
    a=p.parse_args();root=Path(a.output);root.mkdir(parents=True,exist_ok=True)
    if not torch.cuda.is_available():raise RuntimeError('A dedicated idle CUDA GPU is required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.init();torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    prior=a.variant.endswith('prior')
    from motius.pipelines.flowhmr.sam.sam_3d_body_estimator import SAM3DBodyEstimator
    from motius.models.vision.vggt_omega.models.vggt_omega import VGGTOmega
    if prior:
        from motius.models.vision.sam3d_body.bundle import SAM3DBodyBundle
        original_sam_init = SAM3DBodyBundle.__init__
        def eager_sam_init(self, *args, **kwargs):
            return original_sam_init(self, *args, low_cpu_mem_usage=False, **kwargs)
        SAM3DBodyBundle.__init__ = eager_sam_init
        from motius.models.vision.vggt_omega.bundle import VGGTOmegaBundle
        original_vggt_init = VGGTOmegaBundle.__init__
        def eager_vggt_init(self, checkpoint, **kwargs):
            return original_vggt_init(self, checkpoint, low_cpu_mem_usage=False, **kwargs)
        VGGTOmegaBundle.__init__ = eager_vggt_init
        original=SAM3DBodyEstimator.process_one_image
        def flush_each_frame(self,*args,**kwargs):
            torch.cuda.empty_cache()
            kwargs.pop("token_only",None)
            return original(self,*args,**kwargs)
        SAM3DBodyEstimator.process_one_image=flush_each_frame
        from motius.models.vision.vggt_omega.models.vggt_omega import VGGTOmega
        original_camera=VGGTOmega.forward
        def full_camera(self,images,**kwargs):return original_camera(self,images,camera_only=False)
        VGGTOmega.forward=full_camera
    from motius.pipelines.flowhmr import FlowHMRPipeline
    paths=json.loads(Path(a.assets).read_text())
    print("Construct pipeline",flush=True)
    started=time.perf_counter()
    pipe=FlowHMRPipeline.from_pretrained(a.checkpoint,
        bundle_kwargs=dict(body_model_path=a.body_model,device='cuda',low_cpu_mem_usage=not prior),
        lightweight=a.variant.startswith('lightweight-'),profile=True,
        vggt_checkpoint=paths['preprocessor/vggt_omega_1b_512.pt'],
        preprocessor_kwargs=dict(sam_fast=a.variant.startswith('lightweight-') and not prior, sam3d_ckpt=paths['preprocessor/sam/model.ckpt'],
          sam3d_config=paths['preprocessor/sam/model_config.yaml'],
          sam3d_mhr=paths['preprocessor/sam/assets/mhr_model.pt'],
          yolox_ckpt=paths['preprocessor/yolox_l.pth']))
    torch.cuda.synchronize();load=time.perf_counter()-started
    original_infer=pipe.infer_from_feature
    def capture_conditioning(feature,camera_RT,**kwargs):
        import numpy as np
        np.savez(root/'conditioning.npz', feature=feature.detach().cpu().numpy(),
                 camera_RT=torch.as_tensor(camera_RT).cpu().numpy())
        return original_infer(feature,camera_RT,**kwargs)
    pipe.infer_from_feature=capture_conditioning
    result=pipe.infer_monocular_motion_capture(a.video,work_dir=str(root),max_frames=a.frames,
        seeds=[0],postprocess=False,vggt_frame_interval=1)
    torch.cuda.synchronize();total=time.perf_counter()-started
    import numpy as np
    track,=result.tracks
    np.savez(root/'parity.npz',rot6d=track.native_parameters['rot6d'],joints=track.joints_world)
    report=dict(variant=a.variant,frames=len(track.frame_ids),model_loading_seconds=load,
        total_seconds=total,process_seconds=time.perf_counter()-PROCESS_STARTED,stages=pipe.stage_timings,gpu=torch.cuda.get_device_name(),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        checkpoint_sha256=pipe.bundle.checkpoint_sha256,seed=0,camera_interval=1,
        frontends_reused=False,postprocess=False)
    (root/'profile.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
