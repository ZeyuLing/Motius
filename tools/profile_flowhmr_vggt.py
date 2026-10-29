"""Full-frame VGGT schedule profiles; all keys/frames and model dtypes retained."""
import argparse,json,time,resource
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from motius.pipelines.flowhmr import camera_estimation as camera
from motius.pipelines.flowhmr.vggt_utils import load_fn


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--variant',choices=['baseline','wide16','wide32','cache_parallel','efficient','cudnn'],required=True)
    a=p.parse_args();root=a.root;dest=root/'vggt-schedules'/a.variant;dest.mkdir(parents=True,exist_ok=True)
    config={'baseline':(8,4096,4096,256),'wide16':(16,16384,16384,1024),'wide32':(32,32768,32768,1024),'cache_parallel':(8,4096,4096,256),'efficient':(8,4096,4096,256),'cudnn':(8,4096,4096,256)}[a.variant]
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    times=defaultdict(float);names={};events=[]
    def timed(name,fn):
        def wrap(*args,**kwargs):
            start=time.perf_counter()
            try:return fn(*args,**kwargs)
            finally:times[name]+=time.perf_counter()-start
        return wrap
    camera._save_video_frames=timed('video_decode_jpeg_write',camera._save_video_frames)
    original_images=load_fn.load_and_preprocess_images
    parallel=a.variant in ('cache_parallel','efficient','cudnn')
    def images(*args,**kwargs):
        kwargs['workers']=4 if parallel else 0
        return original_images(*args,**kwargs)
    load_fn.load_and_preprocess_images=timed('image_read_resize_normalize',images)
    original_load=camera._load_vggt_model
    def load(*args,**kwargs):
        model=timed('model_cpu_load',original_load)(*args,**kwargs)
        names.update({id(module):name for name,module in model.named_modules()})
        original=model.aggregator.forward_camera_streamed
        def streamed(images,**kwargs):
            kwargs.update(frame_batch_size=config[0],token_chunk_size=config[1],query_chunk_size=config[2])
            return original(images,**kwargs)
        model.aggregator.forward_camera_streamed=streamed
        head=model.camera_head.forward
        def head_forward(*args,**kwargs):
            kwargs['query_chunk_size']=config[3]
            return head(*args,**kwargs)
        model.camera_head.forward=head_forward
        return model
    camera._load_vggt_model=load
    class Staged:
        def __init__(self,device):self.device=device
        def __call__(self,module,fn):
            name=names[id(module)];group='.'.join(name.split('.')[:2])
            start=time.perf_counter()
            try:
                module.to(self.device);torch.cuda.synchronize();times[group+'.upload']+=time.perf_counter()-start
                start=time.perf_counter()
                if group == 'aggregator.inter_frame_blocks' and a.variant in ('efficient','cudnn'):
                    from torch.nn.attention import sdpa_kernel,SDPBackend
                    backend=SDPBackend.EFFICIENT_ATTENTION if a.variant=='efficient' else SDPBackend.CUDNN_ATTENTION
                    with sdpa_kernel(backend):result=fn()
                else:result=fn()
                torch.cuda.synchronize();times[group+'.compute']+=time.perf_counter()-start
                return result
            finally:
                start=time.perf_counter();module.to('cpu')
                if parallel:
                    before=torch.cuda.memory_allocated();torch.clear_autocast_cache()
                    times['amp_cache_freed_bytes']+=before-torch.cuda.memory_allocated()
                torch.cuda.empty_cache();times[group+'.offload']+=time.perf_counter()-start
    camera._StagedCameraModules=Staged
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    result=camera.extract_vggt_camera(str(root/'fast-sam-full/upstream/video/input_30fps.mp4'),
        expected_len=750,frame_range=(0,749),cache_model=False,memory_mode='streamed',device='cuda',
        checkpoint_path=json.loads((root/'asset_paths.json').read_text())['preprocessor/vggt_omega_1b_512.pt'],
        progress=lambda phase,current=0,total=0:events.append({'phase':phase,'current':current,'total':total,'seconds':time.perf_counter()-start}))
    torch.cuda.synchronize()
    report=dict(variant=a.variant,seconds=time.perf_counter()-start,config=dict(zip(['frame_batch','ffn_chunk','query_chunk','head_query_chunk'],config)),
        timings=dict(times),frames=len(result.K),peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
        host_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,torch=torch.__version__,gpu=torch.cuda.get_device_name(),budget_gib=6)
    np.savez(dest/'camera.npz',camera_RT=result.camera_RT,camera_K=result.K)
    (dest/'report.json').write_text(json.dumps(report,indent=2));(dest/'progress.json').write_text(json.dumps(events))
    print('VGGT_PROFILE',json.dumps(report),flush=True)

if __name__=='__main__':main()
