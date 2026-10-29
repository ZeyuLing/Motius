"""Compare independent-frame Fast SAM batches with identical frame/box/K inputs."""
import argparse,json,time
from pathlib import Path
import cv2,numpy as np,torch
from motius.pipelines.flowhmr.preprocess import FlowHMRVideoPreprocessor


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--frames',type=int,default=64)
    p.add_argument('--batch-sizes',type=int,nargs='+',default=[1,2,4])
    a=p.parse_args();root=a.root;out=root/'sam-batches';out.mkdir(exist_ok=True)
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(6*2**30/torch.cuda.get_device_properties(0).total_memory)
    paths=json.loads((root/'asset_paths.json').read_text())
    pre=FlowHMRVideoPreprocessor(device='cuda',sam_token_only=True,sam_fast=True,
        sam3d_ckpt=paths['preprocessor/sam/model.ckpt'],sam3d_config=paths['preprocessor/sam/model_config.yaml'],
        sam3d_mhr=paths['preprocessor/sam/assets/mhr_model.pt'])
    extractor=pre.build_sam_extractor()
    inputs=np.load(root/'memory-full/fast/inputs.npz');boxes=inputs['bbox'][:a.frames];cameras=inputs['camera_K'][:a.frames]
    cap=cv2.VideoCapture(str(root/'fast-sam-full/upstream/video/input_30fps.mp4'));images=[]
    try:
        for _ in range(a.frames):
            ok,frame=cap.read()
            if not ok:raise ValueError('Video ended early')
            images.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    finally:cap.release()
    reports=json.loads((out/'report.json').read_text()) if (out/'report.json').exists() else []
    baseline=np.load(out/'batch1.npz')['feature'] if 1 not in a.batch_sizes and (out/'batch1.npz').exists() else None
    for size in a.batch_sizes:
        for _ in range(2):
            extractor.extract_frames(images[:size],boxes[:size],cameras[:size]) if size>1 else extractor.extract_frame(images[0],bboxes=boxes[:1],cam_int=cameras[:1])
        torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.perf_counter();tokens=[]
        for index in range(0,len(images),size):
            if size==1:
                value,_=extractor.extract_frame(images[index],bboxes=boxes[index:index+1],cam_int=cameras[index:index+1]);value=value[None]
            else:value=extractor.extract_frames(images[index:index+size],boxes[index:index+size],cameras[index:index+size])
            tokens.append(value)
        torch.cuda.synchronize();seconds=time.perf_counter()-start;features=torch.cat(tokens).numpy()
        if baseline is None:baseline=features
        x=baseline.reshape(-1,3,1024).astype(float);y=features.reshape(-1,3,1024).astype(float)
        cosine=(x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1))
        r=dict(frame_batch_size=size,frames=len(images),seconds=seconds,fps=len(images)/seconds,
               peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),
               token_cosine_min=cosine.min(0).tolist(),token_cosine_mean=cosine.mean(0).tolist())
        np.savez(out/f'batch{size}.npz',feature=features)
        reports.append(r);(out/'report.json').write_text(json.dumps(reports,indent=2));print('BATCH_RESULT',json.dumps(r),flush=True)
    extractor.close()

if __name__=='__main__':main()
