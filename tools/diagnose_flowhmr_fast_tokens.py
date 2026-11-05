"""Investigate worst full-video hand-token cases using the same upstream model."""
import argparse,json,sys
from pathlib import Path
import cv2,numpy as np,torch


def cos(a,b):
    x,y=a.reshape(3,1024).double(),b.reshape(3,1024).double()
    return torch.nn.functional.cosine_similarity(x,y,dim=-1).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','assets','case-root'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    for d in (root,a.upstream,a.upstream/'third_party/sam-3d-body'):sys.path.insert(0,str(d))
    from sam_3d_body import load_sam_3d_body,SAM3DBodyEstimator
    from flowhmr.utils.runtime_sam_features import Sam3DTokenExtractor as OriginalExtractor
    from motius.pipelines.flowhmr.features import Sam3DTokenExtractor
    from motius.pipelines.flowhmr.sam import inference
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    paths=json.loads(a.assets.read_text());proof=json.loads((a.case_root/'comparison.json').read_text())
    c=np.array(proof['token_cosine_per_frame']);indices=sorted(set(np.argsort(c[:,1])[:4].tolist()+np.argsort(c[:,2])[:4].tolist()))
    data=np.load(a.case_root/'upstream/inputs.npz');baseline=np.load(a.case_root/'upstream/output.npz')['feature']
    model,cfg=load_sam_3d_body(checkpoint_path=paths['preprocessor/sam/model.ckpt'],mhr_path=paths['preprocessor/sam/assets/mhr_model.pt'],device='cuda')
    estimator=SAM3DBodyEstimator(model,cfg);original=OriginalExtractor(estimator)
    fast=Sam3DTokenExtractor(estimator,token_only=True,fast=True)
    get_boxes=model._get_hand_box;boxes={}
    def capture(*args,**kwargs):
        left,right=get_boxes(*args,**kwargs);boxes.update(left=left.copy(),right=right.copy());return left,right
    model._get_hand_box=capture
    get_fast_boxes=inference._get_hand_box
    video=a.case_root/'upstream/video/input_30fps.mp4';cap=cv2.VideoCapture(str(video));rows=[]
    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,image=cap.read()
        if not ok:raise RuntimeError('Missing source frame')
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);bbox=data['bbox'][index:index+1];K=torch.from_numpy(data['camera_K'][index:index+1])
        first,_=original.extract_frame(image,bboxes=bbox,cam_int=K)
        saved={k:v.copy() for k,v in boxes.items()}
        second,_=original.extract_frame(image,bboxes=bbox,cam_int=K)
        normal,_=fast.extract_frame(image,bboxes=bbox,cam_int=K)
        inference._get_hand_box=lambda *args:(saved['left'].copy(),saved['right'].copy())
        try:frozen,_=fast.extract_frame(image,bboxes=bbox,cam_int=K)
        finally:inference._get_hand_box=get_fast_boxes
        row={'frame':index,'original_repeat_cosine':cos(first,second),
             'original_new_process_vs_full_run_cosine':cos(torch.from_numpy(baseline[index]),first),
             'fast_same_upstream_model_cosine':cos(first,normal),
             'fast_frozen_hand_boxes_cosine':cos(first,frozen)}
        rows.append(row);print(json.dumps(row),flush=True)
    cap.release();fast.close()
    (a.case_root/'worst-frame-diagnostic.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
