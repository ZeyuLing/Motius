"""Replay original FlowHMR sampling on both full-video feature sets."""
import argparse
from pathlib import Path
import sys
import json
import time
import numpy as np
import torch
import yaml
from safetensors.torch import load_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('upstream','checkpoint','body-model','j-regressor','case-root'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();sys.path.insert(0,str(a.upstream.resolve()))
    from flowhmr.pipeline.pipeline_v2m import V2MPipeline
    from flowhmr.datasets.v2m_generation.base_dataset import compute_camera_features
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    cfg=yaml.safe_load((a.checkpoint/'config.yml').read_text());args=cfg['train_pipeline_args'].copy()
    args.update(smpl_model_path=str(a.body_model),j_regressor_path=str(a.j_regressor),mean_std=str(a.checkpoint/'motion_stats.json'))
    model=V2MPipeline(network_module=cfg['network_module'],network_module_args=cfg['network_module_args'],**args)
    state=model.state_dict();weights=load_file(str(a.checkpoint/'model.safetensors'))
    if set(weights)-state.keys():raise RuntimeError('Unexpected upstream weights')
    missing=state.keys()-weights.keys()
    if any(not (k=='J_regressor' or k.startswith(('body_model.','mesh_model.','vertex_loss.body_model.'))) for k in missing):
        raise RuntimeError('Missing upstream learned parameters')
    state.update(weights);model.load_state_dict(state,strict=True);del state,weights
    model.to('cuda').eval();results={}
    with torch.no_grad():
        for variant in ('upstream','fast'):
            inputs=np.load(a.case_root/variant/'inputs.npz');data=np.load(a.case_root/variant/'output.npz')
            features=torch.from_numpy(data['feature']).cuda();rt=torch.from_numpy(inputs['camera_RT'])
            relative,_=compute_camera_features(rt);n=len(features)
            torch.cuda.synchronize();started=time.perf_counter()
            result=model.generate({'feature':features[None],'camera_R':relative.reshape(1,n,9).cuda()},[0],n,cfg_scale=1)
            raw=result['trans'].clone()
            first=model.mesh_model(dict(rot6d=result['rot6d'][:,0],shapes=result['shapes'][:,0],trans=result['trans'][:,0]))
            result['trans']=result['trans'].clone()
            result['trans'][...,1]-=first['vertices'][...,1].min(-1).values[:,None]
            joints=model.body_model(dict(rot6d=result['rot6d'].reshape(n,52,6),shapes=result['shapes'].expand(1,n,-1).reshape(n,16),trans=result['trans'].reshape(n,3)))['keypoints3d'].reshape(1,n,52,3)
            torch.cuda.synchronize()
            np.savez(a.case_root/(variant+'-original-sampling.npz'),rot6d=result['rot6d'].cpu().numpy(),trans=result['trans'].cpu().numpy(),raw_trans=raw.cpu().numpy(),joints_world=joints.cpu().numpy())
            results[variant]={'seconds':time.perf_counter()-started,'frames':n}
    (a.case_root/'sampling-replay.json').write_text(json.dumps(results,indent=2)+'\n')
    print('SAMPLING_REPLAY_COMPLETE',flush=True)

if __name__=='__main__':main()
