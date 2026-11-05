"""Summarize measured full-video tokens, poses, timing and CUDA memory."""
import argparse
import json
from pathlib import Path
import numpy as np


def rotation_matrices(values):
    v=values.astype(np.float64).reshape(*values.shape[:-1],3,2)
    a=v[...,0];a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-12)
    b=v[...,1];b=b-(a*b).sum(-1,keepdims=True)*a
    b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-12)
    return np.stack((a,b,np.cross(a,b)),axis=-1)


def rotation_errors(a,b):
    ra,rb=rotation_matrices(a),rotation_matrices(b)
    r=ra.swapaxes(-1,-2)@rb
    skew=np.stack((r[...,2,1]-r[...,1,2],r[...,0,2]-r[...,2,0],r[...,1,0]-r[...,0,1]),axis=-1)/2
    return np.rad2deg(np.arctan2(np.linalg.norm(skew,axis=-1),np.clip((np.trace(r,axis1=-2,axis2=-1)-1)/2,-1,1)))


def stats(values):
    values=np.asarray(values)
    return {'mean':float(values.mean()),'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95)),'p99':float(np.percentile(values,99)),'max':float(values.max())}


def motion_comparison(a,b):
    errors=rotation_errors(a['rot6d'],b['rot6d']).reshape(-1,52)
    joint=np.linalg.norm(a['joints_world'].astype(float)-b['joints_world'].astype(float),axis=-1).reshape(-1,52)*1000
    trans=np.linalg.norm(a['trans'].astype(float)-b['trans'].astype(float),axis=-1).reshape(-1)*1000
    raw=np.linalg.norm(a['raw_trans'].astype(float)-b['raw_trans'].astype(float),axis=-1).reshape(-1)*1000
    return {'rotation_degrees':{name:stats(errors[:,s]) for name,s in [('all',slice(None)),('body',slice(0,22)),('left_hand',slice(22,37)),('right_hand',slice(37,52))]},
            'joint_displacement_mm':{'all':stats(joint),'body':stats(joint[:,:22])},
            'root_translation_mm':stats(trans),'raw_root_translation_mm':stats(raw),
            'worst_rotation_frame_joint':list(map(int,np.unravel_index(errors.argmax(),errors.shape))),
            'per_frame_rotation_max_degrees':errors.max(-1).tolist(),
            'per_frame_root_translation_mm':trans.tolist(),
            'per_frame_joint_max_mm':joint.max(-1).tolist()}


def compare(root):
    a=np.load(root/'upstream/output.npz');b=np.load(root/'fast/output.npz')
    ia=np.load(root/'upstream/inputs.npz');ib=np.load(root/'fast/inputs.npz')
    reports={v:json.loads((root/v/'report.json').read_text()) for v in ('upstream','fast')}
    frames=a['feature'].shape[0]
    report={'frames':frames,'complete':all(v['complete'] for v in reports.values()),'reports':reports,
            'observations':{k:{'shape':list(ia[k].shape),'bitwise_equal':bool(np.array_equal(ia[k],ib[k])),
                'max_abs':float(np.abs(ia[k]-ib[k]).max())} for k in ia.files},
            'all_frames_and_hands':b['feature'].shape==(frames,3072) and ia['camera_RT'].shape[0]==frames and ib['camera_RT'].shape[0]==frames,
            'motion':motion_comparison(a,b)}
    x=a['feature'].reshape(frames,3,1024).astype(float);y=b['feature'].reshape(frames,3,1024).astype(float)
    cosine=np.clip((x*y).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)),-1,1)
    report['tokens']={name:{'cosine_min':float(cosine[:,i].min()),'cosine_mean':float(cosine[:,i].mean()),
        'max_abs':float(np.abs(x[:,i]-y[:,i]).max())} for i,name in enumerate(('body','left_hand','right_hand'))}
    report['token_cosine_per_frame']=cosine.tolist()
    report['timing']={v:{'total_seconds':r['total_seconds'],
        'sam_extract_seconds':next(s['seconds'] for s in r['stages'] if s['stage']=='sam_extract'),
        'sam_frame_ms_including_warmup':stats(np.array(r['sam_frame_seconds'])*1000),
        'sam_frame_ms_excluding_first_5':stats(np.array(r['sam_frame_seconds'][5:])*1000),
        'peak_allocated_GiB':r['peak_allocated_bytes']/2**30,'peak_reserved_GiB':r['peak_reserved_bytes']/2**30} for v,r in reports.items()}
    report['speedup']={key:report['timing']['upstream'][key]/report['timing']['fast'][key] for key in ('total_seconds','sam_extract_seconds')}
    for v in ('upstream','fast'):
        replay=root/(v+'-original-sampling.npz')
        if replay.exists():report[v+'_sampling_replay']=motion_comparison(np.load(root/v/'output.npz'),np.load(replay))
    repeat=root/'upstream-repeat/output.npz'
    if repeat.exists():
        report['upstream_full_repeat']=motion_comparison(a,np.load(repeat))
        report['upstream_repeat_report']=json.loads((repeat.parent/'report.json').read_text())
        repeated_inputs=np.load(repeat.parent/'inputs.npz')
        report['repeat_observations']={k:bool(np.array_equal(ia[k],repeated_inputs[k])) for k in ia.files}
    report['finite_outputs']={v:all(bool(np.isfinite(np.load(root/v/'output.npz')[k]).all()) for k in a.files) for v in ('upstream','fast')}
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('case_root',type=Path)
    a=p.parse_args();r=compare(a.case_root)
    (a.case_root/'comparison.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'frames':r['frames'],'tokens':r['tokens'],'rotation_degrees':r['motion']['rotation_degrees'],
                     'root_translation_mm':r['motion']['root_translation_mm'],'joints_mm':r['motion']['joint_displacement_mm'],
                     'observations':r['observations'],'speedup':r['speedup'],'timing':r['timing']},indent=2))

if __name__=='__main__':main()
