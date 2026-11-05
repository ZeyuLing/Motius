"""Reversible world / in-place motion decomposition (NumPy + SciPy only).

Active column-vector rotations. Y-up. Reference motion is separate from hips.
Default proxy extracts smoothed horizontal travel and heading; vertical hips
motion, pitch/roll and residual sway stay in the in-place clip. An authored
reference track takes precedence. This is not an Autodesk solver emulation.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

PROTOCOL = 'reference-planar-heading-v1'


def _rotations(r):
    r = np.asarray(r, dtype=np.float64)
    if r.shape[-2:] != (3, 3) or not np.isfinite(r).all():
        raise ValueError('Expected finite rotation matrices')
    if not np.allclose(r.swapaxes(-1,-2) @ r, np.eye(3), atol=2e-5) or not np.allclose(np.linalg.det(r), 1, atol=2e-5):
        raise ValueError('Expected proper SO(3) rotations')
    return r


def smooth_track(x, fps, seconds):
    """Smooth residual about endpoint line, preserving constant-speed travel."""
    x = np.asarray(x, dtype=np.float64)
    if seconds < 0 or not np.isfinite(seconds) or fps <= 0 or not np.isfinite(fps):
        raise ValueError('Invalid FPS or smoothing duration')
    if len(x) < 3 or seconds == 0:
        return x.copy()
    line = np.linspace(x[0], x[-1], len(x))
    return line + gaussian_filter1d(x-line, sigma=seconds*fps, axis=0, mode='reflect')


def extract_reference(root_position, root_rotation, *, fps, forward_axis=(0.,0.,1.),
                      mode='planar_heading', smoothing_seconds=.25):
    """Estimate a Y-up reference; report heading validity instead of hiding it.

    forward_axis is skeleton-specific in root LOCAL coordinates, not world up.
    Choose (0,1,0) for the repository Vicon29 convention, (0,0,1) otherwise
    only when it is really the rig's local forward axis.
    """
    p = np.asarray(root_position, dtype=np.float64); r = _rotations(root_rotation)
    if p.ndim != 2 or p.shape[1] != 3 or r.shape != (len(p),3,3) or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('Expected finite [T,3] and [T,3,3], T>=2')
    if mode not in ('planar', 'planar_heading'):
        raise ValueError('Unknown extraction mode')
    axis = np.asarray(forward_axis,dtype=float)
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-8:
        raise ValueError('Invalid local forward axis')
    f = r @ (axis / np.linalg.norm(axis)); valid = np.linalg.norm(f[:,[0,2]],axis=-1) > .1
    if mode == 'planar_heading' and valid.sum()<2:
        raise ValueError('Insufficient upright heading evidence; supply an authored reference')
    yaw = np.zeros(len(p))
    if mode == 'planar_heading':
        # Interpolate only on the extraction side, never deliver removed yaw as input.
        good=np.flatnonzero(valid); angles=np.unwrap(np.arctan2(f[good,0],f[good,2]))
        yaw=smooth_track(np.interp(np.arange(len(p)),good,angles),fps,smoothing_seconds)
    q=Rotation.from_rotvec(yaw[:,None]*np.array([0.,1.,0.])).as_matrix()
    c=smooth_track(p,fps,smoothing_seconds);c[:,1]=0.
    return dict(rotation=q,translation=c,heading_valid=valid,
                metadata=dict(protocol=PROTOCOL,mode=mode,up_axis='Y',forward_axis=axis.tolist(),
                              fps=float(fps),smoothing_seconds=float(smoothing_seconds),
                              heading_valid_fraction=float(valid.mean()),vertical_policy='preserve_in_pose',
                              extraction='smoothed_hips_proxy; not authored locomotion root'))


def world_to_inplace(positions, rotations, *, fps, reference=None, **options):
    """Factor ALL global joint transforms: W_j(t) = C(t) @ L_j(t).

    positions [T,J,3], rotations [T,J,3,3]. Non-root local rotations and bone
    lengths are invariant. Returned reference is ground truth, NOT model input.
    """
    p=np.asarray(positions,dtype=np.float64);r=_rotations(rotations)
    if p.ndim!=3 or p.shape[-1]!=3 or r.shape!=(*p.shape[:2],3,3) or not np.isfinite(p).all():
        raise ValueError('Expected matching finite global positions/rotations')
    ref=extract_reference(p[:,0],r[:,0],fps=fps,**options) if reference is None else reference
    q=_rotations(ref['rotation']);c=np.asarray(ref['translation'],dtype=float)
    if q.shape!=(len(p),3,3) or c.shape!=(len(p),3) or not np.isfinite(c).all():
        raise ValueError('Invalid reference track')
    qi=q.swapaxes(-1,-2)
    return dict(positions=np.einsum('tij,tkj->tki',qi,p-c[:,None]),
                rotations=qi[:,None]@r,reference=ref)


def inplace_to_world(positions, rotations, reference):
    p=np.asarray(positions,dtype=float);r=_rotations(rotations)
    q=_rotations(reference['rotation']);c=np.asarray(reference['translation'],dtype=float)
    if p.ndim!=3 or r.shape!=(*p.shape[:2],3,3) or q.shape!=(len(p),3,3) or c.shape!=(len(p),3) or not np.isfinite(p).all() or not np.isfinite(c).all():
        raise ValueError('Invalid pose/reference shapes or values')
    return dict(positions=np.einsum('tij,tkj->tki',q,p)+c[:,None],rotations=q[:,None]@r)


def unpack_row6d(hq, offsets, parents):
    """Decode [T,6*J+3] global row-6D + root XYZ with FK."""
    x=np.asarray(hq,dtype=float);parents=np.asarray(parents,dtype=int);off=np.asarray(offsets,dtype=float)
    j=len(parents)
    if x.ndim!=2 or x.shape[1]!=6*j+3 or off.shape!=(j,3) or parents[0]!=-1 or any(not 0<=v<i for i,v in enumerate(parents[1:],1)) or not np.isfinite(x).all() or not np.isfinite(off).all():
        raise ValueError('Invalid row6D skeleton/motion')
    a,b=x[:,:6*j].reshape(-1,j,2,3).transpose(2,0,1,3)
    an=np.linalg.norm(a,axis=-1,keepdims=True)
    if (an<1e-8).any():raise ValueError('Degenerate 6D rotation')
    a=a/an;b=b-(a*b).sum(-1,keepdims=True)*a;bn=np.linalg.norm(b,axis=-1,keepdims=True)
    if (bn<1e-8).any():raise ValueError('Degenerate 6D rotation')
    b=b/bn;r=np.stack((a,b,np.cross(a,b)),axis=-2);p=[x[:,-3:]]
    for k in range(1,j):p.append(p[parents[k]]+np.einsum('tij,j->ti',r[:,parents[k]],off[k]))
    return np.stack(p,axis=1),r


def pack_row6d(positions,rotations):
    return np.concatenate((rotations[:,:,:2,:].reshape(len(rotations),-1),positions[:,0]),axis=-1)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['extract','restore']);parser.add_argument('input',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--trajectory',type=Path,required=True,help='Separate reference NPZ; output for extract, input for restore')
    parser.add_argument('--reference-input',type=Path,help='Optional authored rotation/translation reference NPZ')
    parser.add_argument('--fps',type=float);parser.add_argument('--forward-axis',choices=['x','y','z'],default='z')
    parser.add_argument('--mode',choices=['planar','planar_heading'],default='planar_heading');parser.add_argument('--smoothing-seconds',type=float,default=.25)
    args=parser.parse_args()
    outputs=[args.output]+([args.trajectory] if args.action=='extract' else [])
    if len({p.resolve() for p in outputs})!=len(outputs) or any(p.exists() for p in outputs):raise ValueError('Outputs must be distinct new files')
    with np.load(args.input,allow_pickle=False) as z:data=dict(z)
    if 'hq' in data:p,r=unpack_row6d(data['hq'],data['offsets'],data['parents'])
    else:p,r=data['positions'],data['rotations']
    fps=args.fps if args.fps is not None else float(data['fps'])
    if args.action=='extract':
        ref=None
        if args.reference_input:
            with np.load(args.reference_input,allow_pickle=False) as z:ref=dict(z)
        result=world_to_inplace(p,r,fps=fps,reference=ref,forward_axis=np.eye(3)['xyz'.index(args.forward_axis)],mode=args.mode,smoothing_seconds=args.smoothing_seconds)
        ref=result['reference'];ref={k:(json.dumps(v) if isinstance(v,dict) else v) for k,v in ref.items()}
        args.trajectory.parent.mkdir(parents=True,exist_ok=True)
        with args.trajectory.open('xb') as f:np.savez_compressed(f,**ref)
    else:
        with np.load(args.trajectory,allow_pickle=False) as z:ref=dict(z)
        result=inplace_to_world(p,r,ref)
    # Explicit whitelist prevents copying GT/removed trajectories into in-place inputs.
    out={k:data[k] for k in ['offsets','parents','joint_names','height','tpose_local_rot_quat'] if k in data}
    out.update(positions=result['positions'],rotations=result['rotations'],fps=fps)
    if 'hq' in data:out['hq']=pack_row6d(result['positions'],result['rotations'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as f:np.savez_compressed(f,**out)


if __name__=='__main__':main()
