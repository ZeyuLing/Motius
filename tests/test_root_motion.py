"""Portable geometry utility tests; no model weights/optional runtimes required."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def test_reference_motion_roundtrip():
    path=Path(__file__).resolve().parents[1]/'motius/motion/root_motion.py'
    spec=importlib.util.spec_from_file_location('root_motion_under_test',path);rm=importlib.util.module_from_spec(spec);spec.loader.exec_module(rm)
    t=np.linspace(0,3,91);r=Rotation.from_euler('y',t*.6).as_matrix()[:,None];p=np.stack((t,1+.1*np.sin(t*6),t*t*.2),-1)[:,None]
    v=rm.world_to_inplace(p,r,fps=30);w=rm.inplace_to_world(v['positions'],v['rotations'],v['reference'])
    np.testing.assert_allclose(w['positions'],p,atol=1e-12);np.testing.assert_allclose(w['rotations'],r,atol=1e-12)
    np.testing.assert_allclose(v['positions'][:,:,1],p[:,:,1],atol=1e-12)
