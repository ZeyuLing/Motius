"""Vendored, minimal GMR (General Motion Retargeting) for SMPL-X -> Unitree G1.

This package is a trimmed, in-tree copy of the SMPL-X -> G1 retargeting path from
the GMR project (https://github.com/YanjieZe/GMR), so that
``motius.motion.retarget`` provides high-quality mink-IK retargeting without an
external repository checkout.

Vendored contents:
  - ``motion_retarget.GeneralMotionRetargeting`` — per-frame mink IK solver.
  - ``smpl.load_smplx_file`` / ``smpl.get_smplx_data_offline_fast`` — SMPL-X FK,
    human-height estimation, and fps alignment.
  - ``params`` — asset / IK-config paths (resolved inside this package).
  - ``ik_configs/smplx_to_g1.json`` and ``assets/unitree_g1/`` (mocap mjcf + meshes).

Upstream extras not needed for this path (LAFAN1/BVH/Xsens/OptiTrack loaders,
viewers, other robots/body models) are intentionally omitted. SMPL-X body models
are user-provided data and resolved separately (see ``GMRSMPLToG1Retargeter``).

Runtime deps (lazy): ``mink``, ``daqp``, ``mujoco``, ``smplx``, ``scipy``.
"""
from .motion_retarget import GeneralMotionRetargeting
from .smpl import load_smplx_file, get_smplx_data_offline_fast

__all__ = [
    "GeneralMotionRetargeting",
    "load_smplx_file",
    "get_smplx_data_offline_fast",
]
