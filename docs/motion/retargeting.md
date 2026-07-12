# Motion Retargeting

Retargeters are optional APIs under `motius.motion.retarget`. Body-model files
are user-provided assets; set `MOTIUS_SMPL_MODEL_DIR` or pass a path directly.

## HumanML3D To SMPL

```python
from motius.motion.retarget import retarget_hml263_clip

result = retarget_hml263_clip(
    motion_hml263,
    model_dir="/models/smpl",
    source_fps=20,
    target_fps=30,
    refine_iters=5,
)
motion135 = result["motion_135"]
fit_mpjpe_mm = result["fit_mpjpe_mm"]
```

This route maps the HML263 local-rotation block onto the SMPL rest skeleton,
then optionally refines it against recovered joints. It is inherently lossy:
HML263 does not uniquely determine body shape or twist. Inspect mesh output as
well as joint MPJPE before using converted data for evaluation.

## SMPL To KIMODO SOMA

Set `KIMODO_SKELETON_ASSETS` to a directory containing `smplx22/joints.p` and
`somaskel30/joints.p` from the [KIMODO repository](https://github.com/nv-tlabs/kimodo).

```python
from motius.motion.retarget import SMPLSOMARetargeter

retargeter = SMPLSOMARetargeter(assets_root="/models/kimodo/skeletons")
soma = retargeter.smpl_to_soma(motion135)
print(soma["soma30_joints"].shape)
```

`KIMODOSOMAToSMPLRetargeter` provides the inverse IK path for SOMA/KIMODO joint
outputs, including optional SOMA77 orientation guides.

## SMPL To Unitree G1

The G1 backend is a reduced in-tree integration of
[General Motion Retargeting](https://github.com/YanjieZe/GMR). Install its
runtime dependencies and provide SMPL-X assets:

The vendored subset retains GMR's MIT license at
`motius/motion/retarget/_gmr/LICENSE`.

```bash
python -m pip install -e ".[retarget]"
python -m pip install mink daqp mujoco
export MOTIUS_SMPL_MODEL_DIR=/models/smpl_models
```

```python
from motius.motion.retarget import GMRSMPLToG1Retargeter

retargeter = GMRSMPLToG1Retargeter(tgt_fps=30)
g1 = retargeter.retarget_from_motion135(motion135, fps=30)
qpos = retargeter.to_mujoco_qpos(g1)  # (T, 36): root 7 + G1 29 DOF
retargeter.save_pkl(g1, "outputs/retarget/walk_g1.pkl")
```

The default result is Z-up, joint-limit-clamped, smoothed, and ground-aligned.
Set `mujoco_zup=False` and `ground_align=False` only when you need GMR's raw
solver frame.
