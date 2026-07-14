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

## InterHuman To SMPL

InterHuman-262 stores paired SMPL-22 joint positions directly, so the exact
decode is:

```python
from motius.motion import convert_motion

joints_pair = convert_motion(motion_interhuman, "interhuman262", "joints")
```

The inverse route to SMPL mesh is not exact. InterHuman omits root rotation,
shape, and complete twist, so Motius model cards use a neutral-SMPL position-IK
bridge for previews and report the fit error in the preview metadata. Use that
mesh bridge for qualitative inspection; use the exact joint decode for
joint-position evaluators.

To encode paired SMPL motion into InterHuman-262:

```python
from motius.motion import motion135_to_interhuman262

motion_interhuman = motion135_to_interhuman262(
    motion135_pair,              # (T, 2, 135)
    bone_offsets=smpl22_offsets, # (22, 3)
    source_coordinates="y_up",
)
```

The pair is canonicalized once using person one's first frame. Do not
canonicalize each person independently.

## ARDY-27 And SMPL-22 Joint Bridges

The official ARDY repository does not provide ARDY-to-SMPL or SMPL-to-ARDY
rotation retargeting code. It visualizes the released ARDY-330 checkpoint with its
native ARDY-27 skin. Motius exposes named joint-position bridges:

```python
from motius.motion import (
    ardy_core27_to_smpl22_joints,
    convert_motion,
    smpl22_joints_to_ardy_core27_joints,
)

smpl22_joints = ardy_core27_to_smpl22_joints(ardy27_joints)
ardy27_joints = smpl22_joints_to_ardy_core27_joints(smpl22_joints)

smpl22_joints = convert_motion(
    ardy_features,
    "ardy_330",
    "smpl22_joints",
    motion_rep=ardy_pipe.bundle.motion_rep,
    is_normalized=True,
)
```

These bridges are for skeleton viewers and joint-position evaluators. They do
not produce a ARDY-330 ARDY feature tensor, recover SMPL shape/twist, or create
a valid `motion135` sequence.

This route maps ARDY-27 joints into SMPL-22 joint order for visualization and
joint-position evaluator experiments. It is not a valid SMPL pose, not a
`motion135` sequence, and not sufficient for SMPL mesh rendering. Mesh or
leaderboard evaluation through SMPL requires a later position-IK bridge with a
reported fitting error.

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
