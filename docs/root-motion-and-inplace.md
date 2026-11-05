# World motion / in-place motion / reference motion

This is a reusable geometry tool, not a model or an Autodesk solver wrapper.
The canonical implementation for the M2M worktree is
`tasks/m2m/motion/root_motion.py`; the standalone MotionAI checkout entry is
`tools/motion/root_motion.py`; the synchronized Motius API is
`motius.motion.root_motion`. Implementations are byte-identical at publication.

## Research: Maya and MotionBuilder

Autodesk HumanIK distinguishes the character's Reference transform from Hips.
Maya describes Reference as an abstract parent carrying overall character
translation/rotation/scale; HipsTranslation is a different optional channel.
MotionBuilder likewise maps a Reference object for whole-character movement;
it also offers LockX/LockY/LockZ in-place controls. Locking Y indiscriminately
would remove useful vertical motion and is not this tool's default.

Official sources, checked 2026-11-05:
- [Maya HumanIK character structure](https://help.autodesk.com/cloudhelp/2023/ENU/Maya-CharacterAnimation/files/GUID-5DEFC6E5-033C-45D5-9A0E-224E7A35131B.htm)
- [MotionBuilder Character Mapping List](https://help.autodesk.com/cloudhelp/2024/ENU/MotionBuilder/files/GUID-6FD84BDA-2936-4142-B547-B2205EA0306E.htm)
- [MotionBuilder FBCharacter LockX/Y/Z](https://help.autodesk.com/cloudhelp/2024/ENU/MotionBuilder-SDK/py_ref/class_f_b_character.html)
- [Maya bakeResults](https://help.autodesk.com/cloudhelp/2024/ENU/Maya-Tech-Docs/CommandsPython/bakeResults.html)

These sources establish rig separation and baking/locking facilities. They do
NOT prescribe a unique hip-to-reference filter. Our .25 s smoothing and heading
proxy are explicitly a versioned dataset choice, not an Autodesk algorithm.

Practical DCC workflow: work on a duplicate; identify authored reference/root
and hips separately; evaluate/bake the world transforms at the chosen FPS;
extract the reference track; express the entire character under its inverse;
bake the resulting in-place clip. Keep the extracted reference on a separate
track so restoring its transform reconstructs the original. In Maya, bakeResults
can bake the evaluated constraints/rig; scene-dependent rigs need simulation
baking. In MotionBuilder, map Reference/Hips correctly and plot the evaluated
rig/skeleton rather than assuming a UI axis lock exports a separate root track.
Native DCC execution, FBX roundtrip, scale/shear and solver parity are not tested
by this NumPy tool. Use evaluated rigid transforms (bake scale first).

## Mathematical contract

Y-up, active column-vector rotation matrices. For every joint j:

    world_position[t,j] = C_R[t] @ inplace_position[t,j] + C_p[t]
    world_rotation[t,j] = C_R[t] @ inplace_rotation[t,j]

Apply the inverse to ALL global transforms, not just pelvis XYZ. This preserves
bone lengths and non-root local joint rotations. The source has no dedicated
authored locomotion root; the proxy uses smoothed hips XZ and unwrapped heading.
Filtering operates on the residual about the endpoint line to preserve constant
travel. The default sigma is .25 seconds. Residual hip sway, bob, height,
pitch/roll and high-frequency yaw remain in the pose. This is an anchored,
non-travelling clip, not a requirement that every pelvis coordinate be zero.

`planar`: extract horizontal travel, retain turning in the pose.
`planar_heading`: extract horizontal travel AND heading; default new benchmark.
Authored `reference={'rotation': [T,3,3], 'translation':[T,3]}` overrides proxy
extraction and supports vertical root travel without guessing its decomposition.

The local forward axis is rig-specific! Vicon29 uses (0,1,0) in root local
coordinates. Generic API/CLI defaults to local +Z; pass --forward-axis y for
Vicon29. Heading projected onto the horizontal plane may be undefined in rolls
or inverted motions. Extraction reports valid fraction, interpolates isolated
invalid frames, rejects fewer than two valid frames. The benchmark requires
>=95% valid frames and audits exclusions.

There is no universal decomposition of vertical terrain travel versus hip bob
from hips alone. The current planar benchmark PRESERVES vertical motion and
must not be advertised as terrain-elevation recovery. An authored ground/root
track can support a later separately identified vertical extraction protocol.

## Commands

From MotionAI M2M worktree:

```bash
python -m tasks.m2m.motion.root_motion extract input.npz inplace.npz \
  --trajectory reference.npz --forward-axis y
python -m tasks.m2m.motion.root_motion restore inplace.npz reconstructed.npz \
  --trajectory reference.npz
```

From MotionAI main checkout replace module invocation with
`python tools/motion/root_motion.py`; from Motius use `python tools/root_motion.py`.
Generic NPZ requires positions[T,J,3], rotations[T,J,3,3], fps; source skeleton
metadata is copied by whitelist. Vicon row6D NPZ with hq[T,6J+3], offsets,
parents, fps is also accepted. Unknown/GT fields are not copied. Outputs must
not exist. `--reference-input authored.npz` uses supplied rotation/translation
tracks instead of estimation. API functions work with any J.

## New in-place -> world benchmark

Protocol `inplace-to-world-planar-reference-v1` uses full validation+test
holdout membership; maximum 10,000 cases, original (project,asset) diversity
preferred. Input contains only in-place pose, skeleton, FPS and initial world
reference. Time-varying reference, original world joints and scorer height are
in separate target files and must not be mounted in model input staging.

Predict the planar reference track, compose the world motion, then score
reference ADE/FDE, yaw rotation, velocity/acceleration and world MPJPE. The known
first frame is excluded from averages and reported separately. The input is
not the old benchmark's zero-filled XYZ with original global orientations.

Strata: straight locomotion, turning locomotion, turn-in-place, vertical activity.
The last is based on hips vertical range, NOT a terrain classifier. Reversible
oracle validation and stationary-reference baseline are available. Single-GT
accuracy does not prove unique recoverability; contact/physical baselines and
visual quality review remain separate requirements before model conclusions.

Construction:

```bash
python -m tasks.m2m.eval.root_recovery.build_inplace --workers 24
```

Frozen inputs/targets and receipts are in
`/efs/zeyuling/data/m2m-workbench/benchmarks/inplace-root-holdout-20261105-v1`.
Pilot artifacts use a separate `...-pilot` identity. No old results are renamed.
