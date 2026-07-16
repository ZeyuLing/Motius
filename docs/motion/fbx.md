# FBX Export And Character Retargeting

Motius can export SMPL-family motion as an animated, skinned FBX and can bake
the same animation onto an existing rigged character FBX. The public API lives
in `motius.motion.fbx`; Blender is used only as the FBX scene and animation
backend.

## Requirements

1. Install [Blender](https://www.blender.org/download/) 3.6 or newer.
2. Download a licensed SMPL-family body model as described in the
   [body-model setup](../../README.md#smpl-body-model-setup).
3. Pass the Blender executable or set it once:

```bash
export MOTIUS_BLENDER=/opt/blender/blender
```

Body-model files and Blender are user-provided dependencies. They are not
redistributed by Motius.

## Export Animated SMPL FBX

`SMPLAnimation` accepts canonical Motius `motion135` or raw SMPL axis-angle
parameters. Shape coefficients must be constant across one clip because one
skinned FBX has one rest mesh.

```python
from motius.motion import SMPLAnimation, export_smpl_fbx

animation = SMPLAnimation.from_motion135(
    motion135,  # (T, 135): root translation + 22 local row-6D rotations
    betas=betas,
    fps=30,
)

result = export_smpl_fbx(
    animation,
    "outputs/fbx/walk_smpl.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    model_type="smpl",
    gender="neutral",
)
print(result.output_path)
print(result.manifest_path)
```

For raw SMPL parameters:

```python
animation = SMPLAnimation.from_smpl(
    global_orient,  # (T, 3), axis-angle
    body_pose,      # (T, 21, 3) or (T, 23, 3), axis-angle
    transl,         # (T, 3)
    betas=betas,
    fps=30,
)
```

The exported scene contains:

- a shaped SMPL mesh;
- an armature following the SMPL kinematic tree;
- official body-model linear-blend skin weights;
- one keyframe per input frame, including root translation;
- a sidecar `<output>.fbx.json` manifest recording the model, axes, FPS, and
  export mode.

## Bake Onto A Character FBX

The target must already contain an armature and a mesh skinned to that
armature. Motius preserves the target mesh, materials, hierarchy, and skin
weights, then replaces its armature animation.

```python
from motius.motion import retarget_smpl_to_fbx

result = retarget_smpl_to_fbx(
    animation,
    character_fbx="checkpoints/characters/hero.fbx",
    output_path="outputs/fbx/hero_walk.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    root_motion_scale="auto",
)
```

Motius automatically recognizes its canonical SMPL bone names and common
Mixamo names, including namespace prefixes such as `mixamorig:`. For another
rig, provide an explicit SMPL-22-to-target map:

```python
bone_map = {
    "Pelvis": "CharacterRoot_Hips",
    "L_Hip": "Character_LeftUpLeg",
    "R_Hip": "Character_RightUpLeg",
    # Continue through L_Wrist and R_Wrist.
}

result = retarget_smpl_to_fbx(
    animation,
    "checkpoints/characters/hero.fbx",
    "outputs/fbx/hero_walk.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    bone_map=bone_map,
    target_armature="HeroRig",  # needed only when the FBX has multiple armatures
)
```

Strict mapping is enabled by default: all 22 body bones must resolve and no two
source bones may target the same bone. Set `strict_bone_map=False` only for a
deliberately reduced rig. `Pelvis` is always required because it carries root
motion. `root_motion_scale="auto"` scales translation by target/source
skeleton height; a positive numeric scale can be supplied for an authored rig.

## Command Line

Export `motion135`:

```bash
python tools/export_smpl_fbx.py motion.npy outputs/fbx/walk_smpl.fbx \
  --input-format motion135 \
  --model-path checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
```

Bake a raw SMPL parameter archive onto a character:

```bash
python tools/export_smpl_fbx.py smpl_params.npz outputs/fbx/hero_walk.fbx \
  --input-format smpl-npz \
  --model-path checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --character-fbx checkpoints/characters/hero.fbx \
  --bone-map checkpoints/characters/hero_smpl22_map.json \
  --root-motion-scale auto
```

The SMPL archive must contain `global_orient`, `body_pose`, `transl` (or
`trans`), and optionally `betas`. Pass `--betas shape.npy` to override the
archive or supply shape for a `motion135` input.

## Coordinates And Limits

The API converts canonical SMPL Y-up, +Z-forward motion into Blender Z-up,
-Y-forward scene coordinates. Blender writes conventional FBX Y-up,
-Z-forward axes. This is a basis conversion, so the animation heading is
preserved.

FBX armatures do not reproduce SMPL pose-dependent corrective blend shapes by
default. The exported SMPL character therefore uses the shaped rest mesh and
official linear-blend skin weights, while a target character uses its authored
skin deformation. The API retargets animation; it does not automatically
create a skeleton or skin weights for an unrigged static mesh.

Write generated files under `outputs/`. Keep licensed body models and character
assets under `checkpoints/`; neither belongs in Git.
