# Representation-To-FBX Export

Motius exports every public motion representation onto a rigged
Mixamo-compatible FBX through one API:

```text
model-native tensor -> SMPL-22 animation -> target character armature -> FBX
```

The target mesh, materials, hierarchy, and authored skin weights are
preserved. Blender is used as the headless FBX import, animation, and export
backend; model pipelines do not import Blender.

![HumanML3D-263 source and three character FBX outputs](../../assets/motion/mixamo_fbx_demo/hml263_to_mixamo.gif)

[Open the 1600x450 MP4 source](../../assets/motion/mixamo_fbx_demo/hml263_to_mixamo.mp4)

## Setup

1. Install [Blender](https://www.blender.org/download/) 3.6 or newer.
2. Install a licensed SMPL-family body model using the
   [body-model setup](../../README.md#smpl-body-model-setup).
3. Pass the Blender executable or configure it once:

```bash
export MOTIUS_BLENDER=/opt/blender/blender
```

The built-in `atlas`, `nova`, and `gear` characters are original procedural
CC0 assets packaged with Motius. They use `mixamorig:*` body-bone names but do
not contain Adobe Mixamo meshes, textures, or animation data.

For a character downloaded separately through an Adobe account, keep the file
outside the Python package:

```text
checkpoints/characters/mixamo/
└── x_bot/
    ├── character.fbx
    └── bone_map.json        # optional for non-standard names
```

The [`checkpoints/`](../../checkpoints/README.md) directory scaffold and setup
documentation are tracked, while downloaded character files remain ignored.
Motius does not redistribute Adobe-provided Mixamo assets; review the official
[Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)
and [additional terms](https://wwwimages2.adobe.com/content/dam/cc/en/legal/servicetou/Mixamo-Addl-Terms-en_US-20210623.pdf)
for assets obtained from that service.

## Supported Inputs

| Source | Bridge to SMPL-22 | Requirement | Lossy |
| ------ | ----------------- | ----------- | :---: |
| raw SMPL/SMPL-H parameters | direct axis-angle decode | body model | No |
| `motion135` | direct local-rotation decode | body model | No |
| `hymotion201` | exact 135-channel prefix | body model | No |
| `ms272` | deterministic root and local-rotation decode | body model | No shape |
| `dart276` | DART parameter and coordinate decode | body model | No shape |
| `babel135` | BABEL root integration and row-6D decode | body model | No shape |
| `hml263` | exact joint decode, then position IK | body model | Yes |
| `interhuman262` | exact per-person joints, then position IK | `person_index` | Yes |
| `smpl22_joints` | position IK | body model | Yes |
| `ardy_330` | ARDY-27 decode and named joint bridge, then IK | checkpoint `motion_rep` | Yes |
| `ardy_g1_414` | G1 joint decode and named joint bridge, then IK | checkpoint `motion_rep` | Yes |
| `motionbricks_g1_413/414/418` | native MotionBricks decode and named G1 bridge, then IK | checkpoint `motion_rep` | Yes |
| `g1_38` or `g1_qpos` | MuJoCo FK and named G1 bridge, then IK | `mujoco`, robot XML | Yes |

"No shape" means the source preserves rotations and root motion but not SMPL
shape coefficients. Pass `betas` when a specific output body shape is known.
Position IK is solved on the selected-gender, zero-beta SMPL skeleton; supplied
betas are then applied to the exported skin and are identified separately in
the manifest.
Every IK route records mean, p95, and maximum per-frame MPJPE in
`result.metadata["motion_source"]` and in `<output>.fbx.json`.

## Python API

Use a packaged character slug or an arbitrary FBX path:

```python
from motius.motion import export_motion_to_fbx

result = export_motion_to_fbx(
    motion_hml263,                       # (T, 263)
    source_representation="hml263",
    character_fbx="atlas",              # atlas, nova, gear, or a .fbx path
    output_path="outputs/fbx/atlas_walk.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    model_type="smpl",
    gender="neutral",
    output_fps=30,
    bridge_kwargs={"floor_align": True},
)

print(result.output_path)
print(result.manifest_path)
print(result.metadata["motion_source"])
```

For a separately downloaded character:

```python
result = export_motion_to_fbx(
    motion_hy201,
    "hymotion201",
    "checkpoints/characters/mixamo/x_bot/character.fbx",
    "outputs/fbx/x_bot_motion.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    betas=betas,
    root_motion_scale="auto",
)
```

`root_motion_scale="auto"` scales translation by the target/source skeleton
height ratio. A positive numeric scale can be supplied for an authored rig.

### Checkpoint-Native Representations

ARDY and MotionBricks normalization statistics, skeleton metadata, and feature
decoders are part of the checkpoint bundle. Pass the exact `motion_rep` loaded
with the pipeline:

```python
result = export_motion_to_fbx(
    ardy_features,
    "ardy_330",
    "nova",
    "outputs/fbx/ardy_nova.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    motion_rep=ardy_pipe.bundle.motion_rep,
    is_normalized=True,
)
```

The same pattern applies to `ardy_g1_414` and all three MotionBricks G1
layouts. A dual-root MotionBricks decoder automatically selects its 413D local
or 414D global subset when needed.

### Two-Person Motion

An InterHuman sample contains two synchronized people. Export each track to a
separate FBX while keeping the shared world frame:

```python
for person_index, character in enumerate(("atlas", "nova")):
    export_motion_to_fbx(
        motion_interhuman,               # (T, 2, 262)
        "interhuman262",
        character,
        f"outputs/fbx/person_{person_index}.fbx",
        model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
        person_index=person_index,
    )
```

## Command Line

Array-only representations can be exported without constructing a pipeline:

```bash
python tools/export_motion_fbx.py motion.npy outputs/fbx/atlas_walk.fbx \
  --source hml263 \
  --character atlas \
  --model-path checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --output-fps 30 \
  --floor-align
```

For a user-provided character:

```bash
python tools/export_motion_fbx.py motion.npy outputs/fbx/x_bot_walk.fbx \
  --source hymotion201 \
  --character checkpoints/characters/mixamo/x_bot/character.fbx \
  --model-path checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --bone-map checkpoints/characters/mixamo/x_bot/bone_map.json
```

Checkpoint-native ARDY and MotionBricks tensors use the Python API so their
features cannot be separated accidentally from the matching `motion_rep` and
normalization statistics. The lower-level `tools/export_smpl_fbx.py` remains
available for raw SMPL archives and direct skinned-SMPL export.

## Target Rig Contract

The target FBX must contain exactly one armature and at least one mesh with an
Armature modifier. It must already be rigged and skinned; Motius does not
auto-rig a static mesh.

Common Mixamo and canonical SMPL names, including namespace prefixes such as
`mixamorig:`, are resolved automatically. A custom map is an object from the
22 canonical names (`Pelvis`, `L_Hip`, ..., `R_Wrist`) to target bone names.
Strict mapping is enabled by default: every body bone must resolve, `Pelvis`
must exist for root motion, and no two source bones may share one target.

Motius transfers global rotation deltas in the target rest basis and applies a
posed arm-chain direction correction for A-pose/T-pose compatibility. The
manifest records arm-chain direction-error diagnostics after animation baking.

## Coordinates And Manifest

Canonical input is SMPL Y-up and +Z-forward. The Blender scene is Z-up and
-Y-forward; FBX is written with Y-up and -Z-forward axes. These are basis
changes, so body heading is preserved.

Each export writes `<output>.fbx.json` with:

- source representation, route, FPS, and whether the bridge is lossy;
- position-IK fit MPJPE when applicable;
- source body-model path, type, gender, and target-character path;
- resolved 22-bone map and root-motion scale;
- post-bake arm-chain direction diagnostics;
- input, Blender-scene, and FBX coordinate conventions.

FBX armatures do not reproduce SMPL pose-dependent corrective blend shapes by
default. Direct SMPL export uses the shaped rest mesh and official LBS weights;
character retargeting uses the target's authored deformation. Write generated
artifacts under `outputs/`.
