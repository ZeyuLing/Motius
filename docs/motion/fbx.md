# Representation-To-Character FBX Export

Motius exports every public motion representation onto a caller-provided,
rigged character FBX through one API:

```text
model-native tensor -> SMPL-22 animation -> target character skeleton -> FBX
```

The target mesh, materials, hierarchy, and authored skin weights are preserved.
Motius does not bundle or relabel Adobe Mixamo characters. Downloaded Mixamo
characters remain subject to Adobe's terms and stay outside the Python package.

## Backends

| Backend | Character retarget | Direct SMPL mesh export | Runtime |
| ------- | :----------------: | :---------------------: | ------- |
| `fbxsdk` | Yes | No | Autodesk FBX SDK Python wheel, normally CPython 3.10 |
| `blender` | Yes | Yes | Blender 3.6 or newer |
| `auto` | Prefers `fbxsdk`, falls back to Blender | Uses Blender | Resolves installed runtimes |

The Autodesk backend loads the original FBX scene and writes animation curves
directly. It does not reconstruct the character mesh or skin. Blender is not
required for this route. Blender remains useful when creating a new skinned
SMPL FBX from body-model arrays and when rendering previews.

## Setup

1. Install a licensed SMPL-family body model using the
   [body-model setup](../../checkpoints/body_models/README.md).
2. Install Autodesk FBX SDK using the
   [FBX SDK setup](../../checkpoints/fbxsdk/README.md), or install Blender if
   that backend is preferred.
3. Place the rigged target character at a stable local path.

For a character downloaded separately through an Adobe account:

```text
checkpoints/characters/mixamo/
└── x_bot/
    ├── character.fbx
    └── bone_map.json        # optional for non-standard names
```

The [`checkpoints/`](../../checkpoints/README.md) directory scaffold and setup
documentation are tracked, while downloaded character files remain ignored.
Review the official [Mixamo FAQ](https://helpx.adobe.com/creative-cloud/faq/mixamo-faq.html)
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
betas are applied to direct SMPL skin export and identified separately in the
manifest. Every IK route records mean, p95, and maximum per-frame MPJPE in
`result.metadata["motion_source"]` and `<output>.fbx.json`.

## Python API

```python
from motius.motion import export_motion_to_fbx

result = export_motion_to_fbx(
    motion_hml263,
    source_representation="hml263",
    character_fbx="checkpoints/characters/mixamo/x_bot/character.fbx",
    output_path="outputs/fbx/x_bot_walk.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    model_type="smpl",
    gender="neutral",
    output_fps=30,
    backend="fbxsdk",
    bridge_kwargs={"floor_align": True},
)

print(result.output_path)
print(result.manifest_path)
print(result.metadata["backend"])
print(result.metadata["retarget_diagnostics"])
```

`backend="auto"` is the default. It prefers Autodesk FBX SDK for an existing
character and falls back to Blender only when the SDK runtime cannot be
resolved. Set `backend="fbxsdk"` to prohibit that fallback.

The runtime can also be selected per call:

```python
result = export_motion_to_fbx(
    motion_hy201,
    "hymotion201",
    "checkpoints/characters/mixamo/x_bot/character.fbx",
    "outputs/fbx/x_bot_motion.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    backend="fbxsdk",
    fbxsdk_python="/path/to/python3.10",
    fbxsdk_module_path="checkpoints/fbxsdk/cp310",
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
    "checkpoints/characters/mixamo/x_bot/character.fbx",
    "outputs/fbx/ardy_x_bot.fbx",
    model_path="checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl",
    motion_rep=ardy_pipe.bundle.motion_rep,
    is_normalized=True,
)
```

The same pattern applies to `ardy_g1_414` and all three MotionBricks G1
layouts. A dual-root MotionBricks decoder automatically selects its 413D local
or 414D global subset when needed.

### Two-Person Motion

An InterHuman sample contains two synchronized people. Export each track while
keeping the shared world frame:

```python
characters = ("characters/person_a.fbx", "characters/person_b.fbx")
for person_index, character in enumerate(characters):
    export_motion_to_fbx(
        motion_interhuman,
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
python tools/export_motion_fbx.py motion.npy outputs/fbx/x_bot_walk.fbx \
  --source hml263 \
  --character checkpoints/characters/mixamo/x_bot/character.fbx \
  --model-path checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --backend fbxsdk \
  --output-fps 30 \
  --floor-align
```

For a custom rig mapping:

```bash
python tools/export_motion_fbx.py motion.npy outputs/fbx/character.fbx \
  --source hymotion201 \
  --character checkpoints/characters/custom/character.fbx \
  --model-path checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --bone-map checkpoints/characters/custom/bone_map.json
```

Checkpoint-native ARDY and MotionBricks tensors use the Python API so their
features cannot be separated accidentally from the matching `motion_rep` and
normalization statistics. The lower-level `tools/export_smpl_fbx.py` remains
available for raw SMPL archives and direct skinned-SMPL export.

## Target Rig Contract

The target FBX must contain a skeleton and at least one mesh connected to an
FBX skin deformer. It must already be rigged and skinned; Motius does not
auto-rig a static mesh.

Common Mixamo and canonical SMPL names, including namespace prefixes such as
`mixamorig:`, are resolved automatically. A custom map is an object from the
22 canonical names (`Pelvis`, `L_Hip`, ..., `R_Wrist`) to target bone names.
Strict mapping is enabled by default: every body bone must resolve, `Pelvis`
must exist for root motion, and no two source bones may share one target.

The Autodesk backend transfers source global rotations in the target's rest
basis, solves each target bone's local Euler channels using its FBX rotation
order and pre/post rotations, and aligns arm-chain directions across A/T pose
differences. Existing animation stacks are replaced by one
`Motius_Retargeted_Animation` stack. The target's original FBX axis system is
restored before export.

## Coordinates And Manifest

Canonical input is SMPL Y-up and +Z-forward. The Autodesk backend temporarily
converts the target scene to Maya Z-up/right-handed coordinates for retargeting,
then converts the complete animated scene back to the target FBX's original axis
system before saving. The Blender backend uses its Z-up scene and exports
Y-up/-Z-forward FBX. Both routes use explicit basis changes and preserve body
heading.

Each export writes `<output>.fbx.json` with:

- selected backend, source representation, route, FPS, and lossiness;
- position-IK fit MPJPE when applicable;
- source body-model path, type, gender, and target-character path;
- resolved bone map and root-motion scale;
- rest-basis and post-bake arm-chain diagnostics;
- input, working-scene, and output coordinate conventions.

Write generated FBX, videos, and evaluation artifacts under `outputs/`.
