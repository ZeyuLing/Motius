# Automatic Character Rigging

Motius can fit a canonical SMPL-22 armature to a static humanoid mesh and
generate skin weights without requiring a learned checkpoint:

```text
GLB / GLTF / FBX / OBJ / PLY / STL
  -> Blender import and coordinate normalization
  -> geometry-only SMPL-22 template fit
  -> skin-weight estimation
  -> skinned FBX / GLB / GLTF + JSON manifest
```

The result is a **static rigged character**. Automatic rigging and animation
retargeting are separate stages: this API creates the skeleton and skin, while
the [FBX motion bridge](fbx.md) applies motion to a rigged target.

> [!IMPORTANT]
> The current `template` method is an experimental, deterministic baseline for
> one upright T/A-pose humanoid. A successful export proves that the mesh is
> structurally bound; it does not prove production-quality anatomy, weights,
> bone roll, foot contact, or facial orientation. Inspect the generated rig and
> deformation before using it in a dataset, paper, game, or robot workflow.

## Requirements

- Python 3.10 or newer;
- Motius installed from source;
- Blender 3.6 or newer, either on `PATH`, in `MOTIUS_BLENDER`, or passed with
  `blender_executable` / `--blender`.

Blender performs scene import, armature creation, skin binding, and export. It
is an external runtime and is not installed by the Python package.

## Quick Start

### Command line

```bash
python tools/auto_rig_character.py \
  characters/unrigged_avatar.glb \
  outputs/characters/avatar_rigged.fbx \
  --blender /path/to/blender
```

The command writes both:

```text
outputs/characters/avatar_rigged.fbx
outputs/characters/avatar_rigged.fbx.json
```

### Python

```python
from motius.motion import auto_rig_character

result = auto_rig_character(
    "characters/unrigged_avatar.glb",
    "outputs/characters/avatar_rigged.fbx",
    blender_executable="/path/to/blender",
)

print(result.output_path)
print(result.manifest_path)
print(result.armature_name)
print(result.joint_names)
print(result.metadata["fit_diagnostics"])
print(result.metadata["skin_diagnostics"])
print(result.warnings)
```

## Input Contract

The built-in fitter expects all of the following:

- exactly one humanoid character, not a crowd or a complete environment;
- upright body with the head above the torso and feet below the hips;
- approximately symmetric T pose, A pose, or relaxed arm pose;
- separated enough limbs for the silhouette to expose arms and legs;
- a conventional human topology and proportions;
- correct coordinate metadata for FBX/GLTF, or an explicit up axis for
  OBJ/PLY/STL when Z-up is wrong.

The following inputs require manual cleanup or a different rigging method:

- seated, running, crouched, crossed-limb, or heavily asymmetric poses;
- arms fused to the torso or legs fused to each other;
- loose skirts, capes, wings, tails, weapons, or large disconnected props;
- quadrupeds or non-humanoid creatures;
- detailed finger, face, hair, or cloth rigs;
- multiple characters in one asset;
- scans whose missing or noisy geometry changes the body silhouette.

### Supported formats

| Stage | Formats | Coordinate behavior |
| --- | --- | --- |
| Input | `.fbx`, `.glb`, `.gltf`, `.obj`, `.ply`, `.stl` | FBX/GLTF use embedded metadata; OBJ/PLY/STL default to Z-up |
| Output | `.fbx`, `.glb`, `.gltf` | Exported with conventional Y-up metadata |

For an OBJ, PLY, or STL with a different up axis:

```bash
python tools/auto_rig_character.py character.obj rigged.fbx --up-axis Y
```

Accepted explicit axes are `X`, `Y`, `Z`, `-X`, `-Y`, and `-Z`. An explicit
axis is rejected for FBX and GLTF because those formats already declare their
coordinates and a second conversion would rotate the character twice.

## What The Algorithm Does

### 1. Import and scene filtering

Blender imports the source and collects mesh geometry. Non-character helper
objects used only as custom bone shapes are excluded. Existing armatures and
Armature modifiers are rejected by default so an already rigged asset is not
silently rebound.

Set `replace_existing_rig=True` only when re-rigging is intentional. The source
file is never overwritten: the old rig is removed only from Blender's exported
working scene, and input and output paths must be different.

### 2. Geometry-only skeleton fit

`fit_humanoid_skeleton()` derives a 22-joint rest skeleton from the character's
axis-aligned dimensions and silhouette. It estimates hips, spine, knees,
ankles, feet, neck, head, clavicles, shoulders, elbows, and wrists, then emits
the canonical names and parent topology used by Motius:

```text
Pelvis, L_Hip, R_Hip, Spine1, L_Knee, R_Knee, Spine2,
L_Ankle, R_Ankle, Spine3, L_Foot, R_Foot, Neck,
L_Collar, R_Collar, Head, L_Shoulder, R_Shoulder,
L_Elbow, R_Elbow, L_Wrist, R_Wrist
```

This is not semantic landmark detection. In Blender working coordinates the
fitter assumes Z-up and `-Y` forward; it does not detect the nose, eyes, heel,
toe, or true anatomical joint centers from mesh semantics.

### 3. Skin binding

Two methods are available:

| `weight_method` | Behavior | Recommended use |
| --- | --- | --- |
| `capsules` | Deterministic distance-to-bone weights with endpoint and left/right priors | Default; reproducible baseline and disconnected geometry |
| `automatic` | Blender's topology-aware automatic bone heat, then limited to `top_k` influences | Connected body meshes that work well with Blender automatic weights |

The capsule method:

- computes weights against the segment controlled by each joint;
- penalizes cross-body left/right assignments;
- adds explicit head, wrist, and foot endpoint priors;
- keeps at most `top_k` influences per vertex;
- normalizes each non-empty row to unit weight;
- reports unbound vertices and active joints.

`automatic` can fail on disconnected or non-manifold geometry. When it leaves
vertices unbound, use a connected/watertight body or switch to `capsules`.

### 4. Export

The armature is named `Motius_SMPL22_Rig`. FBX export embeds transferable image
textures where Blender supports them. GLB is self-contained; `.gltf` uses
Blender's separate GLTF export and may create sidecar binary/texture files.
Shader conversion between formats can be lossy, so inspect materials after an
FBX/GLTF round trip.

## API Reference

```python
auto_rig_character(
    character_path,
    output_path,
    *,
    method="template",
    blender_executable=None,
    up_axis="auto",
    top_k=4,
    weight_falloff=1.75,
    side_penalty=0.025,
    weight_method="capsules",
    replace_existing_rig=False,
)
```

| Parameter | Meaning |
| --- | --- |
| `character_path` | Existing unrigged source asset |
| `output_path` | Different `.fbx`, `.glb`, or `.gltf` destination |
| `method` | Currently only `template` |
| `blender_executable` | Blender path; otherwise resolve `MOTIUS_BLENDER` or `PATH` |
| `up_axis` | `auto` or an explicit axis for OBJ/PLY/STL |
| `top_k` | Maximum influences per vertex, from 1 to 4 |
| `weight_falloff` | Capsule-distance softmax falloff; larger values sharpen assignments |
| `side_penalty` | Cross-body influence multiplier in `(0, 1]`; smaller values separate left/right more strongly |
| `weight_method` | `capsules` or Blender `automatic` |
| `replace_existing_rig` | Permit intentional removal of an imported rig in the output scene |

The returned `CharacterRiggingResult` contains the output and manifest paths,
method, armature name, mesh names, joint names, warnings, and the full parsed
manifest under `metadata`.

## Manifest And Validation

Every run writes `<output-path>.json`. Important fields include:

| Field | Interpretation |
| --- | --- |
| `joint_names`, `joint_parents`, `rest_joints` | Generated SMPL-22 rest skeleton |
| `mesh_count`, `vertex_count`, `face_count` | Imported geometry inventory |
| `removed_rigs` | Rigs removed only when replacement was authorized |
| `fit_diagnostics.quality_score` | Geometry-support heuristic, not an accuracy probability |
| `fit_diagnostics.pose_width_ratio` | Width/height evidence for T/A-pose support |
| `fit_diagnostics.depth_height_ratio` | Detects suspicious depth or orientation |
| `fit_diagnostics.endpoint_surface_distance_mean_ratio` | Surface support near head, wrists, and feet |
| `fit_diagnostics.inferred_pose` | Coarse `T` or `A/relaxed` classification |
| `skin_diagnostics.unbound_vertices` | Vertices with no effective influence |
| `skin_diagnostics.active_joints` | Bones that own at least one weighted vertex |
| `skin_diagnostics.weight_sum_error_max` | Maximum deviation from normalized unit weight |
| `skin_diagnostics.max_influences` | Maximum retained influences per vertex |
| `warnings` | Conditions that require visual inspection |

Do not use `quality_score` as an automatic acceptance threshold by itself. At
minimum, inspect the following views before accepting a character:

1. front and side rest skeleton over the source mesh;
2. dominant-weight colors from front and back;
3. shoulder, elbow, hip, knee, ankle, neck, and wrist bends;
4. the entire animation, not only a few selected frames;
5. texture/material round trips in the final target format.

## Rigging Versus Retargeting

Auto-rigging estimates a **rest skeleton and skin weights**. It does not infer
an animation from a static mesh, and it does not make an under-specified motion
representation more informative.

In particular, SMPL-22 joint positions alone do not encode complete joint
rotations:

- `Ankle -> Foot` is one anatomical direction, not a sole plane. Rotation
  around that direction is unobserved, so a position-only bridge cannot prove
  that the sole is parallel to the floor.
- `Neck -> Head` locates the head center, not the nose or gaze direction. A
  position-only bridge cannot recover independent head orientation.
- SMPL-22 has no heel, toe-tip, face, or head-tip landmarks and no finger
  joints.

Accurate foot and head orientation requires richer observations such as SMPL
pose rotations, heel/toe landmarks, face landmarks, or an explicit constrained
IK/contact solver. A heuristic that forces feet flat or faces forward is a
retargeting prior, not evidence that Auto-Rig recovered the true orientation.

The public `export_motion_to_fbx()` path should be preferred when the source
representation can produce proper SMPL pose rotations. Some representations
also require an installed SMPL body model; see the [FBX guide](fbx.md) for the
exact route and license boundary.

## Verified Public-Mesh Demo

`assets/motion/auto_rigging_demo/` contains a reproducible structural smoke
test using Blender Studio's **Human Base Meshes v1.0.0**, released under CC0:

```text
https://download.blender.org/demo/bundles/bundles-3.6/human-base-meshes-bundle-v1.0.0.zip
SHA-256: 46a912c0524072ac3b78c35d5d2471df7b8df102394a050ca8cd7184e3393648
```

The build downloads and verifies the archive, exports a body mesh, re-imports
it in a factory Blender scene, and rejects the input if it contains an
armature, Armature modifier, vertex group, or Action. It then runs the same
public `auto_rig_character()` backend, validates 22 canonical bones, normalized
weights, zero unbound vertices, and measured mesh deformation.

```bash
python tools/build_auto_rigging_demo.py --blender /path/to/blender
```

The committed animation is a deformation and pipeline smoke test for that
specific mesh-motion pair. Because it is driven from a persisted SMPL-22 joint
position trajectory, it is **not** evidence of recovered foot-sole normals,
ground contact, gaze, or true head rotation. Read the machine reports together
with the visual QA instead of treating the rendered video as a universal
quality claim.

## Troubleshooting

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Character is sideways or upside down | Wrong OBJ/PLY/STL up axis | Pass the correct `--up-axis`; do not override FBX/GLTF metadata |
| Arms bind into torso | Pose too narrow or arms touch body | Use a cleaner T/A pose or manual weights |
| Left/right weights cross the body | Character is off-center or asymmetric | Normalize the source and inspect `side_penalty`; manual cleanup may be required |
| Vertices remain unbound with `automatic` | Disconnected/non-manifold topology | Use `capsules` or repair the mesh |
| Clothing deforms with the wrong limb | Geometry-only capsules cannot infer garment semantics | Separate/clean clothing or author weights manually |
| Face points in the wrong direction | Fitter assumes `-Y` forward and does not detect facial semantics | Normalize source forward orientation before rigging |
| Planted foot is tilted after animation | Retarget source lacks sole orientation/contact constraints | Use pose rotations or a heel/toe/contact-aware retargeter |
| Existing rig is rejected | Safety guard detected an armature/modifier | Keep the existing rig or explicitly opt into `replace_existing_rig=True` |
| Textures change after export | Shader conversion or missing sidecars | Prefer GLB for self-contained GLTF; inspect and package external files |

## Reproducibility And Licensing

The template and capsule methods are deterministic for the same imported
geometry and configuration. Blender version and importer/exporter changes can
still affect object transforms, materials, bone heat, and serialized files.

Motius does not grant rights to third-party character meshes, textures, fonts,
or body models. Verify each source license before downloading, redistributing,
or publishing generated FBX/GLTF assets.
