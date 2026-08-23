# Automatic Character Rigging

Motius provides two paths for fitting an animation-ready armature and skin to a
static humanoid mesh:

```text
GLB / GLTF / FBX / OBJ / PLY / STL
  -> Blender import and coordinate normalization
  -> local geometry-only template fit OR Make-It-Animatable inference
  -> canonical SMPL-22 normalization
  -> skin-weight estimation
  -> skinned FBX / GLB / GLTF + JSON manifest
```

The result is a **static rigged character**. Automatic rigging and animation
retargeting are separate stages: this API creates the skeleton and skin, while
the [FBX motion bridge](fbx.md) applies motion to a rigged target.

> [!IMPORTANT]
> The local `template` method is an experimental, deterministic baseline for
> one upright T/A-pose humanoid. A successful export proves that the mesh is
> structurally bound; it does not prove production-quality anatomy, weights,
> bone roll, foot contact, or facial orientation. Inspect the generated rig and
> deformation before using it in a dataset, paper, game, or robot workflow.

## Requirements

- Python 3.10 or newer;
- Motius installed from source;
- Blender 3.6 or newer, either on `PATH`, in `MOTIUS_BLENDER`, or passed with
  `blender_executable` / `--blender`.

The optional Make-It-Animatable backend additionally requires:

```bash
pip install -e ".[auto-rig]"
```

It uses `gradio-client` to call `jasongzy/Make-It-Animatable` by default. That
public Space receives the uploaded character file. Use `mia_space` or
`--mia-space` to point private assets at a trusted self-hosted deployment.

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

### Diverse-character backend

```bash
python tools/auto_rig_character.py \
  characters/stylized_or_unusual.glb \
  outputs/characters/stylized_rigged.fbx \
  --method make-it-animatable \
  --blender /path/to/blender
```

The same path is available in Python:

```python
result = auto_rig_character(
    "characters/stylized_or_unusual.glb",
    "outputs/characters/stylized_rigged.fbx",
    method="make_it_animatable",
    blender_executable="/path/to/blender",
    mia_space="jasongzy/Make-It-Animatable",
)
```

This backend delegates skeleton and skin prediction to the upstream
Make-It-Animatable model, then uses Blender to bake the imported FBX transform,
map the Mixamo-compatible canonical subset to Motius SMPL22 names, preserve
skin groups and textures, and export FBX/GLB/GLTF. Additional upstream bones,
such as fingers, can remain in the output; the named SMPL22 subset is the
contract consumed by Motius retargeting.

## Input Contract

The built-in `template` fitter expects all of the following:

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

## What The Local Template Algorithm Does

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
    mia_space="jasongzy/Make-It-Animatable",
    mia_rest_pose="No",
)
```

| Parameter | Meaning |
| --- | --- |
| `character_path` | Existing unrigged source asset |
| `output_path` | Different `.fbx`, `.glb`, or `.gltf` destination |
| `method` | `template` for the local baseline; `make_it_animatable` (or `mia`) for the optional high-coverage backend |
| `blender_executable` | Blender path; otherwise resolve `MOTIUS_BLENDER` or `PATH` |
| `up_axis` | `auto` or an explicit axis for OBJ/PLY/STL |
| `top_k` | Maximum influences per vertex, from 1 to 4 |
| `weight_falloff` | Capsule-distance softmax falloff; larger values sharpen assignments |
| `side_penalty` | Cross-body influence multiplier in `(0, 1]`; smaller values separate left/right more strongly |
| `weight_method` | `capsules` or Blender `automatic` |
| `replace_existing_rig` | Permit intentional removal of an imported rig in the output scene |
| `mia_space` | Public or trusted self-hosted Make-It-Animatable Gradio Space |
| `mia_rest_pose` | Upstream rest-pose request: `No`, `T-pose`, or `A-pose` |

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

The `make_it_animatable` manifest instead records the upstream Space, canonical
SMPL22 subset, any additional upstream bones, output meshes, and normalization
warnings. Local template fit/skin diagnostics do not exist for that backend and
must not be fabricated from the upstream prediction.

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

## Verified Multi-Character Demo

`assets/motion/auto_rigging_demo/` contains the approved synchronized demo for
three downloaded characters with visibly different proportions:

| Demo role | Source | Source license |
| --- | --- | --- |
| Child | `https://sketchfab.com/3d-models/little-girl-rigged-3d-model-2b5c9b749a714dd6b0d04c7de83c254e` | CC BY 4.0 |
| Big head | `https://sketchfab.com/3d-models/running-boy-0c80a9edd3514af0902349233d2c8d8f` | CC BY-NC 4.0 |
| High weight | `https://sketchfab.com/3d-models/yokai-project-oni-62f0e50b3f3543febaddd4834b05a84d` | CC BY 4.0 |

Despite the display name of the first source, the downloadable Sketchfab GLB
used here contains no rig. For every input, Blender audits found zero
armatures, Armature modifiers, vertex groups, and Actions before AutoRig. The
three authored texture sets are retained in the rigged and animated assets.

The demo pipeline is deliberately explicit:

```text
downloaded unrigged textured GLB
  -> input-contract and material audit
  -> Make-It-Animatable skeleton + skin inference
  -> Motius Mixamo-compatible to SMPL22 normalization
  -> stored HumanML3D motion 004822
  -> Motius retargeting and orientation priors
  -> deformation/direction diagnostics
  -> synchronized Blender render with generated skeleton overlay
```

Run the rendering stage on three already animated SMPL22-compatible FBX files:

```bash
blender --background --python tools/blender_render_textured_autorig_video.py -- \
  --character "CHILD|CG-Moon / Little Girl|CC BY|child.fbx" \
  --character "BIG-HEAD|Suushimi / Running boy|CC BY-NC|running_boy.fbx" \
  --character "HIGH-WEIGHT|kornasale / Yokai Oni|CC BY|oni.fbx" \
  --output demo.mp4 --frames-dir review_frames --report render.json --preview
```

The committed MP4 is a 150-frame, 30 fps deformation and integration smoke
test. It demonstrates three accepted examples, not a universal guarantee for
every mesh. It is driven from an SMPL-22 joint-position trajectory plus explicit
Motius foot-contact and head-stabilization priors; it is not evidence that the
static input alone exposes the true sole plane or gaze direction.

`tools/build_auto_rigging_demo.py` remains available as a fully local,
single-character `template` regression builder. Its default publication path
is under `outputs/` so it cannot overwrite the approved multi-character media.

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
geometry and configuration. Make-It-Animatable results depend on the configured
upstream service and model revision. Blender version and importer/exporter
changes can still affect object transforms, materials, bone heat, and serialized
files.

Motius does not grant rights to third-party character meshes, textures, fonts,
or body models. Verify each source license before downloading, redistributing,
or publishing generated FBX/GLTF assets.
