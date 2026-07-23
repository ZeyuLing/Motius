<p align="center">
  <img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/brand/motius-logo-readme.png" width="520" alt="Motius">
</p>

<p align="center">
  <strong>Open infrastructure for human motion models, benchmarks, and interoperable motion data.</strong>
</p>

<p align="center">
  Train, run, compare, and connect motion systems without rebuilding the runtime around every method.
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.9%2B-2563EB?style=flat-square&logo=python&logoColor=white" alt="Python 3.9+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-F9735B?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch 2.0+"></a>
  <a href="docs/model_zoo/README.md"><img src="https://img.shields.io/badge/Model_Zoo-30_methods-111827?style=flat-square" alt="30 Model Zoo methods"></a>
  <a href="docs/leaderboards/README.md"><img src="https://img.shields.io/badge/Benchmarks-12_suites-15A6C8?style=flat-square&logo=huggingface&logoColor=white" alt="12 benchmark suites"></a>
</p>

<p align="center">
  <a href="#start-here">Start Here</a> ·
  <a href="docs/tasks/README.md">Tasks</a> ·
  <a href="docs/model_zoo/README.md">Models</a> ·
  <a href="docs/leaderboards/README.md">Benchmarks</a> ·
  <a href="docs/motion/README.md">Motion Toolkit</a> ·
  <a href="docs/architecture.md">Architecture</a>
</p>

Motius packages motion methods behind consistent bundles, task pipelines,
trainers, evaluators, and representation bridges. The repository separates
four concerns that motion projects often mix together:

- **Tasks** define input and output contracts.
- **Methods** implement one or more tasks and live in the Model Zoo.
- **Benchmarks** bind a task to a dataset, split, protocol, evaluator, and
  persisted results.
- **Motion data** moves through explicit representation, body, character, and
  robot conversion routes.

## Start Here

Install from source:

```bash
git clone https://github.com/ZeyuLing/Motius.git
cd Motius
python -m pip install -e ".[dev]"
```

Run a released Text-to-Motion model:

```python
from motius.pipelines.momask import MoMaskPipeline

pipe = MoMaskPipeline.from_pretrained(
    "ZeyuLing/hftrainer-momask-humanml3d",
    device="cuda",
)
motions = pipe.infer_t2m(
    ["a person walks forward and then sits down"],
    [120],
)
print(motions[0].shape)  # (120, 263), HumanML3D physical scale
```

Continue with the [installation and smoke tests](docs/getting_started.md), then
choose a [task](docs/tasks/README.md), a released
[method](docs/model_zoo/README.md), or a
[benchmark](docs/leaderboards/README.md).

## Task System

The [Task Registry](docs/tasks/README.md) is the only public task vocabulary.
Tracks such as prediction, in-betweening, sparse keyframes, and TP2M remain
inside their parent task. Dataset and model names never become task names.

- **Language and motion:** Text-to-Motion, Motion-to-Text, Sequential
  Text-to-Motion, and Text-to-Multi-Person Motion.
- **Conditioned motion:** Temporal Motion Completion, Kinematic Motion Control,
  and Part-Level Motion Control.
- **Transformation and restoration:** Motion Editing, Motion Repair, and Motion
  Reconstruction.
- **Audio and motion:** Music-to-Dance, Dance-to-Music, and Speech-to-Gesture.
- **Embodied motion:** Robot Motion Control.

Model cards use these exact labels. Benchmark titles use
`Task · Dataset/Protocol`, such as
`Text-to-Motion · HumanML3D` and
`Sequential Text-to-Motion · BABEL`.

## Models And Benchmarks

The **[Model Zoo](docs/model_zoo/README.md)** provides a task-first index and an
alphabetical catalog of 30 integrated method packages. Every card records its native
motion representation, frame rate, checkpoint, evaluation boundary, paper,
upstream code, and license.

The **[Benchmark Hub](docs/leaderboards/README.md)** maintains twelve suites
across language, temporal and kinematic control, editing and restoration, and
audio-driven motion. Public result pages are backed by persisted repository
protocols and qualitative case explorers.

Evaluator implementations remain separate from benchmark identity. Open the
**[Evaluator Zoo](docs/evaluator_zoo)** for HumanML3D Official,
MotionStreamer, InterCLIP, TMR-G1, AIST++, and shared joint-position evaluators;
see [physical metrics](docs/evaluation/physical_metrics.md) for foot slide,
floating, jitter, dynamics, and floor penetration.

## Motion Interoperability

### Representation And Embodiment

<p align="center">
  <img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/motion/representation_demo/004822_hml_smpl_soma_core_g1_1920_30fps.gif" width="920" alt="One motion converted across HumanML3D, SMPL, SOMA, ARDY, and Unitree G1">
</p>

<p align="center">
  <code>HumanML3D-263</code> → <code>SMPL-22</code> → <code>SOMA-30</code> /
  <code>ARDY-Core-27</code> → <code>Unitree G1</code>
</p>

<p align="center">
  <a href="assets/motion/representation_demo/index.html"><strong>Open synchronized viewer</strong></a> ·
  <a href="docs/motion/representations.md">Representation matrix</a> ·
  <a href="docs/motion/conversion.md">Conversion API</a> ·
  <a href="docs/motion/retargeting.md">Retargeting routes</a>
</p>

The shared bridge connects native feature vectors, SMPL-family bodies,
kinematic skeletons, rigged characters, and robot embodiments. Exact routes
preserve source state; lossy joint-only and cross-skeleton routes expose their
IK or retargeting diagnostics.

Actor count is an orthogonal layout property. Single-person motion uses
`(T, D)`; paired and multi-person motion use `(T, A, D)` in one shared world
frame. The [GT InterX comparison](assets/motion/interhuman_representation_demo/interx_smplh_gt_G021T002A012R014_skeleton_smpl_mesh.gif)
and [Three.js viewer](assets/motion/interhuman_representation_demo/index.html)
show InterHuman-262 and SMPL-H under the same conversion contract.

### Character Export

<p align="center">
  <img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/motion/fbx_character_demo/004822_skeleton_smpl_mixamo_1440_readme_30fps.gif" width="920" alt="One motion transferred from an SMPL skeleton and mesh to four rigged FBX characters">
</p>

Any supported human-motion representation can pass through the SMPL-22 bridge
into a rigged FBX character. Compare the same motion as a skeleton, an SMPL
mesh, and four Mixamo characters in the
[30 fps character preview](assets/motion/fbx_character_demo/004822_skeleton_smpl_mixamo_1440_30fps.gif),
then follow the [FBX export guide](docs/motion/fbx.md).

## Train And Extend

Train a Python config locally or with Accelerate:

```bash
python tools/train.py path/to/config.py --work-dir outputs/my_experiment
accelerate launch tools/train.py path/to/config.py \
  --work-dir outputs/my_distributed_experiment --auto-resume
```

A method package owns its model, bundle, trainer adapter, task pipeline, and
evaluation adapters. The common runtime owns distributed execution,
checkpoint I/O, registration, conversion, and lifecycle hooks.

Use the [architecture guide](docs/architecture.md) to add a package, the
[development guide](docs/development.md) for repository conventions, and the
[PRISM, TMR, and HY-Motion recipes](docs/training/prism_tmr_hymotion_t2m.md)
for released training examples.

## Architecture

```mermaid
flowchart LR
    C["Config and dataset"] --> R["Registry"]
    R --> B["ModelBundle"]
    B --> T["Trainer"]
    B --> P["Task pipeline"]
    P --> E["Benchmark adapters"]
    P --> M["Motion Toolkit"]
    T --> X["Distributed runtime"]
    E --> L["Persisted results"]
    M --> O["Bodies, characters, robots"]
```

The split is intentional: a method can change architecture without renaming
its task, a benchmark can change dataset without becoming a new method, and a
motion representation can change without silently changing evaluation space.

## Documentation

[Getting Started](docs/getting_started.md) ·
[Task Registry](docs/tasks/README.md) ·
[Model Zoo](docs/model_zoo/README.md) ·
[Benchmark Hub](docs/leaderboards/README.md) ·
[Evaluator Zoo](docs/evaluator_zoo) ·
[Motion Toolkit](docs/motion/README.md) ·
[Architecture](docs/architecture.md) ·
[Training Recipes](docs/training/prism_tmr_hymotion_t2m.md) ·
[Development](docs/development.md)

## Project Status

Motius is an active research release. Public artifacts are versioned,
evaluation protocols are persisted with their results, and method-specific
licenses and upstream attribution remain documented in each model card. Core
APIs may evolve as additional methods move onto the shared runtime.
