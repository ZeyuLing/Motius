<!-- MOTIUS_DOCS_NAV:START -->
<p><a href="../../README.md"><strong>Motius</strong></a> / <a href="../README.md">Documentation</a> / <strong>Dataset Hub</strong></p>
<p>
  <a href="../getting_started.md">Quickstart</a> ·
  <a href="../tasks/README.md">Tasks</a> ·
  <strong>Datasets</strong> ·
  <a href="../model_zoo/README.md">Models</a> ·
  <a href="../training/README.md">Training</a> ·
  <a href="../evaluator_zoo/README.md">Evaluators</a> ·
  <a href="../leaderboards/README.md">Benchmarks</a> ·
  <a href="../motion/README.md">Motion I/O</a>
</p>
<!-- MOTIUS_DOCS_NAV:END -->

# Motius Dataset Hub

This page is the source of truth for dataset access in Motius. It distinguishes
an upstream dataset from a Motius-normalized copy and records the local root
expected by released tools. Always follow the upstream license and citation
requirements, including when a processed copy is hosted by Motius.

## Dataset Directory

| Dataset | Motius access | Official source | Default local root | Used by |
| --- | --- | --- | --- | --- |
| <a id="motionhub"></a> **MotionHub** | [Full Hugging Face release](https://huggingface.co/datasets/ZeyuLing/MotionHub) | [Source repository](https://github.com/ZeyuLing/MotionHub) | `data/motionhub` | Generalist T2M/M2T, music, speech, and interaction training |
| <a id="humanml3d"></a> **HumanML3D** | [SMPL-H AMASS subset](https://huggingface.co/datasets/ZeyuLing/MotionHub/tree/main/HumanML3D_AMASS) · [SMPL-H HumanAct12 subset](https://huggingface.co/datasets/ZeyuLing/MotionHub/tree/main/HumanML3D_HumanACT12) | [Official reconstruction workflow](https://github.com/EricGuo5513/HumanML3D) | `data/HumanML3D` | T2M, M2T, temporal/part control, reconstruction |
| <a id="babel"></a> **BABEL** | [Motius official-layout pack](https://huggingface.co/datasets/ZeyuLing/babel-official) | [Official dataset](https://babel.is.tue.mpg.de/data.html) | `data/babel` | Sequential Text-to-Motion |
| <a id="interhuman"></a> **InterHuman** | Not redistributed by Motius | [Official InterGen release](https://github.com/tr3e/InterGen) | `data/InterHuman` | Text-to-Multi-Person Motion |
| <a id="permo"></a> **PerMo** | [Motius Hugging Face release](https://huggingface.co/datasets/ZeyuLing/PerMo) | Terms and attribution in the dataset card | `data/PerMo` | Style/content motion editing, personalized T2M |
| <a id="motionfix"></a> **MotionFix** | [Motius SMPL-H release](https://huggingface.co/datasets/ZeyuLing/MotionFix) | [Official dataset](https://motionfix.is.tue.mpg.de/) | `data/MotionFix` | Instruction-based motion editing |
| <a id="aistpp"></a> **AIST++** | [MotionHub subset](https://huggingface.co/datasets/ZeyuLing/MotionHub/tree/main/aist) | [Official dataset](https://google.github.io/aistplusplus_dataset/) · [API](https://github.com/google/aistplusplus_api) | `data/aistpp` | Music-to-Dance, Dance-to-Music |
| <a id="beat2"></a> **BEAT2** | [MotionHub subset](https://huggingface.co/datasets/ZeyuLing/MotionHub/tree/main/beat_v2.0.0) | [Official Hugging Face release](https://huggingface.co/datasets/H-Liu1997/BEAT2) | `data/beat2` | Speech-to-Gesture |
| <a id="amass"></a> **AMASS** | [Private AMASS-G1 cache](https://huggingface.co/datasets/ZeyuLing/AMASS_GMR_for_G1) · not a public release | [Official download](https://amass.is.tue.mpg.de/) · [License](https://amass.is.tue.mpg.de/license.html) | `data/amass` | BrokenAMASS motion repair; licensed local Isaac Lab tracking preparation |
| <a id="lafan1"></a> **LAFAN1** | [OpenTrack Unitree G1 retarget](https://huggingface.co/datasets/robfiras/loco-mujoco-datasets/tree/main/Lafan1/mocap/UnitreeG1) | [Official dataset](https://github.com/ubisoft/ubisoft-laforge-animation-dataset) | `data/LAFAN1` | MuJoCo Motion Tracking benchmark |
| <a id="3dpw"></a> **3DPW** | Not redistributed by Motius | [Official registration and download](https://virtualhumans.mpi-inf.mpg.de/3DPW/) | `data/3DPW` | Monocular Motion Capture |
| <a id="emdb"></a> **EMDB** | Not redistributed by Motius | [Official project](https://eth-ait.github.io/emdb/) · [Download](https://emdb.ait.ethz.ch/) | `data/EMDB` | Monocular Motion Capture protocol support |

The machine-readable inventory is [`catalog.json`](catalog.json). Dataset names
on the [Task Registry](../tasks/README.md) and
[Benchmark Hub](../leaderboards/README.md) link back to the entries above.

## Download Motius Releases

Install or update `huggingface_hub`, then download a complete Motius-hosted
release into the path expected by the framework:

```bash
python -m pip install -U huggingface_hub

huggingface-cli download ZeyuLing/MotionHub \
  --repo-type dataset \
  --local-dir data/motionhub

huggingface-cli download ZeyuLing/babel-official \
  --repo-type dataset \
  --local-dir data/babel

huggingface-cli download ZeyuLing/MotionFix \
  --repo-type dataset \
  --local-dir data/MotionFix

huggingface-cli download ZeyuLing/PerMo \
  --repo-type dataset \
  --local-dir data/PerMo

huggingface-cli download robfiras/loco-mujoco-datasets \
  --repo-type dataset \
  --include "Lafan1/mocap/UnitreeG1/*.npz" README.md \
  --local-dir data/LAFAN1
```

For an authenticated or gated repository, run `huggingface-cli login` first
after accepting its upstream terms. The equivalent Python API is:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="ZeyuLing/MotionHub",
    repo_type="dataset",
    local_dir="data/motionhub",
)
```

MotionHub can also be downloaded one subset at a time:

```bash
huggingface-cli download ZeyuLing/MotionHub \
  --repo-type dataset \
  --include "aist/**" "aist/*.json" \
  --local-dir data/motionhub
```

## Which Copy To Use

| Goal | Required copy | Why |
| --- | --- | --- |
| Train a generalist model over Motius datasets | Motius [MotionHub](#motionhub) release | Unified SMPL-H files, task annotations, and shared statistics |
| Reproduce a HumanML3D leaderboard | Official [HumanML3D](#humanml3d) root | The protocol requires official HumanML3D-263 features, split files, normalization statistics, and selected captions |
| Train on HumanML3D inside a universal SMPL-H mixture | HumanML3D subsets inside [MotionHub](#motionhub) | These are converted to the MotionHub SMPL-H convention and annotation schema |
| Reproduce Sequential Text-to-Motion | Motius [BABEL](#babel) pack plus accepted upstream terms | The benchmark consumes the processed action-group manifest and resolved AMASS clips |
| Reproduce AIST++ paper protocols | Official [AIST++](#aistpp) data and audio | The MotionHub subset is normalized for Motius training; it is not a drop-in replacement for every paper feature cache |
| Reproduce Speech-to-Gesture | Official [BEAT2](#beat2) release | Use the exact test split and synchronized audio required by the protocol |
| Train/evaluate instruction editing | Motius [MotionFix](#motionfix) release | Source/target SMPL-H pairs and edit instructions share the released split |
| Prepare licensed tracking references from AMASS | User-downloaded [AMASS](#amass) root | The AMASS license is personal and prohibits third-party redistribution; retarget locally |
| Evaluate G1 tracking in MuJoCo | OpenTrack [LAFAN1 G1 retarget](#lafan1) | Uses all 40 named trajectories and the registered 50 Hz protocol |

## Dataset Contracts

### MotionHub

MotionHub is the public multi-domain training release used by the shared
dataset implementations under `motius/datasets/motion/motionhub/`. Its
top-level contract is:

```text
data/motionhub/
  annotations/
    all/{train,test}.json
    text_motion/{train,test}.json
    music_dance/{train,test}.json
    speech_gesture/{train,test}.json
    two_person_interaction/{train,test}.json
  statistics/
  <subset>/
    smplh_52/
    hierarchical_caption/
```

`MotionGV` is train-only in the current public release. Source datasets retain
their own licenses and citation requirements even after conversion into the
MotionHub format.

### HumanML3D

Motius does not publish a standalone `ZeyuLing/HumanML3D` dataset repository.
HumanML3D inherits AMASS redistribution constraints, so reconstruct the
official release using its upstream workflow. A benchmark-ready root contains:

```text
data/HumanML3D/
  new_joint_vecs/
  new_joints/
  texts/
  Mean.npy
  Std.npy
  train.txt
  val.txt
  test.txt
```

The HumanML3D benchmark uses the official test split and the repository's
selected-caption manifest. The two HumanML3D subsets hosted inside MotionHub
contain converted SMPL-H motions for unified training. They do not replace the
official `new_joint_vecs`, `Mean.npy`, `Std.npy`, or caption protocol.

### BABEL

The Motius BABEL pack preserves official labels and the AMASS-backed local
layout while adding normalized sequential-generation manifests:

```text
data/babel/
  labels/babel_v1.0_release/
  amass/
  processed/
    manifests/
    ms272/
    babel_shortmerge_caption_rewrites.json
```

The public protocol is documented in
[Sequential Text-to-Motion evaluation](../evaluation/babel_sequential.md).
Do not upload or redistribute BABEL/AMASS files outside the permissions granted
by their upstream licenses.

### MotionFix And PerMo

The Motius MotionFix release stores paired source/target SMPL-H motions,
free-form edit instructions, and the official train/validation/test split.
PerMo stores SMPL-H motions, captions, style groups, and neutral-to-style edit
pairs. Both are consumed as physical-scale motion; do not apply HumanML3D
statistics to them.

### AIST++, BEAT2, InterHuman, 3DPW, And EMDB

These benchmarks require source assets that carry their own distribution or
registration terms:

- AIST++ evaluation uses synchronized music, split metadata, and SMPL motion
  from the official release.
- BEAT2 evaluation uses synchronized speech audio and its declared test split.
- InterHuman prohibits redistribution; obtain it through the official InterGen
  instructions.
- 3DPW and EMDB require registration and remain user-supplied inputs for
  monocular motion-capture evaluation.

Motius-normalized AIST++ and BEAT2 subsets are available inside MotionHub for
shared-format training. Their presence does not waive upstream terms.

### AMASS And LAFAN1 Tracking References

AMASS requires registration and grants a personal, non-transferable research
license. Its license prohibits copying, sharing, distributing, or making the
dataset available to third parties. Motius therefore does not publish AMASS-G1
motions. Licensed users download AMASS themselves and write any retargeted G1
outputs under `outputs/`.

BrokenAMASS is a protocol-derived repair benchmark, not an independent source
dataset: its 299 pair-validated cases are corruptions of AMASS DanceDB clips.
It therefore inherits AMASS access and redistribution constraints.

LAFAN1 is released under CC BY-NC-ND 4.0. The Motion Tracking benchmark uses
the 40 Unitree G1 NPZ trajectories published for OpenTrack in
`robfiras/loco-mujoco-datasets`, rather than the unrelated 30 Hz CSV retarget.
Motius loads the named 23-DOF layout, inserts absent G1 joints as zero, and
resamples the 40 Hz trajectory to the registered 50 Hz control clock. Keep the
upstream attribution and license metadata with every local copy.

## Evaluation Artifacts Are Not Training Data

The
[Motius Leaderboard Cases](https://huggingface.co/datasets/ZeyuLing/Motius-Leaderboard-Cases)
repository stores persisted predictions, viewer manifests, and qualitative
assets used by public leaderboards. It is an evaluation-artifact store, not a
training dataset and not a replacement for any benchmark source above.

## Validation

Run the local catalog and cross-link audit before publishing dataset-related
changes:

```bash
python tools/audit_datasets.py
```

Add `--online` to probe every external access URL:

```bash
python tools/audit_datasets.py --online
```
