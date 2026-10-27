# Sources and scope

The Blender integration is independently implemented for Motius and licensed
GPL-3.0-or-later. It does not bundle machine-learning dependencies, model
weights, licensed body models, or code from external repositories.

Workflow reference: [Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig),
by MaxLiu22 and contributors, Apache-2.0. Its external-Python, asynchronous-job,
character-import workflow informed this integration. Its implementation uses
GVHMR; this add-on uses Motius's native FlowHMR pipeline. No Motion2MixamoRig
source code is copied into this package.

Acceleration reference: [FlowHMR issue #4](https://github.com/flowhmr/flowhmr/issues/4),
including the corrections about numerical equivalence and the joint softmax
and global-refiner requirements of query-batched attention. This add-on calls
the tested Motius lightweight implementation; it does not install Fast-SAM or
enable body-only/mean-hand conditioning, compilation, or reduced precision.

FlowHMR and its frontend checkpoints have separate component licenses. In
particular, FlowHMR restricts use to educational, research and nonprofit use.
The add-on license grants no additional rights to those models. SMPL-H assets
must be obtained under their separate license. See the Motius FlowHMR model
card and component attribution files for the applicable terms.
