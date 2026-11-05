# Current promotional cut — selection, reconstruction review and measured performance

[Watch the updated demo](motius-flowhmr-explained-final-1080p.mp4) · [Edit/frame audit](../../../../docs/architecture/validation/flowhmr-blender-explained.json) · [Performance evidence and limits](../../../../docs/architecture/validation/flowhmr-user-performance.json)

The selected-person review is shown continuously for 8.64 seconds. A newly recorded 13.5-second raw reconstruction inspection enlarges the model, rotates the view and inspects another pose. Both result stages retain room to see the output before the next operation. The result-first opening and complete 25-second / 750-frame original-speed finale remain. A nine-second performance page reports the measured H200 30-frame case, includes loading time, identifies PyTorch allocated memory and explicitly avoids a consumer-GPU/8 GB compatibility claim.

The H200 single-run full short-video timings are 45.56 s standard and 44.79 s accelerated; allocated peaks are 9.70 and 7.23 GiB. This supports reduced memory use, not a broad claim of faster full-video inference. Consumer-GPU full-length cold-start speed and whole-application memory are not measured. The separate CPU demonstration reused verified frontends and took about 88 seconds from Generate to visible reconstruction. These are different measurements and must not be substituted for one another.

# Results-first promotional demo — preferred final cut

[Watch the 64-second final cut](motius-flowhmr-results-first-final-1080p.mp4) · [Edit decisions and final frame audit](../../../../docs/architecture/validation/flowhmr-blender-results-first.json)

Open with seven seconds of three unobscured characters at original speed. Then show one task per caption: select video, click/review the person, generate motion, import a character, inspect the reconstruction, retarget and ground the cast. Real Blender control close-ups replace wide shots of tiny buttons. Repeated character imports and waits are omitted. End with the complete 750-frame / 30 fps / 25-second performance. All result shots have the plugin sidebar closed; procedural close-ups focus on controls rather than presenting an obstructed cast.

The final encoded 25-second section contains all source counters 1–750 in order, verified optically across its 1500 screen frames. UI close-ups are crops of actual screen recordings. There is no motion retiming. Footage, caching and the identical-motion provenance remain as documented for the prior cut below.

# Final complete Blender workflow demo

[Watch the final 1080p promotional cut](motius-flowhmr-complete-demo-final-1080p.mp4) · [Provenance and frame audit](../../../../docs/architecture/validation/flowhmr-blender-promo-final.json)

102 seconds of actual Blender UI: select the supplied video, click/review the detected person, run fresh FlowHMR sampling while importing three characters, inspect the native reconstruction, retarget/ground the cast, then play all three alongside the source at the original 30 fps. **All 750 motion frames play in 25 seconds.** Only waits are shortened, with explicit on-screen captions; the playback shot is continuous and unretimed.

The operation take was recorded on October 26. The final playback shot is the previously verified October 23 capture, using identical inputs and numerically identical fresh motion: maximum differences in joint positions, global rotations and rest joints are all zero. Detection and body/hand/camera preprocessing are cached; neural sampling and surface reconstruction run again. This is an edited workflow demonstration, not a cold-start speed benchmark. The final encoded playback was optically checked for source frames 1–750. The clip is silent.

# Original-speed Blender playback — 30 fps

[Watch the actual UI recording](motius-flowhmr-blender-original-speed-1080p.mp4) ·
[Verification](../../../../docs/architecture/validation/flowhmr-blender-original-speed.json)

**750 frames play in 25.000 seconds**, with no skipped frames or video time stretch.
The 29.15-second recording includes clicking the real playback button and an end hold.
Blender prepares a viewport/source cache first, then plays it at the original rate; the
editable scene returns afterward. Cache preparation precedes this recording. Both draw
sequences and all 1749 screen-video frames were checked, including the visible source counter.
This replaces the slow playback in the historical v2 walkthrough below.

Updated scene and both 0.2.1 extension packages:
`outputs/validation/flowhmr/blender-workflow-v3/flowhmr-blender-original-speed.zip`.

# FlowHMR · Motius — actual Blender workflow v2

[Watch the continuous 1080p UI recording](motius-flowhmr-blender-workflow-v2-1080p.mp4) ·
[Actual UI still](motius-flowhmr-blender-workflow-v2-poster.png) ·
[Verification](../../../../docs/architecture/validation/flowhmr-blender-workflow-v2.json)

- **00:01** — review real bounding boxes, click the intended person, review trajectory and confirm.
- **00:10** — Generate motion using verified frontend reuse; sampling runs again.
- **00:13** — import Y-Bot, Xbot and Michelle while inference runs.
- **01:23** — inspect the native FlowHMR reconstruction in an isolated view.
- **01:26** — retarget and ground all three characters individually.
- **02:10** — play the entire animation and source: both viewports draw frames 1–750 in order.
- **03:58** — playback stops at frame 750; save the editable scene.

This is uncut actual Blender UI operation with the supplied video, not a rendered comparison film.
It runs on CPU with software graphics. **The full 25-second animation plays in about 108 seconds**
to preserve every frame; no retiming is baked into the 30 fps actions. The recording is silent.
Detector output is prepared before recording; full-hand frontend caches are verified and reused,
while neural sampling and surface reconstruction execute again. This is not a cold-start GPU benchmark.

The editable portable bundle is `outputs/validation/flowhmr/blender-workflow-v2/flowhmr-blender-workflow-v2.zip`.
It includes both 0.2.0 extension variants, the scene, source video, 750 preview frames and validation.
Read the included README for setup, asset attribution and selection/grounding limits.
The synthetic two-person test checks separate track selection; it does not establish reidentification
through arbitrary real crossings or occlusions.

# Earlier Blender walkthrough (superseded)

[Watch the actual Blender UI walkthrough](motius-flowhmr-blender-workflow-1080p.mp4) ·
[Actual UI still](motius-flowhmr-blender-workflow-poster.png) ·
[Workflow verification](../../../../docs/architecture/validation/flowhmr-blender-workflow.json)

The continuous 1080p / 30 fps screen capture shows actual Blender 4.5.3 UI operation:

- **00:00** — choose the supplied video and preview it in Blender.
- **00:07** — show Standard / Accelerated mode selection.
- **00:11** — enable hash-verified video frontend reuse.
- **00:12** — Generate Motion: run fresh 20-step FlowHMR sampling on CPU.
- **01:36** — inspect and play the actual reconstructed SMPL-H surface.
- **01:43** — import the Mixamo character FBX through the file browser.
- **01:54** — retarget the generated body and hand motion.
- **02:07** — apply Ground & Plant Feet, including actual mesh validation.
- **02:34** — compare Raw / Grounded actions.
- **02:47** — play the 750-frame range; this earlier take did not audit actual frame presentation.
- **03:16** — export the grounded FBX and save the editable scene.

All window frames and waiting times are retained; there are no composited model plates,
preloaded result animations or substituted progress screens in this walkthrough.
The floor grid is scene geometry; viewport following preserves the character's world travel.
The recording uses the Latest checkpoint and previously measured full body/hand frontend
features from the same video, camera interval 100 and the same frontend weights. Cache
reuse is explicit in the panel; neural motion sampling and surface reconstruction run again.
This is a CPU workflow demonstration, not a cold-start GPU speed benchmark.

The editable local scene is `outputs/validation/flowhmr/blender-workflow/flowhmr-workflow.blend`;
keep its sibling `input.mp4` alongside it. Raw and Grounded actions are both retained.
The exported character is `flowhmr-workflow-grounded.fbx` in the same directory.
The 750-frame grounding check measures actual skinned soles, locks detected support feet,
checks swing-foot collisions and preserves 14 verified flight frames with 0.527 m peak clearance.
This is a contact-based correction of the Mixamo animation; the raw FlowHMR prediction remains available.

# Earlier rendered result film

[Watch / download the 1080p film](motius-flowhmr-video-to-mixamo-1080p.mp4) ·
[Poster](motius-flowhmr-video-to-mixamo-poster.png) ·
[Provenance and checks](motius-flowhmr-video-to-mixamo-1080p.json)

This 25-second film shows the supplied video, its actual FlowHMR reconstruction,
and a skinned Mixamo character driven by that same prediction. All 750 frames
play at their original 30 fps. The original AAC audio packets are copied unchanged.

FlowHMR is **integrated into Motius**. The checkpoint is
[FlowHMR Latest](https://huggingface.co/ZeyuLing/Motius-FlowHMR-Latest), using seed 0,
20 Euler steps and CFG 1. Full body and both-hand features are extracted for every
frame. VGGT estimates 9 camera keyframes at interval 100, interpolated to 750;
this does not subsample the video or the per-frame motion conditioning.
This run uses native CPU FP32 execution with the lightweight attention/token path.

The input is reframed for the three-column composition. Both 3D views use a
fixed camera, horizontal root centering and one constant floor offset per stage.
The saved prediction receives no pose smoothing, foot IK or motion cleanup.

The editable local scene is
`outputs/validation/flowhmr/promo/flowhmr-mixamo-studio.blend`.
It contains independently playable SMPL-H surface animation and the skinned Mixamo
animation; playback needs neither Torch nor the add-on. The local FBX retains the
predicted horizontal root trajectory scaled to the character's leg length.
For matching Blender frame numbers when reimporting that FBX, use Animation
Offset **0**. The default FBX importer otherwise adds one frame to the timeline.

The model comes from the public three.js
[Samba Dancing FBX example](https://github.com/mrdoob/three.js/blob/dev/examples/models/fbx/Samba%20Dancing.fbx).
Its original animation is not imported. All 52 mapped joints, including fingers,
are driven by this FlowHMR capture. Character credit: Mixamo / three.js example.
No source body models or inference weights are included in the release media.

Generation uses the Motius FlowHMR environment and these repository tools:

1. `tools/infer_flowhmr.py` — native video inference / capture.
2. `tools/prepare_flowhmr_blender_assets.py` — capture to mesh and Blender protocol.
3. `tools/render_flowhmr_promo.py` — genuine Blender SMPL-H and Mixamo plates.
4. `tools/compose_flowhmr_promo.py` — synchronized 1080p composition with audio.

References: [FlowHMR](https://github.com/flowhmr/flowhmr),
[Motion2MixamoRig workflow](https://github.com/MaxLiu22/Motion2MixamoRig),
[FlowHMR engineering discussion](https://github.com/flowhmr/flowhmr/issues/4).
