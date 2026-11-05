# Changelog

## 0.3.23 — 2026-10-30

- Add native MotionBricks CUDA pose services with official alignment, boundary decoding, G1 filtering, endpoint correction, SOMA IK, and headless scene/Smart Primitive control.
- Verify GPU raw inference against independent official PyTorch and full generation against the official TensorRT FP32 engine; publish reproducible validation and updated native Hub metadata.

## 0.3.22 — 2026-10-30

- Replace the MotionBricks ONNX adapter with native SOMA/G1 PyTorch neural roots, separate motion representations and pipeline policies, SafeTensors artifacts and unified Hub loading.
- Verify eight raw-generation cases against independently loaded official source at `54fc563a`; CPU FP32 outputs are bitwise equal for tested valid frames. Preserve the legacy G1 navigation API.

## 0.3.21 — 2026-10-30

- Add official MotionBricks SOMA/G1 ONNX pose-inbetweening pipelines, pinned model preparation, batched validation, boundary trimming and G1 quaternion conversion; retain legacy PyTorch navigation.

## 0.3.20 — 2026-10-27

- Bound Fast SAM's paired-hand decoder workspace while retaining all body and hand features; report end-to-end inference time per frame and throughput in the Blender queue.

## 0.3.19 — 2026-10-27

- Select a pinned CUDA 12.8 / PyTorch 2.7 runtime when Blackwell GPUs are detected, retain CUDA 12.4 for older GPUs, and probe GPU kernels before downloading weights.
- Add simulated RTX 50, mixed-device and Windows installer tests; physical Windows and consumer-GPU validation remains outstanding.

## 0.3.18 — 2026-10-27

- Distribute one FlowHMR Blender extension with runtime-selectable standard and lightweight modes; add Windows installer paths and process handling for evaluation.
- Pin Windows installer and FFmpeg executable hashes. Windows and consumer NVIDIA GPU support require physical validation.

## 0.3.17 — 2026-10-27

- Bundle the FlowHMR backend with the Blender extension and add an isolated one-click Linux NVIDIA runtime installer with dependency/model download progress and environment checks.

## 0.3.16 — 2026-10-27

- Add explicit mixed-FP16 SAM/VGGT inference across the unified pipeline and Blender worker, with checkpoint range validation and precision-specific camera caches.
- Keep upstream precision as the default; clarify that the Blender desktop workflow uses external Python and does not require Docker.

## 0.3.15 — 2026-10-27

- Give standard inference the same staged resources, global streamed VGGT, attention scheduling and frame batching while retaining official SAM CPU crops and separate hand decoders.
- Expose upstream-default and FP32 frontend arithmetic; record the selected policy and prevent cross-precision frontend cache reuse.
- On the 750-frame H200 test, standard inference measured 108.64 seconds and 5.11 GiB process driver peak; report numerical differences against the previous implementation.
- Add a transferable NVIDIA runtime build context and an isolated container worker adapter with job-scoped cancellation and external licensed assets.


## 0.3.14 — 2026-10-27

- Blender extension 0.2.8 adds a durable SQLite task queue, an independent serial runner, pause/reorder/cancel/retry controls, persistent attempt history, and explicit result import.
- Check CUDA execution, original BF16 support, available GPU/system memory, video size, disk space and FFmpeg before neural inference; select bounded workspaces from available memory.
- Measure the supplied complete video in standard mode at 366.81 seconds / 49.78 GiB driver peak on H200, and verify real CUDA OOM recovery from 8-frame to 4-frame batching across all 750 frames.
- Expose VGGT frame/attention/FFN batching through the unified pipeline and retain every frame and global attention key. Recover CUDA OOM in fresh worker processes with smaller workspace schedules and unchanged model precision and sampling settings.


## 0.3.13 — 2026-10-27

- Simplify the Blender release film to operation labels only; remove slogans, timing explanations, performance cards and closing links.
- Keep the result showcase and actual workflow footage, and align subtitles and release documentation with the revised cut.

## 0.3.12 — 2026-10-27

- Add a Blender product release film with fresh full-video GPU inference, explicit waiting cuts, complete original-speed playback and reversible grounding demonstrations.
- Add optical checks of both source and viewport frame counters, all-frame character framing validation, and an auditable screen-recording edit manifest.
- Add an illustrated first-use guide and align the extension documentation with current controls and measured performance boundaries.
- Preserve the 0.3.11 neural inference behavior and Blender extension 0.2.7 runtime.

## 0.3.11 — 2026-10-26

- Remove unused final SAM mesh predictions, mesh projection and topology readbacks from Fast conditioning while retaining every intermediate keypoint update and all per-frame body/hand tokens.
- Cache frozen dummy prompts, initial-pose embeddings and device ray grids in the pipeline-owned extractor; retain an explicit original-path control and unchanged general SAM outputs.
- Blender extension 0.2.7 shares scene evaluation across independent characters during grounding, retaining full deformed-mesh sampling and verification, with conservative serial fallback and transactional action restoration.
- Record complete-video compute-only latency, numerical differences, memory and rejected acceleration experiments; the measured backend still exceeds 60 seconds.

## 0.3.10 — 2026-10-26

- Profile complete VGGT camera execution and retain the bounded frame/query schedule after larger batches show little benefit and exceed the memory budget.
- Parallelize independent image preparation without changing pixels/order and release AMP cast-weight caches at GPU module ownership boundaries.
- Add actual-input selection between installed PyTorch and cuDNN exact SDPA kernels, with safe per-operation automatic fallback before residual mutation; retain all-frame key visibility and original model dtypes.
- Record selected camera backend and calibration/fallback details; expose `vggt_attention_backend=auto|torch|cudnn` and distinguish camera caches by requested policy.
- Blender extension 0.2.6 selects the automatic camera policy in accelerated mode and checks backend capability.

## 0.3.9 — 2026-10-26

- Connect per-frame tracking/SAM and per-batch/per-layer VGGT progress to Blender, with measured per-phase rate and ETA, throttled atomic status updates and the existing cancellation flow.
- Batch independent Fast SAM video frames while retaining each frame's camera, selected box, body token and both hand tokens; lightweight mode defaults to eight frames, with a one-frame fallback setting.
- Add bounded GPU batch-size profiles and full-video accuracy/runtime/memory validation, preserving camera interval=1 and the original network dtypes.
- Blender extension 0.2.5 exposes the feature frame batch size and checks backend support before inference.

## 0.3.8 — 2026-10-26

- Stage VGGT camera modules through CPU, microbatch independent frame encoders and frame attention, and bound global attention/FFN workspaces without discarding frames or keys.
- Retain only camera/register prefixes for the camera head and avoid redundant AMP key/value copies.
- Add explicit `vggt_memory_mode=auto|reference|streamed`; lightweight mode and Blender extension 0.2.4 select streamed evaluation.
- Add full-video allocator-budget measurements and numerical comparisons against the original camera and motion outputs.

## 0.3.7 — 2026-10-26

- Integrate Fast SAM GPU body/hand preprocessing and one merged left/right hand network pass into the native FlowHMR pipeline, preserving per-frame full-hand features and original hand boxes.
- Match OpenCV's uint8 interpolation/rounding on GPU; retain original SAM networks, weights, dtypes and intermediate predictions.
- Add explicit `sam_backend=auto|official|fast` selection, independent upstream full-video validation and per-frame token/motion comparisons. VGGT still observes every frame by default.
- Blender extension 0.2.3 selects the Fast frontend in lightweight mode and checks backend support before starting a job.

## 0.3.6 — 2026-10-26

- Avoid disposable random initialization when loading the fully checkpointed FlowHMR, VGGT and SAM DINO inference networks, while retaining ordinary training construction and checkpoint dtypes.
- Extract camera outputs without unused VGGT depth reconstruction or intermediate features; reuse SAM CUDA allocations and skip unused post-decoder hand processing after conditioning capture.
- Add synchronized inference stage timings and controlled GPU latency/output comparisons. Blender extension 0.2.2 records model-loading time, stage timings, cache use and CUDA memory per job.

## 0.3.5 — 2026-10-26

- Extend subject and raw reconstruction review in the Blender demo and document measured speed/memory with hardware, cache and benchmark limits.


## 0.3.4 — 2026-10-26

- Recut the Blender promotional demo to lead with results, show focused UI operations and keep all character performance shots unobscured by the sidebar.


## 0.3.3 — 2026-10-26

- Add a complete Blender workflow promotional demo with explicit wait cuts and the full verified 750-frame original-speed playback.


## 0.3.2 — 2026-10-23

- Prepare a viewport cache before original-rate animation playback instead of slowing the animation to preserve frames.
- Reuse existing subject markers safely when reviewing a detected person again.

## 0.3.1 — 2026-10-23

- Add an explicit person picker with observed bounding boxes, overlapping-box selection,
  trajectory review, source identity validation and guarded frontend cache reuse.
- Add real inference-stage progress, character preparation during inference, batch
  character retargeting/grounding, and full-range playback without frame skipping.
- Simplify the native Blender workflow and support Mixamo FBX and glTF characters.

- Add Blender contact grounding with planted-foot IK, swing-foot collision checks,
  preserved jump clearance and switchable Raw / Grounded actions.
- Preview the input video alongside actual FlowHMR surfaces in Blender and allow
  hash-verified frontend cache reuse while rerunning neural motion sampling.

- Add the FlowHMR Blender extension with standard/accelerated packages, external
  inference, full SMPL-H hand animation, Mixamo retargeting and animated FBX export.
- Preserve Mixamo FBX object rotations and centimetre units when transferring
  world-space rotations and root travel onto the target armature.
- Follow the selected device throughout FlowHMR SAM preprocessing and avoid a
  circular import when reconstructing SMPL-H meshes independently of inference.

## 0.3.0 — 2026-10-22

- Replace copied method applications with neural ModelBundles, native Trainers,
  inference Pipelines and separately owned geometry, conditions and solvers.
- Add scoped original-source comparisons and current-checkpoint cold-loading
  checks; record verified behavior and outstanding variants per method.
- Fix deterministic training seeds and stochastic full-resume behavior; add
  native joint TM2D training and complete DART primitive checkpoints and EMA.
- Preserve original evaluator statistic precision and batch padding, package
  runtime assets and source notices, and audit installed wheel contents.

## 0.2.0 — 2026-10-05

- Replace retired HYMotionV2M with the official FlowHMR 491-D, 0.46B release.
- Add native base flow-matching training, official data preprocessing, full-state
  resume and strict base/latest inference checkpoint loading.
- Bundle the official YOLOX/VGGT-Omega/SAM frontend and preserve component licenses.
- Add Hugging Face inference artifacts, standardized Model Card and source-parity
  checks covering losses, gradients, sampling, camera and video preprocessing.
- Export native SMPL-H parameters and shared world-space monocular capture results.

## 0.1.0

- Existing Motius framework and method integrations.
