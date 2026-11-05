# FlowHMR inference latency — 2026-10-26

Motius 0.3.6 removes unused initialization and reconstruction work without
reducing frames, sampling steps, camera observations or hand conditioning.
Blender extension 0.2.2 uses the updated backend and saves per-job profiles.

## Controlled measurements

The input is the user's unmodified 1280×720, 30 fps, 750-frame video. Its SHA-256
matches `assets/model_zoo/flowhmr/demo_input.mp4`. Cases use actual consecutive
video frames, live detection, VGGT and SAM, with no precomputed features.

One H200, Torch 2.6.0a0+ecf3bae40a.nv25.01, four CPU threads, FP32 motion sampling
with TF32 disabled, released frontend precision, 20 Euler steps, seed 0, CFG 1,
all camera frames and no motion postprocessing. Every trial starts a new Python
process and reloads every model; files and CUDA disk caches are already warm.
An idle Web service keeps approximately 1.5 GiB of GPU memory; it was left
running. Benchmark guards stop only the benchmark if competing work appears.
These are available-container measurements, not consumer-card measurements.

“Prior” restores eager model initialization, full VGGT depth/intermediate
outputs, per-image SAM allocator flushing and the prior lightweight hand path.
This is a controlled prior native execution path, not a fresh direct execution
of the entire upstream repository.

| Actual input | Prior lightweight | Optimized lightweight | Time reduction | PyTorch allocated peak, prior → optimized |
| --- | ---: | ---: | ---: | ---: |
| First 30 frames | 43.77 s | 25.73 s | 41.2% | 7.23 → 6.83 GiB |
| First 120 frames | 80.56 s | 60.17 s | 25.3% | 11.07 → 9.44 GiB |
| Complete 750 frames | Not measured | 365.89 s | Not established | Not measured → 27.80 GiB |

The 30-frame numbers are medians of two trials per variant in A–B–B–A order;
120 frames has one trial per variant. Timings run from pipeline construction
through decoded joint output and include frontend teardown, not just network
forward time. Including Python/Torch/module startup, the 30-frame medians are
46.14 → 28.19 seconds. Downloads are excluded.

The initial exploratory prior trial took 71.82 seconds with cold file/kernel
caches. It is **not** the baseline used to claim the 41.2% improvement. The
historical 45.56/44.79-second standard/lightweight measurements in the model
page also belong to an earlier experiment and must not be mixed into this pair.

### Complete source video

All 750 frames completed with finite 750×52×6 local rotation output,
750×22×3 joint output, 750×3072 body/hand features and 750 camera extrinsics.
Detection/tracking took 9.43 seconds, camera extraction 69.78 seconds, SAM
features 278.91 seconds and motion sampling/decoding 1.63 seconds. Total backend
latency was 365.89 seconds (368.42 including process startup), approximately
14.6 times the 25-second source duration. Allocated peak was 27.80 GiB and
reserved peak was 47.29 GiB. This is **not a lightweight consumer-card result**;
there is no measured prior 750-frame control or upstream quality comparison
for this case. Do not infer a full-video speedup from the short-video pairs.

### Where the 30-frame time goes

| Stage | Prior median | Optimized median |
| --- | ---: | ---: |
| Motion pipeline/model loading | 4.46 s | 2.20 s |
| Detection and tracking | 1.41 s | 1.29 s |
| Camera, including VGGT loading | 13.63 s | 4.59 s |
| Features, including SAM loading | 21.13 s | 14.60 s |
| Motion sampling and decode | 0.86 s | 0.78 s |

Stage sums exclude some residency transfers and teardown that are included in
the total. Model-loading savings are not presented as whole-pipeline speedup.
The original lightweight path primarily reduced memory and long-sequence
attention costs. At 30 frames, frontend loading and framewise features dominate;
accelerating a subsecond sampler cannot remove tens of seconds of frontend work.
H200 alone does not explain the small historical difference between modes.

A separate current-standard 30-frame run took **26.50 s / 9.70 GiB**, versus
**25.73 s / 6.83 GiB** for current lightweight mode. It is one standard trial,
not a throughput average. Common loading/camera improvements benefit both
modes; lightweight mode mainly reduces residency/attention memory. The Blender
mode label now says **Lightweight / 低显存**; the `accelerated` archive filename
remains compatible. Neither the label nor this release implies a fixed speedup.

## Implemented changes

- Load fully supplied FlowHMR/VGGT networks and SAM's DINO backbone from meta
  tensors, avoiding random initialization of parameters immediately overwritten
  by checkpoints. Preserve eager target dtypes and strict key checking. Rebuild
  FlowHMR's nonpersistent RoPE tables; preserve the released DINO zero mask token.
  Other SAM/MHR metadata still follows ordinary construction.
- Keep only VGGT's final aggregator output for camera extraction. Skip unused
  dense depth and alignment heads. The full VGGT API remains available.
- Reuse CUDA allocator blocks between SAM frames. Stage teardown still releases
  frontends to keep lightweight residency bounded.
- In lightweight SAM, retain original body reconstruction and both original
  hand crops. Capture each hand's decoder token before post-decoder hand-box
  output, wrist unflipping/fusion and the final prompted-body pass. Intermediate
  pose/mesh predictions inside the decoders remain intact: subsequent layers
  depend on them. No hand tokens are
  filled in, averaged or omitted.
- Save synchronized stage timings and allocated/reserved CUDA peaks through
  `--profile`. The Blender worker writes `inference_profile.json` automatically
  and reports whether the preprocessing cache was used. CPU threads are bounded
  to four rather than forcing a single thread during GPU model construction.

## Output checks and limits

The 30/120-frame camera extrinsics match the prior path **bit for bit** in these
runs. Eager/meta network loading and the small real VGGT camera network pass
zero-tolerance CPU comparisons, including nonpersistent buffers and stored
FP16 weights being loaded at the original FP32 inference dtype.

The entire CUDA pipeline is **not bitwise identical**. Across the 30-frame
comparisons, the maximum rotation difference is 0.246 degrees and maximum
22-joint position displacement is 0.221 mm. Repeating the prior path itself
produces up to 0.162 degrees and 0.205 mm of difference. In the 120-frame pair,
maximum rotation difference is 0.168 degrees and joint displacement is 0.343 mm.
Camera tensors remain equal; SAM's BF16 features and subsequent sampling exhibit
small numerical variation. These observations are not accuracy guarantees for
arbitrary videos and do not establish bitwise upstream equivalence.

Peak values are **PyTorch allocation in this process**, excluding CUDA context,
other libraries and the resident service. They do not certify that an 8 GB card
can run these clips. VGGT memory still grows with sequence length. Neither
consumer-card latency nor a consumer VRAM budget has been measured here.

The remaining throughput bottleneck is framewise SAM feature extraction. At
120 frames it takes 44.56 of the optimized 60.17 seconds, including loading;
at 750 frames it accounts for 278.91 of 365.89 seconds. VGGT is the peak-memory
stage, even after dropping unused dense outputs.
This work improves latency but does not establish real-time inference. Frame
batching and a persistent frontend service are further optimization candidates;
they are not shipped or claimed by this report. Camera subsampling already
exists as an explicit option but changes conditioning and was not used here.

Blender mesh reconstruction/import, retargeting, grounding and viewport playback
preparation are excluded. A backend latency result must not be described as the
complete time until a Blender scene is ready.

## Reproduction and evidence

The [machine-readable record](flowhmr-latency-20261026.json) contains all per-run
stages, peaks, checkpoint hashes, pinned Hub revisions, source hashes and output
comparisons. Weights must already exist locally. Supply a JSON mapping the five
frontend paths (`preprocessor/sam/model.ckpt`, `preprocessor/sam/model_config.yaml`,
`preprocessor/sam/assets/mhr_model.pt`, `preprocessor/vggt_omega_1b_512.pt`, and
`preprocessor/yolox_l.pth`) to their local absolute paths:

```bash
python tools/run_flowhmr_latency_suite.py \
  --checkpoint /path/to/native-flowhmr-latest \
  --body-model /path/to/smplh/neutral/model.npz \
  --video assets/model_zoo/flowhmr/demo_input.mp4 \
  --assets /path/to/frontend_paths.json --frames 30 \
  --output outputs/validation/flowhmr/latency-reproduction
```

Run this on an idle GPU. The suite never stops another process. For an ordinary
single run use `tools/infer_flowhmr.py VIDEO --lightweight --profile`; its JSON
reports the runtime and memory of that user's environment.

Validation: 61 regression tests passed, one skipped; the updated Blender worker
profile and backend guard checks passed. Both 0.2.2 archives passed Blender 4.5.3
extension validation. All benchmark processes exited. The original idle service
remains running; no new GPU container was requested or stopped.
