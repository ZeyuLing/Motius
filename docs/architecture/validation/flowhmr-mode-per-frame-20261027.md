# FlowHMR per-frame performance and GPU memory

On 2026-10-27, the user's full 750-frame / 30 fps video was processed on one NVIDIA H200 with four CPU threads, PyTorch 2.6.0 / CUDA 12.4, source frontend precision, seed 0, 20 sampling steps, the same previously selected track, no preprocessing cache, body frame batch 8, camera frame batch 8, and camera interval 1. The complete 52-joint body and hand outputs were retained. Both runs selected the PyTorch attention backend. Each row is a single run in a fresh process; the unchanged Standard path and old Lightweight path are from 0.2.13, while the final Lightweight run uses the 0.2.14 frontend changes in the same backend environment.

| Mode | Average inference | Throughput | Driver peak | PyTorch allocated peak |
| --- | ---: | ---: | ---: | ---: |
| Standard | 166.47 ms/frame | 6.01 frames/s | 5.10 GiB | 3.90 GiB |
| Lightweight before workspace changes | 149.02 ms/frame | 6.71 frames/s | 6.18 GiB | 4.48 GiB |
| Lightweight with paired-hand chunks and streamed crops | 155.49 ms/frame | 6.43 frames/s | 5.05 GiB | 3.87 GiB |

Inference time includes model loading, camera, body/hand extraction and sampling; it excludes separately running person detection, Blender import, retargeting, rendering and user actions. Milliseconds per frame are the full inference duration divided by the processed frame count; they amortize model loading and are not per-frame latency measured independently. Global camera attention is sequence-dependent, so multiplying this average by an arbitrary number of frames is only a rough estimate.

Driver peaks are sampled from the worker process once per second and can miss brief spikes. PyTorch allocated peaks are measured by the allocator, not the whole process. Other services remained running; a guard stopped the experiment whenever they became active. The final Lightweight and Standard peaks are close, so a consistent memory advantage on other hardware or videos is not established. The 0.2.13 Lightweight and Standard comparison in the former documentation mixed different code versions and is superseded by this same-environment measurement.

The final Lightweight implementation retains every frame and both hand tokens. It processes both hands together in frame groups of at most four and releases crops and body outputs earlier; the body stage still batches eight frames. Compared with Standard, matching frame IDs produced world-joint distances averaging 0.721 mm (95th percentile 1.150 mm, maximum 1.333 mm) and hand-rotation differences averaging 0.012° (95th percentile 0.042°, maximum 0.279°). Against the previous 0.2.13 Lightweight output, the world-joint mean / maximum difference was 0.550 / 1.076 mm and hand-rotation mean / maximum was 0.014° / 0.346°. These are output differences, not ground-truth errors or evidence of bitwise equivalence.

The [machine-readable summary](flowhmr-mode-per-frame-20261027.json) contains timings, schedules, stage and memory values, and the full numerical comparisons. H200 measurements do not establish Windows or consumer-GPU performance.
