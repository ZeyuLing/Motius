# FlowHMR for Blender

Turn a video into a 52-joint body and hand capture, then retarget and ground it on Mixamo characters inside Blender. Standard and Lightweight inference are two choices in **one extension**.

![Three retargeted characters alongside the input video in Blender](assets/cast-final.jpg)

[Watch the complete Blender workflow](assets/demo.mp4) · [Download the extension ZIP](https://github.com/ZeyuLing/Motius/releases/download/flowhmr-blender-v0.2.14/motius-flowhmr-0.2.14.zip) · [Release notes and checksums](https://github.com/ZeyuLing/Motius/releases/tag/flowhmr-blender-v0.2.14)

The video shows the capture result followed by selecting the video and person, inference, retargeting and grounding. It was recorded with an earlier interface; the labels and options below describe the current extension.

## Requirements

- Blender 4.2 or newer and a 64-bit Linux or Windows machine with an NVIDIA CUDA GPU and a compatible driver. The full workflow has been tested on **Linux / H200**. Windows and RTX 50 support have code-level tests only; Windows and consumer GPUs still need physical end-to-end validation. macOS and AMD GPUs are not supported by this CUDA backend.
- Internet access for the initial one-click setup and enough free disk space for Python, CUDA packages, approximately **9.66 GB of model files**, and caches. FFmpeg and FFprobe are installed and verified if missing. The installer detects GPU compute capability using `nvidia-smi` and selects PyTorch 2.7.1 / CUDA 12.8 for Blackwell (including RTX 50) or PyTorch 2.6.0 / CUDA 12.4 for older supported GPUs. It checks CUDA and TorchVision kernels before downloading model weights. RTX 50 requires a compatible recent NVIDIA driver (NVIDIA lists 570.65 for CUDA 12.8 on Windows). Mixed Blackwell and pre-Turing GPU systems are rejected by automatic setup; use a compatible external environment for the intended GPU.
- An authorized **neutral SMPL-H `.npz`** model with 52 joints and 16 shape coefficients. This model is not included in the ZIP or downloaded automatically. Obtain it under its original license; male or female models are not substitutes.
- One or more editable standard Mixamo T-pose characters for retargeting. The extension does not supply character meshes or rigging.

## Installation

1. Download the ZIP above. In Blender, open **Edit → Preferences → Get Extensions → Install from Disk…**, select the ZIP and enable the extension.
2. Enable Blender's online access. Open the 3D Viewport sidebar (**N → FlowHMR**) and select your neutral SMPL-H model.
3. Click **Install / update runtime**. The extension installs a separate Python environment, CUDA PyTorch, dependencies and pinned model weights. Progress shows the current stage, package and actual weight-download bytes. Blender remains usable; reopening it can reconnect to an ongoing install.
4. Click **Check environment**. GPU architecture, CUDA and TorchVision kernels, model files, SMPL-H and video tools are checked before inference. Keep the NVIDIA driver up to date. The installer does not install packages into Blender's own Python.

The extension ZIP includes the exact Motius backend source snapshot in `backend.tar.gz`; it does not contain model weights or the licensed body model. The runtime is stored under `~/.cache/motius/flowhmr-runtime` by default. To use a different folder or your own Python environment, open **Options → Backend setup**; save Blender preferences after changing the runtime folder. An interrupted model download can reuse its cache.

## Capture and animate

1. Select a video under **Your performance** and click **Preview**, then **Detect people**. Click **Choose person**, select a detected box and check the track before confirming. In overlapping boxes, use Tab or click again to cycle candidates.
2. Select **Standard** or **Lightweight** in the same panel and click **Add capture to queue**. Both modes preserve per-frame body and hand features and process every camera frame at the default camera interval. Lightweight uses GPU preprocessing and batches both hands.
3. Continue editing or queue additional videos while inference runs. The queue displays real stages and progress; select **Import result** once the job finishes. Failed and interrupted jobs can be retried, and memory-related retries reduce working batch sizes rather than frames or selected precision.
4. Import your Mixamo character, select it, and use **Apply motion to character** followed by **Plant feet on the floor**. For multiple characters, use **Animate and ground all characters**. **Prepare & play at original speed** prepares a view for real-time playback.

![Person selection in the Blender viewport](assets/person-picker.jpg)

The tracker can lose identity through occlusion or crossings; inspect the selected track. Ground contact is an estimate and should be reviewed for the character and action. The extension supports standard editable Mixamo rigs; constrained control rigs and nonuniformly scaled rigs require preparation.

## Measured performance

A controlled H200 test used the supplied 750-frame video and the same selected-person track, PyTorch 2.6 / CUDA 12.4, upstream frontend precision, 8-frame body batches, all camera frames, full body and hand features, 20 sampling steps, and no preprocessing cache. Each run used a fresh process. The Standard and previous Lightweight runs use 0.2.13; the optimized Lightweight run uses the 0.2.14 frontend implementation in the same backend environment. Both modes selected the same `torch` attention backend.

| Mode | Average inference | Throughput | Driver peak | PyTorch allocated peak |
| --- | ---: | ---: | ---: | ---: |
| Standard | 166.5 ms/frame | 6.01 frames/s | 5.10 GiB | 3.90 GiB |
| Lightweight 0.2.14 | 155.5 ms/frame | 6.43 frames/s | 5.05 GiB | 3.87 GiB |
| Lightweight 0.2.13, before workspace changes | 149.0 ms/frame | 6.71 frames/s | 6.18 GiB | 4.48 GiB |

The new Lightweight path keeps body batches of eight, batches left and right hands together in groups of at most four frames, and releases temporary GPU crops sooner. It lowers the measured peak compared with the previous Lightweight implementation, at a throughput cost. Its memory advantage over Standard in this one test is small and is not a guarantee for every video or GPU: both share the same camera stage.

Milliseconds per frame divide measured inference time by processed frames and include model loading. They exclude the separate person-detection operation, Blender import, retargeting, and interaction. Driver peaks are sampled once per second and can miss brief spikes; PyTorch allocated peaks cover its allocator, not the entire process. These are single-run H200 measurements, not Windows or consumer-GPU benchmarks. Averages amortize startup costs; global camera attention, video length, resolution and hardware affect runtime and memory.

The optimized output retained all 750 frame IDs. Compared with Standard, world-joint differences averaged 0.721 mm (maximum 1.333 mm) and hand-rotation differences averaged 0.012° (maximum 0.279°). These are differences between outputs, not ground-truth accuracy or bitwise-equivalence claims. See the [measurement methodology and results](../architecture/validation/flowhmr-mode-per-frame-20261027.md) and [Windows / GPU compatibility checks](../architecture/validation/flowhmr-windows-compat-20261027.md).

## Licenses and attribution

The extension is GPL-3.0-or-later. FlowHMR weights, SMPL-H, tracking models and Mixamo characters have their own licenses; FlowHMR has education/research/nonprofit restrictions. See the extension's `NOTICE.md`. The workflow is informed by [Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig) and [FlowHMR issue #4](https://github.com/flowhmr/flowhmr/issues/4).
