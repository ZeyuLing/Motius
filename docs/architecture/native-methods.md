# Native method architecture migration

All 50 current neural packages pass the static architecture audit. Earlier
installed-wheel import results are recorded separately for the 49 packages
covered at the time. Copied application trees have been removed from the
model packages; this structural result does not by itself establish numerical
source parity. The method table below distinguishes original-source evidence,
original Motius migration evidence, and unverified variants or training protocols.

Full independent MoGenDiT acceptance remains **open** because the complete
original trainer/refiner source is unavailable. See
[source-availability.json](source-availability.json). Do not claim that every
method and every configuration has an independent, complete source certificate.

## Responsibility boundaries

- `motius/models/<method>/`: neural modules, their atomic forward calls, strict
  checkpoint import/export, and architectural configuration. No upstream
  application tree, datasets, renderers, training loops, losses, diffusion
  sampling loops, subprocess launchers, or import-path aliases.
- `motius/trainers/<method>/`: training graph, corruption/noise construction,
  objectives, and validation, executed by the existing runner and accelerator.
- `motius/pipelines/<method>/`: inference graph, conditioning, guidance,
  iterative generation, preprocessing, decoding, and output conversion.
- `motius/diffusion/`: shared or method-specific diffusion equations and solvers.
- `motius/motion/representation/`, `motion/body_models/`, and
  `motion/conditioning/`: codecs, geometry, skeletons, and condition construction.
- `motius/configuration/` and `motius/licenses/`: packaged defaults and preserved
  source notices. Runtime loading must also work from an installed wheel.

Moving an upstream application wholesale to a differently named directory does
not satisfy this contract. Source adapters must be replaced by native components;
unused demos, evaluators, training frameworks, and exporters must be removed.

## Current evidence

| Method | Structural work | Numerical evidence | Remaining acceptance work |
| --- | --- | --- | --- |
| FlowHMR, including YOLOX, SAM3DBody and VGGT frontends | Vendor tree removed; model, trainer, solver, codec, video preprocessing separated | Original repository and released Base/Latest checkpoints: exact sampled outputs, source losses/gradients, 30-frame real-video features, camera results, floor alignment and joints | Broader optional configurations and frontend training remain separate validation targets |
| InterGen | Network-only bundle; native trainer and inference pipeline; artifact-local stats | Original repository and released checkpoint: 30-frame B=1 and 60-frame B=2 outputs bitwise equal; native two-process training synchronization | Original training-objective/gradient comparison beyond the native training tests |
| InterMask | Network-only transformer/VQ; native token trainer; sampling/reaction and joint packing outside models | Original repository and released InterHuman checkpoint: three generation cases including Gumbel sampling, every-step logits, decoded motions, and reaction IDs bitwise equal; two-process step-unroll training | Inter-X artifact and VQ training stage |
| KIMODO | Neural bundle separated from sampling, CFG, representations, skeletons, constraints, assets and motion correction | Pinned original repository, released SOMA-RP weights and real LLM2Vec: text features and five inference cases (single, batch, multiprompt, postprocessing, root control) bitwise equal | Other skeleton/data checkpoints; source training is not released |
| Bailando | Sampling, token/root-velocity objectives and optimizer construction removed from networks; native supervised/root trainer | Pinned original repository and released weights: three music-feature cases, all token logits, codes, decoded motion, sequential root trajectory and VQ encoding bitwise equal | Raw-audio preprocessing parity, actor-critic reinforcement stage, original training-gradient comparison |
| MoMask | Neural base/residual transformers separated from masking, editing, CFG, sampling and token objectives; native trainer | Pinned original repository and released HumanML3D weights: 234 tensor comparisons, including sampling/editing, losses and all base/residual gradients, bitwise equal; two-process training synchronization | VQ training stage and complete dataset protocol |
| MaskControl | Neural regularizer separated from control geometry, iterative logits/embedding optimization and native training objective | Pinned original repository and released weights: ordinary and relative-control inference, random/cross objectives and gradients bitwise equal with deterministic cuDNN; native gradient-optimization test | Full dataset protocol, additional control/composition configurations |
| FlowMDM | Copied data/evaluation/training/visualization applications removed; neural bundle, diffusion, CFG, sequence sampler and native trainer separated | Pinned original source and released HumanML3D/BABEL checkpoints: full 1000-step single/multi-prompt outputs, objective and gradients bitwise equal (128 tensor comparisons per checkpoint); two-process native training | Complete dataset training protocol; broader optional attention/conditioning configurations |
| GVHMR | Vendor application removed; motion/vision neural bundles, native trainer, body/camera codec, tracking and video inference separated | Pinned original source and released checkpoint: 328 core inference/loss/gradient comparisons bitwise equal; real 30-frame YOLO, crops, ViTPose and HMR2 including decoder layers bitwise equal; full public static/dynamic video calls, camera/world parameters and materialized SMPL vertices/joints bitwise equal in fresh workers; native two-process neural training | Dataset integration and optional DPVO/rendering validation |
| MoGenTS | Network-only dual-stream VQ, auxiliary and spatial-temporal transformers; packing, fusion composition, sampling and token objectives separated | Pinned original source and released HumanML3D weights: 859 comparisons, including dual-stream generation, VQ encoding/reconstruction, four token-stage objectives and all gradients bitwise equal; native four-stage two-process test | VQ training stage and complete dataset protocol |
| MotionGPT | Removed upstream package scaffolding, global import aliases and constructor monkeypatch; neural T5/VQ bundle, task/text pipeline and native language/VQ trainer | Pinned original repository and released weights: generation-step logits, tokens, captions, VQ reconstruction, language/VQ losses, all gradients and updated EMA codebook bitwise equal; native two-process language training | Complete dataset protocol and distributed VQ codebook validation |
| MotionGPT3 | Removed the upstream application tree, fake data module, import aliases, runtime patches and unused neural modules; native modality network, VAE, diffusion head, sampler and trainer | Pinned original repository and released checkpoint: 710 bitwise comparisons covering full inference, two language/diffusion training cases, all gradients, and VAE objective/gradients; native two-process language/diffusion training | Complete dataset protocol, broader architecture variants and distributed VAE validation |
| UniMuMo | Neural bundle separated from token pattern, audio/motion codec composition, text conditioning, CFG sampling and caption generation; native token/caption trainer on encoded conditions | Pinned original repository and full released checkpoint: motion encoding, real T5 conditions, logits, greedy/stochastic tokens, audio/motion output, caption contexts and generated captions bitwise equal; source token/caption losses and all gradients bitwise equal (798 comparisons); native two-stage two-process training | Raw-data protocol and motion-codec training stage |
| GEM-SMPL | Native neural Bundle, video/sampling Pipeline, representation/body geometry and Trainer; vendor removed | Neural 606 and feature-to-body 66 comparisons exact; independent cold video/public result including meshes exact | Default weighted regression/diffusion and joint two-stage objectives plus all active gradients: 1,750 comparisons exact; repeated-forward DDP passed. Dataset/optimizer protocols pending |
| GEM-X | Native neural Bundle, frontend/sampling Pipeline, SOMA representation and Trainer; vendor removed | Neural 296, feature-to-body 68 and cold-video frontend 17 comparisons exact; complete public Auto Pipeline video results including joints/meshes exact under the same declared deterministic geometry policy in both workers | 19 original weighted regression/joint objective scalars exact; geometry backward varies in 260/265 parameter gradients even between two original-source workers, so bitwise gradients remain uncertified. Full dataset/optimizer protocol pending; inference certificates explicitly use the shared reproducible runtime |
| PromptHMR | Native image/prompt/temporal neural Bundle, separate frontend/DROID/Metric3D Bundles, in-process capture Pipeline and temporal Trainer; external runtime removed | Original source: 96 image/body/temporal, 21 frontend, 20 DROID neural, 58 Metric3D recurrent-output and full masked SLAM trajectory/BA comparisons exact; complete static video 181 comparisons exact; temporal objectives/all gradients 520 comparisons exact | Complete moving-camera video: 179 results bitwise exact under shared deterministic SLAM settings; current-weight export/all 986 tensors/cold Auto Pipeline video and two-process temporal DDP passed. Image training source and complete dataset protocol unavailable |
| MotionBricks | Five neural roots only; native pose/root/VQ Trainers, G1 representation/body assets and navigation Pipeline; copied application removed | Original source and released weights: 382 neural comparisons; two complete 160-frame public qpos rollouts; all original stage objectives, active gradients and VQ EMA updates exact; actual two-process neural-root training passed | Full dataset/optimizer protocol; original release publishes a synthetic dataset rather than its full training data loader |
| VerMo | Neural-only language/motion/audio bundle; preprocessing, normalization, vocabulary, weighted losses and boundary diagnostics in native owners; current export packages tokenizer/statistics and identifies the public Pipeline | Original Motius baseline: released-weight pre-migration motion prompts/logits/captions/objective/all gradients exact; native tied-head two-process training and modified-current language/motion networks through offline cold inference pass | Optional audio variant and full training protocol; this is an original Motius baseline rather than an external-paper reproduction |
| GenTrack | Network-only generator/reference bundle; policy sampling and objectives in Pipeline/Trainer; exports identify GenTrack and preserve its sampler settings | Policy/anchor/reward regressions and a current-weight, nondefault-sampler export through a fresh automatic loader pass with a real small MMDiT | Released-checkpoint end-to-end comparison and full simulator/training protocols |
| DART | Neural denoiser/VAE/CLIP Bundle; native seed/body/rollout Pipeline and primitive Trainers, AdamW/EMA, explicit latent calibration and complete paired-network checkpoints | Original full FlowMDM rollout with real licensed SMPL-H/SMPL-X: four DDPM/DDIM cases, world parameters/joints, public Motion135 and RNG exact. Both full body objectives/all gradients exact; original optimizer/EMA/calibration exact. Each stage passes two-GPU updates and public CLI resume versus uninterrupted training, including every neural/EMA/Adam tensor. Actual trained, calibrated EMA pair and current CLIP survive cold automatic export exactly | Original raw corpus, multi-primitive rollout curriculum, AMP, ordinary seed-inclusive rollout_mld and full benchmark rerun |
| ViMoGen | Removed copied package tree and unused model/retargeting utilities; pure motion/UMT5 neural modules, native flow scheduler, conditioning, sampler and trainer | Original repository and released checkpoint with real UMT5: full 50-step single/batch generation and source weighted flow objective/all gradients bitwise equal; source smoothing, dtype restoration and physical-scale export order preserved (1,417 comparisons); real Wan network two-GPU BF16 checkpointed training | Dataset integration and extended reference-corruption configurations |
| MoGenDiT | Vendor removed; network, diffusion, representation, corruption, geometric loss and refiner separated | Available original diffusion/representation files verified against the recorded source hashes. Full refiner preserves the pre-migration Motius implementation on three nonzero synthetic-weight cases; this does not certify the unavailable original network/trainer/refiner archive | Complete original network/refiner/trainer source and released-weight comparison; original archive currently unavailable |
| TMR | Encoders/decoder in neural Bundle; objective in native Trainer; sampling, dropout order and indexed contrastive filtering preserve source arithmetic; wrapped forward for DDP | Independent Mathux/TMR source with identical released Motius-trained evaluator weights: prepared embeddings/reconstruction, all weighted losses and all parameter gradients bitwise exact in eval/train and mean/stochastic VAE modes; two-process training passed | Complete raw-text preprocessing/dataset protocol; this checkpoint is Motius-trained, not an original TMR release |
| MDM, CondMDI, OmniControl | Neural bundles; native prepared-motion Trainers; diffusion/CFG/text processing outside networks | Pinned original repositories and released neural weights: two ragged minibatch objectives, all parameter-gradient hashes and NumPy/CUDA RNG exact; complete 1000-step guided/unguided sampling and control outputs bitwise equal in fresh processes. CondMDI preserves original absolute-root representation for this certificate. All three modified denoiser/CLIP exports reproduce full sampling through offline cold AutoPipeline loading; three shared-Runner two-GPU updates have identical replica hashes, frozen CLIP and full checkpoints; public training CLI/data reader run two updates then resume to three | Original corpus/observation selection and optimizer/AMP training protocol |
| EDGE | Decoder-only neural network; original four-term Trainer, native Adan/EMA and separate cosine DDIM/CFG/stitching/FK owners | Author original checkpoint: all 249 EMA states and both scaler tensors exact. Full 50-step one/two-window inference, axis-angle/quaternion stitching/FK and three objective/gradient/Adan/EMA updates bitwise equal to original source under pinned PyTorch3D v0.7.2 FP32/math. Three native two-GPU updates synchronize online and EMA states; public CLI two updates then full resume to three; actual resumed current EMA exports/cold public inference exactly | Raw audio/Jukebox extraction, complete original corpus/epoch/validation protocol and FP16 |
| MotionLCM | Network-only VAE/text/teacher/online/target roots; native consistency distillation Trainer and few-step inference Pipeline | Original V1 teacher archive provenance; full source inference 1/2/4 steps exact; three Source Huber/gradient/clip/AdamW/target-EMA updates exact including RNG; three shared-Runner two-GPU updates synchronize online and target hashes; public CLI two updates then full resume to three | ControlNet, full raw corpus/validation/scheduler protocol and benchmark rerun |
| MLD | Native VAE and latent-diffusion Trainers; original CLIP text-only neural tower and explicit legacy SentenceT5 artifact metadata | Author original CLIP checkpoint/code: full 50-step inference, both stage objectives/all gradient hashes/CPU+CUDA+NumPy RNG exact; modified-current full inference survives offline loading; each stage passes three shared-Runner two-GPU updates and public prepared-data CLI two updates then full resume to three | Full original corpus, optimizer schedule and benchmark rerun; published T5 artifact is the MotionLCM V1 teacher, independently matched across all 125 denoiser tensors |
| MotionCLR | Native masked reconstruction Trainer; raw captions and fused CFG outside the neural network; original float64 statistics preserved | Original repository and hash-verified 6.7 GB EMA release: all neural states, both ragged minibatch losses, all gradients and CPU/CUDA RNG bitwise equal; complete 10-step DPM and 50-step DDIM normalized/physical outputs exact in controlled FP32; modified current denoiser/CLIP and float64 statistics survive full offline cold inference exactly; three shared-Runner two-GPU updates synchronize all 963 neural states and preserve frozen CLIP; public prepared-data CLI runs two updates and fully resumes to three | FP16 and full original EMA/data/optimizer protocol |
| ProjFlow | Network-only ACMDM prior; flow objective in native Trainer; CFG, iterative sampler and projection in Pipeline; statistics in motion assets | Original repository, its pinned timm 1.0.9 and released weights versus native timm 1.0.26: two ragged training objectives, all gradient hashes and CPU/CUDA RNG exact; two full 100-step guided/unguided projection/control cases bitwise equal; modified current denoiser/CLIP offline cold export also exact; three native shared-Runner two-GPU updates synchronize every neural hash and preserve frozen CLIP; public prepared-data CLI/resume also passed | Complete raw corpus/EMA/optimizer protocol |
| TM2D | Five neural roots only; native joint Trainer shares the source audio decoder; original token rejection sampling and window composition in Pipeline | Author source/pinned released weights: three text and three music cases, physical motion/joints/VQ, both weighted token objectives, all gradients and RNG exact. Three native two-GPU Adam updates; public full resume equals uninterrupted training in all 1,090 neural/optimizer tensors; trained/current export and cold text/music inference exact | Raw music extraction, joint bimodal fusion, VQ training stage and full original corpus/epoch protocol |
| PRISM | Neural transformer/VAE/UMT5 Bundle; backend, scheduler and processor in Pipeline/Trainer; self-describing current exports preserve runtime defaults | Original Motius method: actual released PRISM-KT/UMT5 and all three public 50-step tasks match pre-migration code exactly (101 comparisons). Both real-network mode/stochastic training objectives, all gradients and RNG exact; three native two-GPU updates synchronize all 825 transformer states and preserve frozen roots. Modified-current transformer/VAE/UMT5 cold export passes all three tasks (100 comparisons) | Full corpus/optimizer/AMP protocol and PRISM 1.0 checkpoint variant |
| HumanML3D matching evaluator | Three neural roots in Bundle; raw text, original-precision normalization, sorting and padding in Pipeline | Current author release: all 42 neural tensors/statistics/GloVe bytes exact. Actual original raw-text/dataset/wrapper versus native public single/ragged Float32/Float64 inference exact with explicit inverse ordering; modified-current neural/statistic/config/vocabulary cold exports pass both safetensors and PyTorch | Full dataset crop/caption-selection protocol and historical benchmark revalidation |
| MotionStreamer matching evaluator | Two neural encoders only; original Float64 normalization and batch-length padding in Pipeline; current export preserves the actual attention implementation | Author original epoch99 checkpoint and untouched dataset/collate/encoders: all neural hashes, raw-caption and single/ragged Float32/Float64 motion embeddings exact. Modified current encoders/BERT, sub-Float32 statistics and nondefault preprocessing survive offline cold loading exactly | Complete benchmark crop/caption-selection and aggregate metric rerun |
| BeyondMimic | Neural actor/critic and running-normalization Bundle; native PPO Trainer, online AccelerateRunner, external reference data, inference/ONNX Pipeline and native Isaac Lab terms; copied application removed | Independent original RSL-RL: 252 PPO and 550 complete online-loop comparisons exact; original CSV/deployment/export comparisons exact; native iteration/checkpoint and regression checks passed. Uses original seeded networks, not released pretrained G1 weights | Isaac Sim 4.5 / Isaac Lab 2.1 installed on the existing H200 container; real GPU rigid-body test and original/native 99-frame G1 FK/velocity conversion exact. Dynamic original/native tracking exact in 4 real G1 environments for 60 steps, including observations, actions, rewards, terminations, robot states and RNG. Same original seeded policy, not pretrained tracking quality; real-environment PPO integration also exact for two 24x4 rollouts and 40 minibatch updates, including gradients, Adam and checkpoint writing; full motion training protocol pending |
| ProtoMotions | Native G1 actor, task critic, AMP discriminator and discriminator critic; observation encoding and PD conversion outside neural models; native PPO/AMP Trainer component added | Released source-pinned G1 weights: 392 network/output/statistic/gradient-probe comparisons exact; complete original raw-input deployment composition matches native Pipeline (8 outputs across B=1/B=3); modified current-weight export and cold Auto Pipeline reload exact | Original PPO/L2C2, both clipped value objectives and AMP losses/active gradients/normalization/Adam updates exact on three synthetic minibatches, including ordinary nonzero discriminator-value updates and the source excessive-gradient guard. Original AMP.fit and native AccelerateRunner also match for two 32x4 tensor-environment rollouts, including normalizers, EMA, replay and all optimizer states; both default actor skip behavior and a forced-all-updates coverage case passed. Data components have an independent 77-comparison certificate; full two-GPU original Fabric versus native Accelerate DDP also matches (2,684 comparisons), with different rank-local environments and all minibatch updates. Native G1 environment, FK/motion reader, real reference expert sampling and public CLI now replace the legacy application tree, which has been removed. Four real G1 environments pass two 32x4 rollout/update iterations with 1,670 exact neural/optimizer/physical/RNG comparisons. Actual original evaluator versus native Pipeline also matches 1,860 functional tensors over 60 physical steps; contact-force diagnostics use the separately measured original-self-repeat absolute bound (3.052e-5 N), not a bitwise claim. Assets are separately hash-pinned and materialized outside the model package. Synthetic short reference, controlled FP32/eager runtime; long training, full real motion datasets and original optimizer/progress resume remain unverified |
| SONIC | Native modality encoders, FSQ, action/reconstruction decoders and value network; deployment packing outside neural roots; PPO/auxiliary Trainer component added | All 72 public checkpoint tensors and 400 original network comparisons exact. Default public deployment Torch composition matches native Pipeline and modified-weight cold automatic loading; public ONNX tokens exact in all three modes, CPU ONNX/CUDA Torch decoder error below 1.2e-6 | Actual original PPO/KL adaptation and all five auxiliary objectives, gradients, normalizers, RNG and controlled AdamW updates exact (1,034 comparisons). Native online collector/GAE/PPO, shared Runner and selected G1 physics/data now replace the removed 144-file vendor application. Original online protocol: exact losses, optimizer/RMS/RNG state and adaptive synchronization boundaries. Real original environment versus public Pipeline: 241 exact observation/action/state/reward/RNG comparisons over four G1 robots and 20 control steps spanning all three modalities; source-repeat contact bound is zero. Native CLI real physical updates, full resume and cold current-checkpoint export passed. Two-GPU native Runner: 12 updates with all 72 neural/buffer hashes synchronized. Short synthetic paired reference and plane terrain; full real-data convergence and rough-terrain validation pending |
| HY-Motion T2M | Motion/text NN and learned conditions in Bundle; condition construction, seeded sampling, smoothing and wooden geometry outside models | Complete official Lite checkpoint, actual original raw-caption frontend and original full sampler/decoder: four 50-step cases exact, including 52 joints, special conditions and CPU/GPU noise. Native two-GPU training, public CLI/resume and trained/current full text cold export pass | Full original training is unpublished; full corpus/benchmark revalidation and other checkpoint variants remain outside this certificate |
| AnimateCanvas | Task construction, padding and condition policy moved outside Bundle; native Trainer/Pipeline retained | Actual released 0.46B checkpoint versus preserved pre-migration Motius implementation: all four full 50-step public tasks and condition gradients exact. Two-GPU training, public CLI/resume and trained/current full text cold export pass | Own-code migration evidence; full-corpus convergence and sparse-rollout training validation remain separate |
| Pre-existing native deployment methods and optional variants | Neural-package boundaries and installed-wheel imports audited; existing task APIs retained | See method-specific reports; package imports are not a numerical certificate | Independently recertify additional checkpoints/configurations when those variants are selected |

## Verification

Source comparisons must use independent workers loading an immutable original
checkout and the same checkpoint, inputs, seeds, dtype, device and dependency
versions. Compare intermediate tensors and final outputs at `rtol=0, atol=0`.
A reference that calls the migrated implementation cannot certify source parity.
Synthetic weights and pre-migration comparisons are explicitly labeled; they do
not certify released pretrained results. Reports must start with `complete=false`
and become complete only after every requested case passes.

Current reports are in [validation/](validation/); full tensors and logs are under
`outputs/validation/native_method_architecture/`. Reports record their tested
source/checkpoint/code hashes and are snapshots, not automatic certification of
later edits.

```bash
python tools/audit_method_architecture.py --output docs/architecture/method-audit.json
python tools/audit_method_architecture.py --method flowhmr --method intergen --method intermask --check
python tools/verify_flowhmr_source_parity.py --help
python tools/verify_flowhmr_frontend_parity.py --help
python tools/verify_intergen_source_parity.py --help
python tools/verify_intermask_source_parity.py --help
```

GPU comparisons in this migration use the already allocated Koala container
`zeyuling-g2-depth11-eval60k-4g-1020` (task
`b13dfef5-7470-463a-a4c9-2328975c30af`). No additional GPU container is requested.

## Data processing follow-up

A semantic review found additional preprocessing and objectives inside otherwise
clean model trees. HYMotion/MotionCanvas objectives now belong to their Trainers;
normalization, body decoding, temporal filtering and shared geometry live under
`motion/representation/hymotion_runtime`. Identical geometry/body implementations
were consolidated. Their artifact/G1/inference/GenTrack regression suite passed.

Fifteen further Bundle-level dataset-statistic transforms moved into shared
representation functions, preserving operation order and dtype behavior.
`motion-statistics-migration.json` records 122 exact pre-migration comparisons;
these are migration regressions, not independent original-repository certificates.
TM2D's VQ atoms now accept/return prepared normalized tensors, with raw motion
conversion in the Pipeline. MotionCLIP tokenization, raw motion preparation and
contrastive/distributed objectives are separate from its neural Bundle;
`motionclip-migration.json` records 295 exact synthetic snapshot NN/output/loss/
gradient comparisons. Independent released-weight validation remains pending.
ARDY still contains representation/runtime code that requires migration. Audit
patterns detect known leaks and do not by themselves certify network-only design.


ARDY's native Core-H40 and G1-H52 network and full generation comparison now
passes 868 exact checks against pinned original source and published weights
(`ardy-source-parity.json`). The Bundle holds only the actual root/body neural
core and encoder/quantizer/decoder. The Pipeline owns the two-stage geometric
composition, CFG, DDIM and history. The comparison includes two-window ordinary
and ragged constrained generation using prepared text features. Original raw
text encoding, public output materialization, modified-weight cold export and
training support remain under validation.

ARDY's actual 8B Llama-3/LLM2Vec frontend, MNTP/supervised adapters and full
raw-text generation also match original source exactly (874 comparisons,
`ardy-text-source-parity.json`). The local reference layout includes actual base
model files beside the MNTP adapter because original Transformers 5.8.1 treats
that directory as a complete embedded-adapter model. No original neural class
or inference function was patched. The native frontend now preserves the
original canonical model name used to format prompts. Current neural weights
from both denoiser stages and the tokenizer were independently changed and
cold reloaded through the public Auto Pipeline: 877 comparisons passed
(`ardy-checkpoint.json`), including materialized geometry and G1 qpos.

### Additional text and decoding separation

[Independent source records](clip-text-migration.json) cover raw-text and neural
interfaces for InterGen, InterMask, MoMask, MoGenTS, MaskControl, FlowMDM and
KIMODO. Their reports state the numerical scope, including where training or
data protocols remain outside the comparison. The
[HY-Motion real text record](hymotion-text-source.json) additionally checks
Qwen3-8B/CLIP-L after correcting the previous prompt, padding and precision
policies. MotionCanvas keeps its own existing prompt and padding metadata.

[MotionLCM](motionlcm-source.json) passes the original released text-only
inference graph at 1/2/4 steps, including neural tensor hashes and random state.
[KIMODO current-weight export](kimodo-checkpoint.json) passes modified denoiser
and 8B text-network hashes and complete generation through offline cold loading.
These checks do not establish complete training protocols or full benchmark
revalidation. Static audit coverage continues to expand as additional semantic
leaks are found; a clean static report is not a completion certificate.

[T2M-GPT](t2mgpt-source.json) additionally matches original released-weight
inference and both corrupted-token and VQ objectives, all active gradient hashes,
EMA codebook state and RNG under a controlled deterministic convolution policy.
The Trainer preserves the source's fixed positional parameters when switching
stages. Its public inference now follows the original evaluator's eval mode.

[TM2T](tm2t-source.json) matches the original VQ/caption networks, beam decoder,
summed caption CE, label smoothing and all gradient hashes. Raw motion processing
keeps the original NumPy statistics precision before converting normalized
features to FP32. Full training datasets and the tokenizer/GAN training stage
remain outside that certificate.

[MotionMillion](motionmillion-source.json) now matches the independent original
real 7B/T5/FSQ release for complete default generation, AR corruption and CE,
VQ reconstruction/velocity/acceleration objectives, all neural and gradient
hashes, and RNG. The default generation length and uncached decoding follow the
original evaluator. Full data and optimizer/distributed protocols remain pending.

[SONIC online training](validation/sonic-online-source.json) matches the original
TRL collector, timeout bootstrap, GAE, two PPO epochs, adaptive LR, global
clipping, AdamW, normalizers and RNG over two six-step rollouts in four controlled
tensor environments (890 exact comparisons). This extends the neural/objective
certificates; it does not establish physical-environment or distributed parity.
The native collector and online Trainer compose through `build_sonic_runner`.
The vendor application has been removed. [Physical Source comparison](validation/sonic-physical-source.json),
[two-GPU Runner](validation/sonic-distributed.json) and
[real CLI resume/current export](validation/sonic-cli-checkpoint.json) provide
separate, scoped acceptance evidence.
