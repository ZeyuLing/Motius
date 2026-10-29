# ARDY Attribution

The neural components, motion representation and inference composition are adapted from
[nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) at commit
`693f74d13b3d04a0a22ce127ee79c929dd89756b`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. ARDY is licensed under
Apache-2.0. Neural encoder/decoder/quantizer and two-stage root/body Transformer modules live under `motius/models/ardy/network/`. Skeleton, constraints, coordinate transforms and token statistics live under `motius/motion/representation/ardy/`; inference scheduling and CFG live under `motius/pipelines/ardy/`, diffusion under `motius/diffusion/ardy.py`, and artifact configuration/loading under `motius/configuration/ardy/`. The ModelBundle ownership and current-weight artifact migration is in progress.

The vendored LLM2Vec components retain their upstream MIT attribution from
[McGill-NLP/llm2vec](https://github.com/McGill-NLP/llm2vec). Checkpoint weights
remain governed by the license published in each NVIDIA Hugging Face model
repository.
