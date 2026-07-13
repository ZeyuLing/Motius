# ARDY Attribution

The native runtime under `motius/models/ardy/network/` is adapted from
[nv-tlabs/ardy](https://github.com/nv-tlabs/ardy) at commit
`693f74d13b3d04a0a22ce127ee79c929dd89756b`.

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. ARDY is licensed under
Apache-2.0. Motius changes the Python package namespace, adds Python 3.9-safe
annotations, exposes local-path loading, and wraps inference in Motius bundle,
pipeline, representation, and conversion APIs.

The vendored LLM2Vec components retain their upstream MIT attribution from
[McGill-NLP/llm2vec](https://github.com/McGill-NLP/llm2vec). Checkpoint weights
remain governed by the license published in each NVIDIA Hugging Face model
repository.
