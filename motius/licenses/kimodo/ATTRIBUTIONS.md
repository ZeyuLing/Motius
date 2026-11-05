# KIMODO source attribution

Original implementation: NVIDIA, https://github.com/nv-tlabs/kimodo

Reference revision: `e1c219c99b4b94ba48be1fb6a1cbc64ecee4c954`.
Source headers and the Apache-2.0 license are preserved. The bundled LLM2Vec
implementation retains its McGill NLP MIT and NVIDIA Apache license headers.
Released weights and base language models have their own artifact licenses.

Neural implementations live under `models/kimodo/network`; sampling and
postprocessing under `pipelines/kimodo`; motion representation, skeletons and
constraints under `motion`; configuration/assets under `configuration/kimodo`.
The motion representation and multiprompt behavior track the pinned source.
A correction initializes the default multiprompt sample count before allocating
headings; explicit sample counts are compared against the unmodified source.
