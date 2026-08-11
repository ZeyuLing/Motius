# UniMuMo attribution

This Motius integration independently implements the inference architecture
described by **UniMuMo: Unified Text, Music and Motion Generation** and maps the
authors' published checkpoint into a safe, self-contained artifact.

- Paper: <https://arxiv.org/abs/2410.04534>
- Authors' repository: <https://github.com/hanyangclarence/UniMuMo>
- Authors' checkpoint: <https://huggingface.co/ClarenceY/unimumo>
- Audited source revision: `a75ddac791ff6806b5bd511d1ce887a1980e20d5`

The upstream repository and checkpoint do not declare a license at the audited
revision. Redistribution and downstream use remain subject to the upstream
authors' terms. T5 is distributed under Apache-2.0. The Encodec architecture is
loaded through Hugging Face Transformers; its weights are taken from the
authors' UniMuMo package during artifact conversion.
