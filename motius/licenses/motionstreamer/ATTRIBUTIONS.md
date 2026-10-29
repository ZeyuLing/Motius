# MotionStreamer attribution

Neural TAE, autoregressive transformer, diffusion head, sampling and training
objectives derive from zju3dv/MotionStreamer commit
`8aace3f6f5f564ae1518aa77d925ae1380070c80`. The complete upstream Project
Registration License (PRL) v1.0 is preserved in `UPSTREAM_LICENSE.md`.
SentenceT5 uses the actual Hugging Face T5 encoder and a tensor-only mean
pooling, projection and normalization graph; tokenizer and batching live in
the data processor. Generation, diffusion scheduling and training objectives
are outside the network package. Unused original application and embedding
resize utilities are omitted.

Released original AR/TAE checkpoints: `lxxiao/MotionStreamer`, revision
`24631098fe096ab55c4d95a3af410397bb05109a`. Original text neural files:
`sentence-transformers/sentence-t5-xxl`, revision
`97f38f2f69860f732f602829da6df76ee5c4c0f9`.
