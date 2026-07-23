# Motion-to-Text · HumanML3D

Motius evaluates motion captioning with one model-independent HumanML3D
protocol. TM2T, MotionGPT, MotionGPT3, VerMo, and future methods all emit the
same resumable prediction-record format before any metric is computed.

## Protocol

| Item | Definition |
| ---- | ---------- |
| Split | HumanML3D official `test.txt` |
| Source length | Accept `[40, 200)` source frames; evaluator input is capped at 196 frames |
| Population | 4,400 evaluated samples from 4,402 raw entries, reproducing the released `Motion2TextEvalDataset` overwrite and length semantics |
| Official references | First three TM2T token/lemma captions; one reference is repeated three times, while two references become A/B/A |
| Raw-reference diagnostic | The same three-reference policy over the original HumanML3D captions |
| Candidate group | 32 motions/captions for semantic R-Precision |
| Repeats | One deterministic pass by default |
| Text metrics | BLEU-1/2/3/4, ROUGE-L, CIDEr, raw BERTScore F1, and baseline-rescaled BERTScore F1 |
| Semantic metrics | Text-query-to-motion R@1/2/3 and Matching Score with the official HumanML3D matching network |
| GT reference row | The first TM2T reference is evaluated as the prediction; GT never participates in method ranking |

The protocol manifest is relocatable: motion paths are stored relative to the
HumanML3D root. Prediction records retain both raw and tokenized references.
Language predictions are never rewritten; TM2T-compatible spaCy normalization
is applied only before the semantic evaluator. R-Precision treats each generated
caption as the query and the 32 motions as candidates, matching the official
evaluator's distance-matrix orientation.

## Inference

Every released M2T pipeline exposes the same call:

```python
captions = pipeline.infer_m2t(
    motions,  # list of denormalized HumanML3D-263 arrays
    lengths=[len(motion) for motion in motions],
)
```

Run a full baseline with the shared protocol:

```bash
python tools/run_m2t_humanml3d.py \
  --method motiongpt3 \
  --data-root /path/to/HumanML3D \
  --protocol-manifest outputs/m2t/humanml3d/protocol_manifest.json \
  --output-dir outputs/m2t/humanml3d/motiongpt3
```

Large runs can be resumed and split without overlap:

```bash
python tools/run_m2t_humanml3d.py \
  --method tm2t \
  --data-root /path/to/HumanML3D \
  --output-dir outputs/m2t/humanml3d/tm2t \
  --num-shards 8 --shard-index 0
```

## Evaluation

```bash
python tools/eval_m2t_humanml3d.py \
  --prediction-dir outputs/m2t/humanml3d/motiongpt3 \
  --semantic-artifact ZeyuLing/motius-evaluator-humanml3d-official \
  --chunk-size 32 --n-repeats 1 \
  --output outputs/m2t/humanml3d/motiongpt3/metrics.json
```

Use raw HumanML3D captions to measure sensitivity to the reference style:

```bash
python tools/eval_m2t_humanml3d.py \
  --prediction-dir outputs/m2t/humanml3d/motiongpt3 \
  --protocol-manifest outputs/m2t/humanml3d/protocol_manifest.json \
  --language-reference-mode raw \
  --output outputs/m2t/humanml3d/motiongpt3/metrics_raw_refs.json
```

## Language Metric Protocols

The two published M2T papers do not use an interchangeable language protocol.
The token-reference track reproduces the original TM2T table, while the raw-
caption track closely reproduces MotionGPT's re-evaluation of TM2T. This is a
reference-style effect, not a COCO scorer failure.

| Method | Reference | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled |
| ------ | --------- | -----: | -----: | ------: | ----: | -------: | -------------: |
| TM2T | token | 61.05 | 22.13 | 49.11 | 72.53 | 89.40 | 37.21 |
| TM2T | raw | 49.12 | 7.60 | 38.08 | 17.19 | 88.55 | 32.15 |
| MotionGPT | token | 39.71 | 4.60 | 33.65 | 7.82 | 88.51 | 31.93 |
| MotionGPT | raw | 46.05 | 12.10 | 38.49 | 34.09 | 88.84 | 33.88 |
| MotionGPT3 | token | 46.35 | 6.51 | 36.72 | 10.63 | 88.20 | 30.07 |
| MotionGPT3 | raw | 53.83 | 16.05 | 41.65 | 40.77 | 88.90 | 34.25 |
| VerMo | token | 45.01 | 5.79 | 37.03 | 9.48 | 88.78 | 33.50 |
| VerMo | raw | 52.54 | 16.09 | 42.23 | 40.06 | 89.45 | 37.48 |

These BERT columns are the same underlying similarity, not two independent
metrics. For the default `roberta-large` layer 17 setup, BERTScore uses the
English F1 baseline `0.83122575` and computes
`rescaled = (raw - baseline) / (1 - baseline)`. Raw scores therefore cluster
near 90 while the TM2T-paper scale clusters near 30 to 40.

BLEU, ROUGE, CIDEr, and BERTScore remain compatibility metrics. They should not
be treated as a complete measure of caption correctness because HumanML3D
references omit valid motion details and allow many paraphrases. A learned or
LLM-based judge must be version-pinned and human-calibrated before it can affect
ranking.

The public [Motion-to-Text · HumanML3D benchmark](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard)
only accepts results produced from this complete protocol population.
