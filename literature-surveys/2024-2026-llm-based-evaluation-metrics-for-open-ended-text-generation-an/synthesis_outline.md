# Synthesis Outline: 2024-2026 LLM-based evaluation metrics for open-ended text generation and caption quality

## TL;DR

The low Motius scores are primarily a protocol-style problem, not a broken COCO
implementation. TM2T token/lemma references strongly reward TM2T-like output.
Modern open-ended benchmarks rank systems with human or LLM pairwise preference,
often length-controlled; modern caption metrics increasingly ground the judge in
the source modality. Motius should retain official lexical scores only for paper
compatibility and add a calibrated motion-grounded judge as the primary quality
track.

## Taxonomy

1. Lexical overlap: BLEU, ROUGE-L, CIDEr. Reproducible but paraphrase-sensitive.
2. Text embedding: BERTScore. More semantic but still reference-bound.
3. Text-only LLM judge: candidate plus references, rubric score and explanation.
4. Motion-grounded judge: rendered/joint motion plus candidate, optionally refs.
5. Human pairwise preference: calibration gold standard, not a scalable daily metric.

## Timeline

- 2022-2023: TM2T/MotionGPT use lexical and BERT-based caption metrics.
- 2023: G-Eval, MT-Bench/Arena and CLAIR establish rubric/pairwise LLM judges.
- 2024: AlpacaEval controls length; Arena-Hard emphasizes ranking separation;
  FLEUR grounds caption judging in the input modality; JudgeBench audits judges.
- 2025: VideoJudge reports that video-conditioned judges beat text-only judges.

## Core Papers

See `papers_merged.csv`; P01/P02 define the legacy motion protocol, P04/P06
define leaderboard-style pairwise metrics, and P09/P11 motivate source grounding.

## Method and Benchmark Comparison

Recommended Motius tracks:

| Track | Public score | Role |
|---|---|---|
| Official TM2T | BLEU-4, ROUGE-L, CIDEr, BERTScore | Reproduce historical papers |
| Raw-reference diagnostic | Same metrics over raw HumanML3D captions | Expose style sensitivity |
| Motion semantic | R@1/2/3 and Matching Distance | Cheap motion-text alignment |
| Motion Caption Judge | rubric score plus pairwise win rate | Primary semantic quality after calibration |

The judge rubric should separately score action/pose, body parts and left-right,
temporal order, trajectory/direction, and coverage versus hallucination. Pairwise
evaluation must swap answer order and report caption length; model/judge versions
and prompts must be pinned.

## Debates and Gaps

- Text references are incomplete descriptions of a motion, so even a perfect
  paraphrase metric cannot verify omitted or hallucinated motion facts.
- LLM judges have position, verbosity, family and self-preference biases.
- Human labels on roughly 200 stratified motion/caption pairs are required before
  a judge score should determine leaderboard rank.

## Missing Citations or Novelty Risks

- Motius still needs a dedicated motion-caption human meta-evaluation set.
- The final choice between an open video judge and a joint-sequence-aware judge
  should be based on that calibration, not on general LLM benchmark rankings.

## References

Canonical URLs and identifiers are recorded in `papers_merged.csv`.
