# Scope: 2024-2026 LLM-based evaluation metrics for open-ended text generation and caption quality

## User Goal
Determine whether HumanML3D motion-to-text scores are incorrectly low, identify
current evaluation practice for open-ended generation, and design a fair Motius
M2T protocol that does not reward one model's lexical style.

## Seed File
none

## Extracted Seeds
- task: motion captioning and open-ended generation evaluation
- method family: lexical metrics, embedding metrics, LLM-as-a-judge, multimodal judges
- modality: motion/video plus generated text
- datasets or benchmarks: HumanML3D, MT-Bench, AlpacaEval 2, Arena-Hard, caption evaluation
- venues and years: EMNLP 2023, ACL/CVPR 2024, arXiv 2024-2025
- known baselines: BLEU, ROUGE-L, CIDEr, BERTScore, TM2T evaluator
- positive keywords: pairwise preference, win rate, rubric, reference-free, human alignment
- negative scope: closed-form QA and code benchmarks with executable ground truth
