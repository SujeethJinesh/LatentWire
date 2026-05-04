# Reference Memo 724: Behavior-Aligned Packet Gate Controls

Date: 2026-05-04

## Local Purpose

This memo records the targeted benchmark and reviewer-risk sources used for the
next behavior-aligned Sparse Resonance Packet gate. It sharpens controls for
source-choice copying, target-cache effects, label/order bias, and
cherry-picking.

## Benchmark Surface Sources

- Clark et al., 2018, "Think you have Solved Question Answering? Try ARC, the
  AI2 Reasoning Challenge."
  Source: https://arxiv.org/abs/1803.05457
  Use: ARC-Challenge remains a useful grade-school science MCQ surface, but any
  positive gate needs contamination and option-order controls because it is
  public and multiple choice.

- Mihaylov et al., 2018, "Can a Suit of Armor Conduct Electricity? A New
  Dataset for Open Book Question Answering."
  Source: https://aclanthology.org/D18-1260/
  Use: OpenBookQA specifically combines open-book facts and broad common
  knowledge. LatentWire packet builders must forbid answerKey, fact1,
  humanScore, clarity, and worker metadata.

- Zellers et al., 2019, "HellaSwag: Can a Machine Really Finish Your
  Sentence?"
  Source: https://aclanthology.org/P19-1472/
  Use: HellaSwag is valuable for adversarially filtered commonsense
  continuation, but option-order, candidate permutation, and public benchmark
  contamination controls are mandatory.

## Label and Order Bias Sources

- Pezeshkpour and Hruschka, 2024, "Large Language Models Sensitivity to The
  Order of Options in Multiple-Choice Questions."
  Source: https://aclanthology.org/2024.findings-naacl.130/
  Use: motivates physical candidate-text permutation, canonical remapping,
  wrong-remap collapse, and top-2 ambiguity diagnostics.

- Zheng et al., 2024, "Large Language Models Are Not Robust Multiple Choice
  Selectors."
  Source: https://arxiv.org/abs/2309.03882
  Use: motivates source-index/source-rank/source-score baselines and explicit
  option-ID prior controls because LLMs can prefer option labels independent of
  content.

- Alzahrani et al., 2024, "When Benchmarks are Targets: Revealing the
  Sensitivity of Large Language Model Leaderboards."
  Source: https://arxiv.org/abs/2402.01781
  Use: motivates reporting benchmark perturbation robustness rather than only a
  single leaderboard-style accuracy row.

## Leakage and Cherry-Picking Sources

- Xu et al., 2024, "Benchmarking Benchmark Leakage in Large Language Models."
  Source: https://arxiv.org/abs/2404.18824
  Use: motivates a benchmark transparency card with model snapshots, split
  hashes, prompt hashes, benchmark use history, and reproducible predictions.

- Oren et al., 2023, "Proving Test Set Contamination in Black Box Language
  Models."
  Source: https://arxiv.org/abs/2310.17623
  Use: motivates exchangeability/order-based leakage probes and canonical-vs-
  shuffled likelihood diagnostics on public slices.

- Balloccu et al., 2024, "Leak, Cheat, Repeat: Data Contamination and
  Evaluation Malpractices in Closed-Source LLMs."
  Source: https://arxiv.org/abs/2402.03927
  Use: motivates logging iterative benchmark exposure, missing baselines, and
  reproducibility gaps as reviewer risks.

## Uncertainty Sources

- Dror et al., 2018, "The Hitchhiker's Guide to Testing Statistical
  Significance in Natural Language Processing."
  Source: https://aclanthology.org/P18-1128/
  Use: motivates predeclared significance testing, paired tests for paired
  outputs, and avoiding uncorrected multiple looks.

- Koehn, 2004, "Statistical Significance Tests for Machine Translation
  Evaluation."
  Source: https://aclanthology.org/W04-3250/
  Use: paired/bootstrap resampling precedent for NLP evaluation.

## Control Design Consequence

The next behavior-aligned packet cannot be credited as Sparse Resonance transfer
unless it beats compact candidate-only/source-choice packets, same-source-choice
wrong-row packets, target-derived same-byte packets, physical candidate
permutation controls, destructive atom/coeff controls, and predeclared frozen
slice uncertainty. Otherwise the correct interpretation is source-choice
compression, target-cache reuse, label/order bias, or cherry-picked scout noise.
