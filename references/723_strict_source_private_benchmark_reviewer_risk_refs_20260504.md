# Reference Memo 723: Strict Source-Private Benchmark Reviewer-Risk Design

Date: 2026-05-04

## Local Purpose

This memo records fresh benchmark/reviewer-risk sources for strict
source-private ARC/OpenBookQA/HellaSwag evaluation during the Sparse Resonance
Packet pivot. It supports destructive controls, leakage accounting, paired
uncertainty, and utility-per-byte reporting if the current gates remain
negative.

## Benchmark Sources

- Clark et al., 2018, "Think you have Solved Question Answering? Try ARC, the
  AI2 Reasoning Challenge."
  Source: https://arxiv.org/abs/1803.05457
  Boundary: ARC-Challenge is a natural grade-school science MCQ benchmark whose
  Challenge Set was filtered by retrieval/co-occurrence failures. It is a
  strong small-model reasoning surface but remains vulnerable to public
  benchmark contamination and option-order artifacts.

- Mihaylov et al., 2018, "Can a Suit of Armor Conduct Electricity? A New
  Dataset for Open Book Question Answering."
  Source: https://aclanthology.org/D18-1260/
  Boundary: OpenBookQA needs an open-book fact plus common knowledge. For
  LatentWire, `answerKey`, `fact1`, `humanScore`, `clarity`, and worker metadata
  must stay forbidden to packet builders and calibration code.

- Zellers et al., 2019, "HellaSwag: Can a Machine Really Finish Your Sentence?"
  Source: https://aclanthology.org/P19-1472/
  Boundary: HellaSwag uses adversarial filtering and is useful for commonsense
  continuation, but MCQ option ordering, candidate text permutation, and public
  benchmark contamination must be explicitly audited.

## Leakage and Benchmark Integrity Sources

- Li et al., 2023, "An Open Source Data Contamination Report for Large Language
  Models."
  Source: https://arxiv.org/abs/2310.17589
  Boundary: motivates benchmark contamination reporting for public MCQ tasks,
  including HellaSwag. Use this to justify exact/paraphrase overlap checks and
  model snapshot disclosure.

- Oren et al., 2023, "Proving Test Set Contamination in Black Box Language
  Models."
  Source: https://arxiv.org/abs/2310.17623
  Boundary: motivates permutation/exchangeability-style contamination probes.
  For LatentWire, canonical-vs-shuffled candidate/text order likelihood should
  be treated as a leakage diagnostic, not as the main task score.

- Yang et al., 2023, "Rethinking Benchmark and Contamination for Language
  Models with Rephrased Samples."
  Source: https://arxiv.org/abs/2311.04850
  Boundary: string overlap alone is insufficient; paraphrase and synthetic-data
  overlap are relevant threats. Use one-time frozen slices and content hashes.

- Xu et al., 2024, "Benchmarking Benchmark Leakage in Large Language Models."
  Source: https://arxiv.org/abs/2404.18824
  Boundary: supports a benchmark transparency card: model snapshots, split
  hashes, prompt templates, decontamination checks, and whether eval rows were
  used in any model or receiver selection.

- Golchin and Surdeanu, 2024, "Leak, Cheat, Repeat: Data Contamination and
  Evaluation Malpractices in Closed-Source LLMs."
  Source: https://arxiv.org/abs/2402.03927
  Boundary: indirect leakage via iterative eval use is a reviewer risk. Treat
  every post-hoc gate on ARC/OpenBookQA/HellaSwag as a development exposure and
  require a frozen final slice.

## MCQ Order and Label Bias Sources

- Pezeshkpour and Hruschka, 2023, "Large Language Models Sensitivity to The
  Order of Options in Multiple-Choice Questions."
  Source: https://arxiv.org/abs/2308.11483
  Boundary: motivates physical answer-option permutation, canonical remapping,
  and all label-copy/position-copy baselines.

- Zheng et al., 2024, "Large Language Models Are Not Robust Multiple Choice
  Selectors."
  Source: https://arxiv.org/abs/2309.03882
  Boundary: option-ID token bias is a direct risk for source-index, source-rank,
  and tiny candidate-code packets. Every positive row needs candidate-roll,
  label-permutation, and wrong-remap collapse.

## Paired Uncertainty Sources

- Koehn, 2004, "Statistical Significance Tests for Machine Translation
  Evaluation."
  Source: https://www.research.ed.ac.uk/en/publications/statistical-significance-tests-for-machine-translation-evaluation/
  Boundary: paired bootstrap resampling is accepted in NLP, but tiny n slices
  are not enough to distinguish one- or two-example gains.

- Dror et al., 2018, "The Hitchhiker's Guide to Testing Statistical
  Significance in Natural Language Processing."
  Source: https://aclanthology.org/P18-1128/
  Boundary: motivates explicit paired test selection, full CIs, and avoiding
  uncorrected multiple looks across many method variants.

## Communication and Utility-Per-Byte Baselines

- Fu et al., 2026, "Cache-to-Cache: Direct Semantic Communication Between Large
  Language Models."
  Source: https://arxiv.org/abs/2510.03215
  Boundary: strongest dense semantic communication competitor. LatentWire can
  claim source privacy and byte-scale interpretability, not generic latent
  communication novelty, unless it beats C2C or reports complementary tradeoffs.

- Ye et al., 2025, "KVCOMM: Online Cross-context KV-cache Communication for
  Efficient LLM-based Multi-agent Systems."
  Source: https://arxiv.org/abs/2510.12872
  Boundary: native KV reuse/speed competitor. LatentWire must not claim
  throughput superiority without native vLLM/SGLang measurements.

- Kwon et al., 2023, "Efficient Memory Management for Large Language Model
  Serving with PagedAttention."
  Source: https://arxiv.org/abs/2309.06180
  Boundary: vLLM/PagedAttention is the native serving reference point for TTFT,
  TPOT, goodput, HBM, and KV accounting claims.

- Zandieh et al., 2025, "TurboQuant: Online Vector Quantization with
  Near-optimal Distortion Rate."
  Source: https://arxiv.org/abs/2504.19874
  Boundary: strong KV/vector quantization byte-floor comparator. Sparse
  Resonance Packet utility-per-byte should report dense and quantized KV floors
  separately from framed packet bytes.

## Reviewer-Risk Conclusion

If Sparse Resonance gates remain negative, COLM_v2 can still claim a rigorous
strict source-private evaluation framework, byte/exposure accounting, and a
negative method map showing that shallow PCA/score/top2/soft-prefix receivers
do not beat target-cache and source-family substitution controls. It cannot
claim a positive ICLR method, hidden-state communication, native acceleration,
or superiority over dense KV communication.
