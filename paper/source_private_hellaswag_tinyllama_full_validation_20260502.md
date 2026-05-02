# TinyLlama Full-Validation Source-Family Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: stronger COLM-ready positive method; ICLR still gated by strict receiver-family transfer, benchmark diversity, and native GPU systems rows.
- Current story: a fixed `2B` raw / `5B` framed source-private hidden-innovation packet improves HellaSwag over label-copy, score-only, zero-hidden, wrong-example, and candidate-roll controls. This now holds for Qwen full validation and TinyLlama full validation.
- Exact remaining blocker: show that a different receiver family can decode or exploit the packet under the same source-private contract, then add native systems measurements.

## What Was Tested

We replaced the Qwen source with TinyLlama and ran the same hidden-innovation packet gate on all `10042` frozen HellaSwag validation rows. In plain terms: a different small model generated the tiny hidden hint for every validation example, and we tested whether that hint still helped the frozen downstream selector choose the right ending.

The run required a Mac systems fix. Dynamic MPS padding crashed on an Apple MPS attention-shape compiler error. We replaced arbitrary dynamic padding with bucketed MPS padding to the next 32-token boundary, which completed the full run.

## Result

Artifact:
`results/source_private_hellaswag_hidden_innovation_eval_full_stress_20260502_tinyllama_train512_validation0_10042/hellaswag_hidden_innovation_eval_slice_stress.json`

Important note: the wrapper artifact reports `pass_gate=false` because the older slice wrapper only marks post-first1024 heldout slices as standard. The actual full-validation `bagged_gate.pass_gate` is `true`, and the full-validation source-family card consumes that bagged result.

Headline:

| Metric | Value |
|---|---:|
| eval rows | `10042` |
| selected accuracy | `0.619199` |
| best label-copy accuracy | `0.558753` |
| score-only accuracy | `0.558753` |
| delta vs best label-copy | `+0.060446` |
| CI95 low vs best label-copy | `+0.053423` |
| CI95 low vs score-only | `+0.054319` |
| jackknife subbags passing | `3/3` |
| wrong-example hidden control | `0.456682` |
| candidate-roll hidden control | `0.366162` |
| packet | `2B` raw / `5B` framed |

Systems card:
`results/source_private_hellaswag_tinyllama_full_mac_systems_card_20260502/tinyllama_full_mac_systems_card.json`

Mac-local timing:

| Phase | Seconds |
|---|---:|
| source scoring | `1765.874` |
| source hidden extraction | `1709.470` |
| packet build and gate | `94.698` |
| total wall time | `3570.041` |

## Interpretation

This promotes the source-family robustness claim. The method is no longer supported only by Qwen full validation plus a TinyLlama slice; it now has full-validation evidence from TinyLlama as the source family. That weakens the reviewer objection that the signal is just Qwen-specific hidden-coordinate geometry.

This does not prove general cross-family latent communication. The receiver/evaluator side did not change family. The safe ICLR wording is: source-family-robust, source-private hidden-innovation repair under a fixed receiver contract.

## Contribution Status

Promoted:

- Extreme-rate source-private hidden-innovation packet: full HellaSwag validation pass for Qwen and TinyLlama source families.
- Destructive-control evaluation ladder: label-copy, score-only, zero-hidden, wrong-example hidden, candidate-roll hidden, and train-sample jackknife all remain part of the gate.
- Mac systems boundary: full run records MPS source scoring, hidden extraction, packet bytes, cache sizes, and no-text/no-KV/no-raw-hidden exposure flags.

Still blocked:

- True receiver-family transfer, e.g. Phi/Qwen or Qwen/Phi under the same packet contract.
- Native NVIDIA/vLLM/SGLang systems rows.
- Benchmark diversity beyond HellaSwag.

## Decision

Promote TinyLlama full validation from a scout to a core evidence row. The next exact gate should be a strict receiver-family falsification pair if the Mac can run it; otherwise, run a smaller Phi source-family slice first to determine feasibility.
