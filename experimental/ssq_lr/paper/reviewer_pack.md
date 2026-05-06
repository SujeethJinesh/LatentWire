# SSQ-LR Reviewer Pack

- status: preregistered Mac-gate scaffold; no positive method result
- current decision: blocked on real hybrid SSM-state dumps
- camera-readiness: not submittable as a paper result until S1--S3 have real
  trace evidence

## Paper Link

- Draft PDF: `experimental/ssq_lr/paper/ssq_lr_colm2026.pdf`
- Draft TeX: `experimental/ssq_lr/paper/ssq_lr_colm2026.tex`
- Preregistration:
  `experimental/ssq_lr/phase2/preregister_ssq_lr_20260506.md`

## Current Claim

SSQ-LR asks whether recurrent SSM state in hybrid reasoners can be quantized
below FP16 without quality loss. The current artifacts do not claim that the
recipe exists. They define the Mac-first gate, exact packet contract, controls,
and kill conditions.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | The intended surfaces are AIME/GSM/MATH-style reasoning traces plus continuation NLL, but no live hybrid model has been dumped yet. | Gate pending. |
| Ablations | BF16 no-op, INT8, FP8-style, MXFP4-style, random same-L2 noise, scale shuffle, and byte accounting are preregistered. | Adequate before real S2. |
| Correctness | The packet checker now requires prefill_end, 2k_or_end, 8k_or_end, and final_minus_128 buckets for every prompt/layer pair, at least 12 prompts unless resource-limited, matching model IDs, BF16 controls, finite row fields, hash-shaped prompt/architecture provenance, and decision-grade `summary.json` aggregates. | Artifact path is hardened. |
| Reproducibility | Synthetic S1 packet is reproducible, but real S1 cannot run until hybrid weights are available on the host. | Not model evidence. |
| Novelty | The wedge is sub-FP16 recurrent state for hybrid reasoners, not weight-only or KV-cache quantization. | Plausible only if real S2/S3 pass. |
| Camera-readiness | The draft is a preregistration shell. It needs real S1/S2/S3 tables before submission as a method paper. | Not camera-ready. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| synthetic S1 | late/early max-abs ratio 8.461, std ratio 3.640, kurtosis ratio 3.141 | validates readout only |
| architecture provenance | shared config-derived hashes exist for live hybrid targets | packet provenance ready |
| model eligibility | live targets are identified, but weights are not cached locally | blocked on model load |
| real-packet checker | rejects missing buckets, incomplete prompt/layer matrices, too few prompts, promotable resource-limited decisions, mismatched model IDs, and missing controls | ready for real S1 |

## Reviewer Risks

- No real hybrid SSM state has been measured.
- No quality, NLL, accuracy, memory, throughput, or GPU result exists.
- A sub-FP16 state recipe may need per-model retuning, which would kill the
  cross-model claim.
- Byte savings must include scale and metadata overhead, not just nominal dtype.

## Next Exact Gate

Run S1 on the smallest available live hybrid model. The first packet must pass:

```bash
./venv_arm64/bin/python -m experimental.shared.check_gate_packet \
  experimental/ssq_lr/phase2/results/ssq_lr_gate_s1_<YYYYMMDD>_<model_slug> \
  --mode real --project ssq_lr
```

Continue only if real state heterogeneity passes the preregistered S1 rule.
If the run is resource-limited, record it with
`RESOURCE_LIMITED_NOT_PROMOTABLE` and do not treat it as S1 promotion.
