# ThoughtFlow-FP8 Reviewer Pack

- status: diagnostic/falsification workshop note only
- current decision: no live positive method branch
- camera-readiness: ready only as a negative/mixed workshop diagnostic

## Paper Link

- Draft PDF: `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`
- Draft TeX: `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`
- Current decision manifest:
  `experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`

## Current Claim

ThoughtFlow-FP8 shows a sparse-cache falsification ladder: an interpretable
retention signal can look positive on one frozen surface and then fail stricter
same-family and cross-family reproduction. Later prefix-surprisal and
value-weighted attention-contribution successors also fail on fresh surfaces. The
project does not currently claim a new KV-compression method, FP8 serving
result, CUDA kernel result, or latency/throughput win.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | Mac-local `distilgpt2` sparse-cache scoring is useful for falsification, but it is not a reasoning-model benchmark and does not include faithful LongFlow, ThinKV, R-KV/R-KVHash, PM-KVQ, KVQuant, or RaaS implementations. | Diagnostic only. |
| Ablations | The project has synthetic retention, retained-text NLL, hidden/KV telemetry, cache-dropping quality, train-fixed sparse sweeps, same-slice RDU reproduction, alternate-surface RDU, independent-trace RDU, and fresh PSI/VWAC successor gates. The decisive ablation set is negative: `rdu_topk` fails independent cross-family separation, while `psi_topk` and `vwac_topk` fail on fresh surfaces. | Sufficient to stop this branch family. |
| Correctness | CPU sparse-cache scoring, paired uncertainty, oracle/headroom reporting, RDU telemetry, and int8 anchor/phase Triton-interpreter parity are tested. These do not establish native FP8, CUDA, latency, throughput, or serving correctness. | Correctness scaffold only. |
| Reproducibility | Markdown and JSON artifacts are tracked for every gate, and the owned test command is stable under `./venv_arm64`. Historical `ALIVE`/`PROMOTED` artifacts are preserved for auditability but superseded by the current decision manifest. | Good enough for a diagnostic note. |
| Novelty | The method space is crowded by recent sparse/quantized KV-cache work, including LongFlow, ThinKV, R-KV/R-KVHash, LazyEviction/ForesightKV-style future-use signals, PM-KVQ, and KVQuant. The defensible novelty is the falsification ladder and stop rule, not a new compression method. | Do not claim method novelty. |
| Camera-readiness | The current draft is camera-ready only as a negative/mixed workshop diagnostic with the positive-looking RDU table clearly marked as failed-to-reproduce. It is not a mainline method or systems paper. | Workshop diagnostic at most. |

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| retained-text NLL | original ThoughtFlow rows do not beat strongest proxy | weakened |
| hidden/KV telemetry | phase recall improves, math-state utility is unstable | diagnostic |
| CPU sparse-cache probe | tuned row is promising in mean but not robust against ThinKV-like uncertainty | not promoted |
| frozen 74-trace `rdu_topk` | beats ThinKV-like and R-KV-like on first frozen surface | historical candidate only |
| same-slice rerun | reproduces the cached gate exactly | bookkeeping only |
| alternate surface | stopped same-family row beats `rdu_topk` by 0.006 NLL | weakened |
| independent saved traces | R-KV-like is best compressed; `rdu_topk` fails cross-family separation | stopped |
| fresh prefix-surprisal utility | `psi_topk` NLL 7.899 versus ThinKV-like 3.906 and R-KV-like 3.960 on 70 fresh C2C GSM70 traces | killed |
| fresh value-weighted attention utility | `vwac_topk` NLL 4.336 versus R-KV-like 4.096 and ThinKV-like 4.162 on 64 fresh C2C SVAMP70 traces | killed |
| Triton interpreter | anchor/phase int8 primitive matches CPU reference | kernel logic only |

## Reviewer Risks

- The project name still says FP8, but no real FP8 serving result exists.
- The positive-looking `rdu_topk` table is historical and failed reproduction.
- Current evidence is not a positive method result.
- A useful future branch requires fresh preregistration and a new frozen
  surface, not another retune on current traces.
- The phrase "ThoughtFlow-FP8" is historical; the current artifacts contain no
  real FP8 numerical drift or native FP8 serving evidence.
- Any table showing the first 74-trace `rdu_topk` win must be paired with the
  alternate-surface and independent-trace failures in the same section.

## Next Exact Gate

Stop local method experimentation on the current branches. Reopen only with a
new pre-registered utility signal and a one-shot fresh/larger sparse-cache gate
with same-family, cross-family, paired uncertainty, and oracle/headroom
reporting.

## Fresh Utility Signal Status

No fresh utility signal is currently pre-registered. The consumed `rdu_topk`,
`psi_topk`, and `vwac_topk` registrations cannot be rerun or retuned on the
current or fresh surfaces used here. A future Mac-feasible gate must first create a new
preregistration artifact that specifies:

- the new utility family and exact policy transform;
- forbidden inputs, including continuation loss and prior frozen outcomes;
- the fresh/larger frozen input surface;
- the one allowed evaluation command;
- promotion rules requiring matched-budget quality wins over R-KV-like and
  ThinKV-like, strict same-family separation, paired uncertainty, and
  oracle/headroom diagnostics.

Until that artifact exists, there is no runnable successor gate.

## Primary Literature Anchors

- LongFlow: `https://openreview.net/forum?id=rz6WybXjgk`
- ThinKV: `https://arxiv.org/abs/2510.01290`
- LazyEviction: `https://openreview.net/pdf?id=Mac3RzkEQu`
- ForesightKV: `https://arxiv.org/abs/2602.03203`
