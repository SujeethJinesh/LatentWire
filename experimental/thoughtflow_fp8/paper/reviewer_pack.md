# ThoughtFlow-FP8 Reviewer Pack

- status: falsification-methodology workshop note; no live compression method
- current decision: no live positive method branch
- camera-readiness: ready as a methodology/negative-results workshop diagnostic after copyedit, venue-specific framing, and historical phase-doc supersession; remaining work is not new experiments

## Paper Link

- Draft PDF: `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`
- Draft TeX: `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`
- Current decision manifest:
  `experimental/thoughtflow_fp8/phase2/current_decision_manifest_20260506.md`
- Tracked diagnostic packet:
  `experimental/thoughtflow_fp8/phase2/diagnostic_packets/thoughtflow_diagnostic_packet_20260506/`

## Current Claim

ThoughtFlow-FP8 shows a reusable sparse-cache falsification protocol: freeze a
quality surface, pre-register each successor signal, require nominal-budget
proxy wins with achieved keep rates reported, separate stopped-family and
proxy-baseline rows, and report paired uncertainty plus oracle/headroom
diagnostics. The case study shows an
interpretable retention signal looking positive on one frozen surface and then
failing stricter reproduction. Later prefix-surprisal and value-weighted
attention-contribution successors also fail on fresh surfaces. The project does
not currently claim a new KV-compression method, FP8 serving result, CUDA kernel
result, or latency/throughput win. All benchmark surfaces are Mac-local
saved-trace falsification fixtures, not reasoning-model benchmark claims.

## COLM Review Readout

| Axis | Reviewer read | Current decision |
|---|---|---|
| Benchmarks | Mac-local `distilgpt2` sparse-cache scoring is useful for falsification, but it is not a reasoning-model benchmark and does not include faithful LongFlow, ThinKV, R-KV/R-KVHash, PM-KVQ, KVQuant, or RaaS implementations. | Diagnostic only. |
| Ablations | The project has synthetic retention, retained-text NLL, hidden/KV telemetry, cache-dropping quality, train-fixed sparse sweeps, same-slice RDU reproduction, alternate-surface RDU, independent-trace RDU, and fresh PSI/VWAC successor gates. The decisive ablation set is negative: `rdu_topk` fails independent proxy-baseline separation, while `psi_topk` and `vwac_topk` fail on fresh surfaces. | Sufficient to stop this branch family. |
| Correctness | CPU sparse-cache scoring, paired uncertainty, oracle/headroom reporting, RDU telemetry, and int8 anchor/phase Triton-interpreter parity are tested. These do not establish native FP8, CUDA, latency, throughput, or serving correctness. | Correctness scaffold only. |
| Reproducibility | Markdown and JSON artifacts are tracked for every gate, and the owned test command is stable under `./venv_arm64`. Historical `ALIVE`/`REPRODUCED`/`PROMOTED` artifacts are preserved for auditability but superseded by the current decision manifest. The diagnostic packet builder now refuses dirty `experimental/thoughtflow_fp8` regeneration, the current manifest records a clean path at generation, and every hashed packet input is present in this local workspace; some legacy inputs are under ignored `results/` paths and are not claimed as clean-checkout assets. | Good enough for a diagnostic note with explicit local-workspace provenance. |
| Novelty | The method space is crowded by recent sparse/quantized KV-cache work, including LongFlow, ThinKV, R-KV/R-KVHash, LazyEviction/ForesightKV-style future-use signals, PM-KVQ, KVQuant, TriAttention, KVzip/KVzap, Q-Filters, and KV-Direct. The defensible novelty is the falsification ladder and stop rule, not a new compression method. | Do not claim method novelty. |
| Camera-readiness | The current draft is close only as a falsification-methodology workshop note with the first-surface RDU table clearly marked as failed-to-reproduce and `distilgpt2`/proxy-baseline/non-native caveats visible on the first read. It is not a mainline method or systems paper. | Workshop diagnostic at most. |

## 2026-05-07 Reviewer-Facing Paper Cleanup

The paper now places a final signal-status matrix immediately after the
protocol contribution and includes a registration-to-measurement ledger with
preregistration and artifact hash prefixes. The stop-rule section has been
reworded around the diagnostic ladder contribution rather than apologizing for
the lack of a positive method, and the stable TeX test command now matches this
reviewer pack's Triton interpreter environment.
The latest polish adds an explicit first-page reviewer takeaway and a final
paragraph stating the artifact's methodological value: it prevents narrow false
positives from becoming GPU/kernel work before robustness exists.

## Strongest Evidence

| Gate | Result | Decision |
|---|---|---|
| retained-text NLL | original ThoughtFlow rows do not beat strongest proxy | weakened |
| hidden/KV telemetry | phase recall improves, math-state utility is unstable | diagnostic |
| CPU sparse-cache probe | tuned row is promising in mean but not robust against ThinKV-like uncertainty | not promoted |
| frozen 74-trace `rdu_topk` | beats ThinKV-like and R-KV-like on first frozen surface | historical candidate only |
| same-slice rerun | reproduces the cached gate exactly | bookkeeping only |
| alternate surface | stopped same-family row beats `rdu_topk` by 0.006 NLL | weakened |
| independent saved traces | R-KV-like is best compressed; `rdu_topk` fails proxy-baseline separation | stopped |
| fresh prefix-surprisal utility | `psi_topk` NLL 7.899 versus ThinKV-like 3.906 and R-KV-like 3.960 on 70 fresh C2C GSM70 traces | killed |
| fresh value-weighted attention utility | `vwac_topk` NLL 4.336 versus R-KV-like 4.096 and ThinKV-like 4.162 on 64 fresh C2C SVAMP70 traces | killed |
| Triton interpreter | anchor/phase int8 primitive matches CPU reference | kernel logic only |
| current authority | historical `ALIVE`/`REPRODUCED`/`PROMOTED` artifacts are superseded by the current decision manifest | manifest controls |

## Stable Owned-Test Command

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python -m pytest \
  experimental/thoughtflow_fp8/phase2/tests \
  experimental/thoughtflow_fp8/phase4/tests -rs
```

## Reviewer Risks

- The project name still says FP8, but no real FP8 serving result exists.
- The first-surface `rdu_topk` table is historical and failed reproduction.
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
with stopped-family, proxy-baseline, paired uncertainty, and oracle/headroom
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
- promotion rules requiring nominal-budget quality wins over R-KV-like and
  ThinKV-like with achieved keep rates reported, strict same-family separation,
  paired uncertainty, and oracle/headroom diagnostics.

Until that artifact exists, there is no runnable successor gate.

## Primary Literature Anchors

See the paper bibliography for canonical URLs. Keep this pack as a reviewer
readout, not a second citation manifest.
