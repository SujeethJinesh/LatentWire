# ThoughtFlow-FP8 Reviewer Pack

- status: diagnostic/falsification workshop note only
- current decision: no live positive method branch

## Paper Link

- Draft PDF: `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.pdf`
- Draft TeX: `experimental/thoughtflow_fp8/paper/thoughtflow_fp8_colm2026.tex`

## Current Claim

ThoughtFlow-FP8 shows a sparse-cache falsification ladder: an interpretable
retention signal can look positive on one frozen surface and then fail stricter
same-family and cross-family reproduction. The project does not currently claim
a new KV-compression method, FP8 serving result, CUDA kernel result, or
latency/throughput win.

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
| Triton interpreter | anchor/phase int8 primitive matches CPU reference | kernel logic only |

## Reviewer Risks

- The project name still says FP8, but no real FP8 serving result exists.
- The positive-looking `rdu_topk` table is historical and failed reproduction.
- Current evidence is not a positive method result.
- A useful future branch requires fresh preregistration and a new frozen
  surface, not another retune on current traces.

## Next Exact Gate

Stop local method experimentation on the current branch. Reopen only with a new
pre-registered utility signal and a one-shot fresh/larger sparse-cache gate with
same-family, cross-family, paired uncertainty, and oracle/headroom reporting.
