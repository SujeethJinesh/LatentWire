# HellaSwag Source-Hidden Summary Repair Probe

- date: `2026-05-01`
- artifact: `results/source_private_hellaswag_hidden_summary_repair_probe_20260501_qwen05_train512_validation1024/`
- pass gate: `false`

## Question

The score-only HellaSwag repair gate failed even though the source top-2 oracle
remained high. This probe asks whether Qwen source hidden summaries contain a
repair signal that source log-likelihood score shape missed.

In plain terms: the earlier packet mostly said "the source model picked option
X." This experiment checks whether the source model's internal activations can
tell us when that first pick is wrong, then still forces the message through a
tiny source-private packet rather than exposing answer text or raw activations.

## Method

- source model: Qwen2.5-0.5B-Instruct, local CPU, continuation prompt
- train rows: `512` deterministic official HellaSwag train rows
- internal split: `384` fit / `128` dev
- frozen eval: HellaSwag validation first `1024`
- hidden feature: per-choice mean-pooled last-layer hidden state over the
  candidate continuation span
- train-only model selection: ridge in `{0.1, 1.0, 10.0, 100.0}`, selected on
  internal dev only
- selected model: layer `-1`, ridge `100.0`
- selected packet: `2B` raw / `5B` framed public hashed-residual sketch

The source-hidden classifier selects a candidate from hidden summaries. The
packet then encodes that selected candidate as a public hashed candidate
residual sketch. The packet does not expose source text, source KV cache, raw
hidden vectors, or raw source scores.

## Result

| Quantity | Value |
|---|---:|
| source-label copy | `0.461914` (`473/1024`) |
| hidden-label copy | `0.414063` (`424/1024`) |
| hidden packet | `0.413086` (`423/1024`) |
| hidden packet minus source-label copy | `-0.048828` |
| hidden packet minus same-byte text | `+0.054688` |
| source top-2 oracle | `0.715820` |
| hidden prediction in source top-2 | `0.610352` |
| paired CI95 packet vs source-label | `[-0.086499, -0.012695]` |
| train hidden extraction | `257.74s` |
| eval hidden extraction | `332.30s` |
| total wall time | `598.29s` |

The packet still beats same-byte truncated text and destructive controls, but
it is well below the strict source-label-copy baseline. The paired confidence
interval versus source-label copy is entirely negative.

## Interpretation

This rules out the simple version of "last-layer hidden summaries fix
HellaSwag." The hidden classifier overfits the small train split
(`0.849` fit accuracy, `0.422` internal dev, `0.414` frozen eval), and the
packet faithfully carries that weaker decision.

The important scientific signal is still the source top-2 oracle: the correct
answer is in the source top-2 for `71.6%` of rows, but neither score shape nor
simple hidden summaries identify the switch reliably. A revived hidden branch
must use a substantially different common-basis learner, such as layer sweeps,
CCA/Procrustes/OT alignment, denoising residual objectives, or cross-model
supervision. Do not spend more Mac cycles on a plain last-layer hidden ridge
repair unless the design changes materially.

## Claim Boundary

Safe to claim:

- HellaSwag remains a useful diagnostic because the source has top-2 headroom.
- Score-only, public-receiver-only, and simple last-layer hidden-summary repair
  have now all failed against source-label-copy.
- The current ICLR positive story should not rely on HellaSwag until a
  non-label-copy repair packet beats source-label-copy.

Not safe to claim:

- hidden-state communication works on HellaSwag
- the method beats C2C/KVComm-style latent/cache communication
- raw hidden summaries are a publishable contribution here

## Next Gate

Stop this branch unless the next method is qualitatively different. Highest
value next moves:

1. Promote ARC/OpenBookQA plus train-donor systems as the ICLR core while
   keeping HellaSwag as a diagnostic limitation.
2. If reviving HellaSwag, use a train-only top-2 repair model with richer
   representation alignment, not scalar scores or last-layer hidden ridge.
3. Add NVIDIA/vLLM native systems rows for the existing positive benchmarks.
