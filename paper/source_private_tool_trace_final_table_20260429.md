# Source-Private Tool-Trace Final Evidence Table

- date: `2026-04-29`
- status: final evidence-table integration for scoped positive method
- live branch: explicit source-private tool-trace packet handoff
- scale rung: large frozen slice, held-out families, seed repeats, reviewer-risk rows

## Claim Boundary

Supported claim:

> Explicit source-private tool-trace packets let a source agent communicate
> hidden execution evidence to a target-side candidate decoder. At a compact
> packet budget, the method beats target-only, target-wrapper/no-source,
> source-destroying controls, answer controls, target-derived controls, and
> same-byte structured text relays, while using far fewer bytes than full
> private log relay.

Unsupported claims:

- raw-log repair inference
- unstructured latent transfer
- universal cross-model communication
- learned target-side neural bridge

## Contribution Map

| Contribution | Evidence row | What it rules out | Current strength |
|---|---|---|---|
| Source-private communication with decoder side information | target-only, wrapper/no-source, zero/shuffled/random/answer controls across 500-example surfaces | target priors, prompt wrappers, answer leakage, and source-free sidecars | large frozen deterministic and model-mediated gates |
| Rate-capped diagnostic packet method | 2-byte deterministic packets; Qwen3/Phi-3 model packets; Gemma 4 E2B n500 packet row | full-log relay as necessary communication channel; high-byte text relay as the only viable interface | large frozen slice, seed repeats, latest-small and non-Qwen local rows |
| Strict reusable evaluation harness | exact-ID parity, candidate-pool recall, bytes/tokens/latency, source-destroying controls, codebook remap | selector leakage, candidate-generation confounds, fixed-codebook memorization | reproducible scripts, manifests, final folder checksums |
| Systems rate frontier | systems summary table; 2-byte packet vs matched-byte text vs full hidden-log relay | full trace relay as necessary at the far-left rate point | `183.2x-186.7x` fewer bytes than full hidden-log relay on deterministic core/held-out surfaces |
| Boundary diagnosis for source emitters | Gemma passes strict prompt; Granite passes weakly; OLMo fails; raw-log/no-trace collapses | universal prompt-invariant source behavior | useful limitation, not a universal claim |

## Model-Mediated Packet Rows

| Surface | Source model | Mode | Matched | Target | Best control | Valid packets | Mean bytes | Delta target 95% CI |
|---|---|---|---:|---:|---:|---:|---:|---:|
| core seed 29 | Qwen3-0.6B | `trace_no_hint` | `0.808` | `0.250` | `0.252` | `0.776` | `1.55` | `[0.516, 0.600]` |
| core seed 29 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.252` | `1.000` | `2.00` | `[0.714, 0.788]` |
| core seed 31 | Qwen3-0.6B | `trace_no_hint` | `0.808` | `0.250` | `0.256` | `0.776` | `1.55` | `[0.516, 0.602]` |
| core seed 31 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.256` | `1.000` | `2.00` | `[0.710, 0.786]` |
| held-out seed 30 | Qwen3-0.6B | `trace_no_hint` | `0.922` | `0.250` | `0.258` | `0.864` | `1.73` | `[0.632, 0.712]` |
| held-out seed 30 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.258` | `1.000` | `2.00` | `[0.710, 0.788]` |
| held-out seed 32 | Qwen3-0.6B | `trace_no_hint` | `0.924` | `0.250` | `0.252` | `0.860` | `1.72` | `[0.634, 0.716]` |
| held-out seed 32 | Phi-3-mini | `trace_no_hint` | `1.000` | `0.250` | `0.252` | `1.000` | `2.00` | `[0.710, 0.786]` |

Negative model row:

- Qwen3 `raw_log_no_trace` returns to `0.250` with `0` valid packets on all
  four `500`-example surfaces.

Post-package local source-emitter rows:

- `google/gemma-4-E2B-it`, MPS n500, strict trace-no-hint:
  `500/500 = 1.000`, target-only `125/500 = 0.250`, best control
  `126/500 = 0.252`, packet valid rate `1.000`, exact-ID parity true, p50
  latency `754 ms`.
- `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B`, CPU n160:
  both reach `160/160 = 1.000` with controls near `0.250`; 0.8B is repeated on
  seeds `29/31`.
- `Qwen/Qwen3.5-4B`, CPU n64:
  `64/64 = 1.000`, controls `16/64 = 0.250`; useful latest-small breadth but
  slower on local CPU.
- `ibm-granite/granite-3.3-2b-instruct`, CPU n160 strict trace-no-hint:
  `101/160 = 0.631` on seeds `29/31`, target-only `0.250`, best control
  `0.250-0.256`; positive but prompt-contract sensitive. Its paired
  raw-log/no-trace seed31 row collapses to `0.250` with zero valid packets.
- `allenai/OLMo-2-0425-1B-Instruct`, MPS n16:
  behavioral negative with zero valid packets.

## Deterministic Control Rows

Representative deterministic surfaces:

- core seed `29`, `500` examples
- held-out seed `30`, `500` examples

At the `2`-byte paper packet budget:

| Surface | Matched packet | Target-only | Wrapper/no-source | Zero-source | Shuffled | Random same-byte | Answer-only | Answer-masked | Target-derived | Full log relay | Full diag |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| core seed 29 | `1.000` | `0.250` | `0.250` | `0.250` | `0.250` | `0.254` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |
| held-out seed 30 | `1.000` | `0.250` | `0.250` | `0.250` | `0.250` | `0.254` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |

## Reviewer-Risk Rows

At the `2`-byte paper packet budget:

| Surface | Matched hidden-log text | JSON relay | Free-text relay | Helper/no-log | Diag-masked log | Expected/actual-masked log | Test-name-masked log |
|---|---:|---:|---:|---:|---:|---:|---:|
| core seed 29 | `0.250` | `0.250` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |
| held-out seed 30 | `0.250` | `0.250` | `0.250` | `0.250` | `0.250` | `1.000` | `1.000` |

At `32` bytes, structured JSON and free-text relays become oracles (`1.000`).
This should be reported as a rate curve: structured text is a valid baseline
when it is allowed enough bytes to carry the diagnostic, but it does not explain
the compact `2`-byte packet result.

## Systems Rows

The consolidated systems artifact is
`results/source_private_systems_summary_20260428/`. It reports deterministic
rate rows, model-produced packet rows, and target-decoder rows.

| Interface | Mean bytes | Mean tokens | Accuracy role |
|---|---:|---:|---|
| model-produced packet, Qwen3 | `1.55-1.73` | approximately `1` | `0.808-0.924` matched |
| model-produced packet, Phi-3 | `2.00` | approximately `1` | `1.000` matched |
| deterministic matched packet | `2.00` | `1` | `1.000` oracle protocol |
| matched-byte hidden-log text | `2-32` | budget-dependent | target-only at compact budgets |
| full hidden-log relay | `366-374` | about `34` | `1.000` oracle text relay |
| full diagnostic text | `14` | `1` | `1.000` oracle diagnostic |

At the deterministic 2-byte point, the packet is `183.2x-186.7x` smaller than
full hidden-log relay and `7.0x` smaller than full diagnostic text while
matched-byte hidden-log/JSON/free-text controls remain at the target floor.
This should be presented as a rate frontier: structured text is a valid oracle
once allowed enough bytes, but it does not explain the compact packet result.

## Candidate-Pool Versus Selector

The deterministic control surfaces have candidate-pool recall `1.000`: the
gold repair candidate is present for every example. The measured improvement is
therefore selector accuracy given a fixed public candidate pool, not candidate
generation recall.

For model-mediated packet rows, validity is reported separately from matched
accuracy. Qwen3 is lower than Phi-3 because some packets are invalid or not
parseable; invalid packets fall back to the target prior.

## Target-Model Decoder Ablation

The main tables use a deterministic protocol decoder so that source-side
communication can be isolated cleanly. A Qwen3 target-decoder ablation reduces
the hand-coded-decoder concern:

| Surface | Target model | Device | N | Matched packet | Target-only | Best control | Valid matched | p50 latency |
|---|---|---|---:|---:|---:|---:|---:|---:|
| core seed 29 | Qwen3-0.6B | MPS | 16 | `0.688` | `0.250` | `0.250` | `1.000` | `1267 ms` |
| holdout seed 30 | Qwen3-0.6B | MPS | 32 | `0.750` | `0.250` | `0.281` | `1.000` | `1315 ms` |
| core seed 29 | Qwen3-0.6B | CPU | 64 | `0.656` | `0.250` | `0.250` | `1.000` | `2182 ms` |
| holdout seed 30 | Qwen3-0.6B | CPU | 64 | `0.719` | `0.250` | `0.266` | `1.000` | `2237 ms` |

The attempted core n160 MPS target-decoder run failed before prediction with an
Apple MPS matmul shape error. The CPU n64 row is therefore the current strongest
local target-decoder confirmation and now passes on both core and held-out
surfaces. It is still an ablation, not the main claim.

## Pass/Fail Summary

Passed:

- strict-small hidden-repair gate
- medium `500`-example core gate
- held-out family `500`-example gate
- four-surface seed repeat
- reviewer-risk rows for matched-byte JSON/free-text, helper/no-log, diagnostic
  masking, and component masking
- bytes/token systems comparison against full hidden-log relay
- Gemma and Granite non-Qwen source-signal ablations: raw-log/no-trace
  collapses to target-only with zero valid packets
- Qwen3 target-decoder ablation up to n64 on CPU, with controls at target floor

Failed or pruned:

- raw-log-without-trace packet generation
- ordinary SVAMP/GSM residual hunting as the near-term paper story
- broad latent-transfer claim from current evidence
- broad MoE/FP8 generalization until Qwen3.6-35B-A3B and FP8 rows pass

## Remaining Risk

The remaining risk is not whether the current scoped method is positive; it is
whether reviewers expect a learned target-side neural decoder rather than a
deterministic protocol decoder. The safest paper framing is to present this as
a rate-capped source-private communication protocol with a reproducible
candidate-decoder benchmark, then list learned target decoders as future work
or a small optional extension if time permits. The second major full-paper risk
is external validity: MoE/FP8 rows are executable through the endpoint runner
but remain unrun because remote execution is currently disallowed.

## Next Gate

`source_private_tool_trace_paper_skeleton_20260429`:

- draft the method, benchmark, results, threat-model, and limitations sections
  around this exact table
- include one figure/table showing the rate curve: 2-byte packet versus
  structured text relay versus full hidden-log relay
