# Source-Private Full-Paper Contribution Review

- date: `2026-04-28`
- status: scoped positive method is strong; broad MoE/FP8 and latent-transfer claims remain unproven
- live method: source-private diagnostic repair packets with target-side candidate side information

## Current Readiness

The project has enough evidence for a strong workshop paper and a plausible
full-paper submission if the claim is scoped carefully. The full-paper risk is
not whether the packet method works on the current benchmark; it does. The risk
is whether reviewers see a general technical contribution rather than a narrow
diagnostic-code benchmark.

## Paper Story

The central claim should be:

> Cross-agent communication should be tested under decoder side information:
> the target has public task context and candidate state, the source has private
> evidence, and only a rate-capped source packet may cross the boundary.

The method is a diagnostic packet protocol. The source reads private
tool/test/log evidence and emits a compact `REPAIR_DIAG` packet. The target
uses public candidates plus the packet to select the repair. Source-destroying
controls show whether the gain is real source-private communication.

## Three Core Technical Contributions

1. **Source-private communication formulation with decoder side information.**
   The benchmark separates public prompt/candidate state from private
   source-only evidence. This maps the problem to source coding with side
   information rather than ordinary prompt relay. The contribution is the
   threat model and evaluation target: gains must disappear when the private
   source evidence is removed, shuffled, randomized, answer-only, answer-masked,
   or replaced with target-derived sidecars.

2. **Rate-capped diagnostic packet method.**
   `REPAIR_DIAG` is a compact, interpretable message family that transmits the
   source-private residual needed by a target-side candidate decoder. It is not
   a verbose chain-of-thought relay. The large rows show the method can recover
   the private signal while target-only and source-destroying controls remain at
   the candidate prior. The codebook-remap gate shows the method is not tied to
   one fixed diagnostic vocabulary.

3. **Strict source-destroying benchmark and reproducibility harness.**
   The harness preserves exact ID parity, logs bytes/latency/token counts, runs
   matched-byte and diagnostic-masked controls, and reports pass/fail gates. This
   is a reusable evaluation contribution for future cross-agent communication
   work, because it catches target-prior, formatting, answer-leakage, and
   selector-artifact explanations.

## Secondary Contributions

- **Large-slice and seed-stability evidence.** Qwen/Phi rows clear 500-example
  frozen slices; Qwen3.5 small rows add latest-small breadth; Gemma 4 E2B now
  clears a strict-prompt n500 local row with MPS latency around `754 ms`.
- **Cross-family prompt-contract diagnosis.** Gemma is clean and seed-stable;
  Granite passes but is weaker; OLMo is a behavioral negative. This is more
  useful than hiding failures because it identifies packet-emitter instruction
  following as a necessary condition.
- **Endpoint runner for future MoE/FP8 gates.** The same evaluator can consume
  an OpenAI/vLLM-compatible source model endpoint, so Qwen3.6 MoE and FP8 rows
  require access rather than new benchmark code.

## Reviewer Critique

The strongest likely reviewer objections are:

- **"This is a codebook/candidate-selection benchmark, not communication."**
  Response: emphasize source-private residual transfer, source-destroying
  controls, codebook remapping, and matched public candidate parity. Do not call
  it broad semantic communication.
- **"Structured text relay could do this."**
  Response: report matched-byte structured text and full-evidence oracles. The
  paper should claim a low-rate systems tradeoff, not that text relay is
  impossible.
- **"The method depends on prompt compliance."**
  Response: accept this and present prompt-contract sensitivity as a diagnostic.
  Gemma/Qwen are strong emitters, Granite is partial, OLMo fails. This bounds
  the claim honestly.
- **"Where are MoE and quantized models?"**
  Response: MoE/FP8 remains the top external-validity gap. The endpoint harness
  is ready, but the paper should not claim MoE until Qwen3.6-35B-A3B and FP8
  pass under the same controls.
- **"Where is the connection to latent transfer?"**
  Response: frame this as a first positive source-private communication method,
  not as latent-vector transfer. Latent/cache/adaptor methods remain future
  work or baselines unless they clear the same controls.

## Highest-Priority Next Gates

1. **MoE/FP8 falsification when allowed.** Run `Qwen/Qwen3.6-35B-A3B` endpoint
   n32, then FP8 n32. Pass requires matched `>= best_no_source + 0.15`, all
   source-destroying controls within `0.02` of no-source, exact-ID parity, and
   packet validity.
2. **Local non-Qwen repeat if remote remains unavailable.** Repeat Gemma n500 on
   seed31 or run Granite trace-no-hint n160 seed31. Gemma seed31 is expected to
   pass; Granite seed31 is more discriminative because it tests the weaker
   prompt-contract boundary.
3. **Paper-facing ablation table.** Build a compact table that groups target,
   matched packet, structured text, raw-log/no-trace, codebook remap, and latest
   model rows. This will make the three contributions obvious to reviewers.

## Recommendation

Keep the full-paper claim scoped:

> We introduce and evaluate rate-capped source-private diagnostic packets for
> cross-agent communication with decoder side information.

Do not claim universal cross-model communication, latent transfer, or MoE
generalization yet. The paper becomes substantially stronger if the next
available gate adds either Qwen3.6 MoE/FP8 evidence or a reviewer-ready
consolidated ablation table showing the three contributions side by side.
