# Source-Private Tool-Trace Final Review

- date: `2026-04-30`
- status: final-review polish applied
- live branch: explicit source-private tool-trace packet handoff
- scale rung: large frozen slice plus paper final-review gate

## Starting Status

Current ICLR readiness: scoped positive-method manuscript compiled, but not
acceptance-secure. The main evidence supports a narrow protocol claim; the
remaining risk is reviewer framing rather than another deterministic control.

Current paper story: compact explicit source-private tool-trace packets
communicate hidden diagnostic evidence to a target-side candidate decoder at
the far-left-rate operating point.

Exact blocker: reviewers could read the compiled draft as a synthetic
coded-label lookup with thin related-work framing and vague artifact
provenance.

## Subagent Inputs

Three independent audits converged on the same decision:

- keep the target-decoder row as a smoke ablation, not a main learned-decoder
  claim
- do not spend this cycle on the 160-example MPS target-decoder scale-up
- patch the paper to emphasize protocol scope, low-rate tradeoff, source-
  private controls, and related-work/baseline positioning

## Edits Applied

- Abstract now calls the benchmark an explicit diagnostic-code protocol and
  weakens the target-decoder claim to a model-mediated protocol-decoder sanity
  check.
- Introduction now states that the novelty is not an unknown semantic
  representation, but a rate-capped source-private evidence interface with
  falsification controls.
- Benchmark now explains why the synthetic design is deliberate: deterministic
  hidden evidence, exact ID parity, complete source controls, and byte
  accounting.
- Rate section now frames the result as a far-left-rate operating point, not
  dominance at all budgets.
- Target-decoder section is renamed to `Model-Mediated Protocol Decoder Smoke`
  and explicitly says it is not learned latent bridging.
- Related work is split into source coding, tool/agent communication, prompt
  compression/text relay, and latent/cache/connector communication.
- Limitations now explicitly separate candidate selection from unconstrained
  program repair, raw-log reasoning, real deployment, and learned latent
  transfer.
- Appendix now includes a decisive artifact-manifest table.

## Reference Updates

Added paper-facing citations for:

- distributed indirect source coding with decoder side information
- AutoGen multi-agent handoff
- LLMLingua prompt compression
- C2C and KVComm latent/cache communication competitors
- Repair-R1 test-guided program-repair framing

Reference memo:

- `references/478_source_private_final_review_refs.md`

## Compile Result

Command:

```bash
cd paper/iclr2026
latexmk -pdf -interaction=nonstopmode -halt-on-error source_private_tool_trace.tex
```

Output:

- `paper/iclr2026/source_private_tool_trace.pdf`
- pages: `7`
- size: `212923` bytes

Log audit:

```bash
rg -n "Overfull|undefined|Citation|LaTeX Warning|Package natbib Warning|Warning--" \
  paper/iclr2026/source_private_tool_trace.log
```

Result: no matches.

## Decision

The next exact gate should be `source_private_tool_trace_submission_polish_20260430`:
line-level polish of the compiled PDF, citation metadata verification for the
new 2025 competitor entries, optional appendix example, and final decision on
whether the scoped protocol paper is strong enough without the target-decoder
scale-up.
