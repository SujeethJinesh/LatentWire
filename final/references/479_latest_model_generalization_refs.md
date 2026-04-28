# Latest-Model Generalization References

Date: `2026-04-28`

## Blocker Helped

The current paper package has strong evidence for Qwen3/Phi-3/Qwen2.5-era source
packet emitters, but it does not yet prove generalization to the newest
Qwen3.5/Qwen3.6 hybrid and MoE releases. This memo identifies the newest
candidate rows and the harness dependency blocker before claiming MoE or latest
model generalization.

## Sources And Experiment Implications

- Qwen3.6-35B-A3B, https://huggingface.co/Qwen/Qwen3.6-35B-A3B
  - Helps: tests whether source-packet emission survives a sparse MoE source
    model with `35B` total parameters and `3B` activated parameters.
  - Mechanism/design idea: the packet task should depend on instruction-following
    and private evidence extraction, not dense-vs-MoE internals.
  - Next experiment change: add an off-machine `n=32` MoE smoke, then widen to
    `n=500` only if controls remain at target-only.
  - Role: cross-family/future model generalization.
- Qwen3.6-35B-A3B-FP8, https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
  - Helps: separates MoE generalization from deployment quantization effects.
  - Mechanism/design idea: FP8 should preserve the two-character diagnostic
    packet if the model retains exact copying/instruction following.
  - Next experiment change: run the same off-machine `n=32` smoke as the BF16
    MoE row; compare valid-packet rate, latency, and controls.
  - Role: systems/quantization robustness row.
- Qwen3.5-0.8B, https://huggingface.co/Qwen/Qwen3.5-0.8B
  - Helps: smallest latest Qwen3.5 source-emitter candidate.
  - Mechanism/design idea: hybrid long-context small model should be a cheap
    local first-pass test after dependency compatibility is solved.
  - Next experiment change: upgrade the repo-local Transformers stack to a
    `qwen3_5`-capable version, then run an `n=16` compatibility smoke.
  - Role: latest-small-model falsification row.
- Qwen3.5-2B, https://huggingface.co/Qwen/Qwen3.5-2B
  - Helps: tests whether the latest small-family row scales beyond the `0.8B`
    candidate.
  - Mechanism/design idea: if `0.8B` is unstable, `2B` may recover exact packet
    following while remaining local-feasible.
  - Next experiment change: run after the same Transformers compatibility gate;
    promote to `n=64` only if `n=16` passes.
  - Role: latest-small-model confirmation.
- Qwen3.5-4B, https://huggingface.co/Qwen/Qwen3.5-4B
  - Helps: upper small-model local row before moving to remote 27B/MoE models.
  - Mechanism/design idea: checks if packet emission improves with capacity at
    manageable local scale.
  - Next experiment change: run only after `0.8B`/`2B`, due local memory risk.
  - Role: capacity scaling row.
- Qwen3.5-35B-A3B, https://huggingface.co/Qwen/Qwen3.5-35B-A3B
  - Helps: adds a Qwen3.5-generation sparse MoE row instead of relying only on
    Qwen3.6 for MoE evidence.
  - Mechanism/design idea: same explicit packet protocol should be insensitive
    to sparse routing if exact diagnostic extraction is preserved.
  - Next experiment change: run after Qwen3.6 MoE or as a fallback if Qwen3.6
    serving is unavailable.
  - Role: MoE cross-generation confirmation.
- Qwen3.5-35B-A3B-FP8, https://huggingface.co/Qwen/Qwen3.5-35B-A3B-FP8
  - Helps: checks whether Qwen3.5 MoE packet emission survives FP8 deployment.
  - Mechanism/design idea: compare valid-packet rate against the non-FP8 MoE
    row.
  - Next experiment change: remote/API `n=32` after the BF16 MoE row.
  - Role: quantization robustness.
- Qwen3.6-27B, https://huggingface.co/Qwen/Qwen3.6-27B
  - Helps: dense comparator for Qwen3.6, separating family freshness from MoE
    routing.
  - Mechanism/design idea: if dense 27B works but MoE fails, routing or
    quantized serving may be the issue; if both work, the protocol generalizes.
  - Next experiment change: remote/API `n=32` dense comparator.
  - Role: cross-family control.

## Consequence For The Paper

Do not claim MoE or latest-model generalization yet. The right next gate is a
compatibility/harness update, not another paper-story edit:

1. Update the repo-local Transformers dependency so `qwen3_5` configs load.
2. Run `Qwen/Qwen3.5-0.8B` and `Qwen/Qwen3.5-2B` `n=16` smokes.
3. If small rows pass, run `Qwen/Qwen3.6-35B-A3B` and FP8 on a remote GPU/API
   at `n=32`.
4. Promote only if matched packets beat target/control floor and all
   source-destroying controls stay near target-only.

## Local Result Update

Update `2026-04-28`: Qwen3.5 small rows now have local evidence after upgrading
the repo-local Transformers stack. `Qwen/Qwen3.5-0.8B` passes CPU n160 with a
seed repeat, `Qwen/Qwen3.5-2B` passes CPU n160, and `Qwen/Qwen3.5-4B` passes CPU
n16. The 4B row is useful as a capacity-scaling smoke, but its p50 CPU packet
latency is about `32.5s`, so the next higher-value gate remains off-machine
Qwen3.6 MoE/FP8 n32 unless local n64 time is acceptable.
