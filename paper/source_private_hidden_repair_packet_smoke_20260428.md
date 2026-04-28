# Source-Private Hidden-Repair Packet Smoke

- date: `2026-04-28`
- status: promoted smoke, not yet ICLR-ready
- live branch: source-private hidden-repair evidence packets
- scale rung: smoke

## Question

Can the source-private packet story survive when the private evidence is an
actual hidden execution failure from buggy code, rather than a synthetic
`TRACE_SIG` field?

## Setup

The benchmark freezes `64` Python repair examples from `8` bug families. The
target sees public issue text, buggy code, and a four-candidate repair pool.
The source sees a private hidden-test execution log with actual expected/actual
or exception evidence plus a compact repair diagnostic. The target can only
select a candidate by decoding a rate-capped packet against candidate metadata.

Conditions include target-only, target-wrapper/no-source, matched source
packet, zero-source, shuffled-source, random same-byte, answer-only,
answer-masked, target-derived sidecar, matched-byte structured text, full
hidden-log oracle, and full diagnostic oracle.

## Commands

```bash
./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_smoke.py \
  --examples 64 \
  --candidates 4 \
  --seed 28 \
  --budgets 2,4,8,16,32 \
  --output-dir results/source_private_hidden_repair_packet_smoke_20260428
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/run_source_private_hidden_repair_packet_llm.py \
  --benchmark-jsonl results/source_private_hidden_repair_packet_smoke_20260428/benchmark.jsonl \
  --output-dir results/source_private_hidden_repair_packet_llm_20260428/qwen3_0_6b_helper \
  --model Qwen/Qwen3-0.6B \
  --device mps \
  --dtype float32 \
  --limit 64 \
  --seed 28 \
  --max-new-tokens 8 \
  --no-enable-thinking
```

## Results

Deterministic source packets passed at every budget:

| Budget bytes | Matched | Best no-source | Best control | Matched text | Full log |
|---:|---:|---:|---:|---:|---:|
| 2 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 4 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 8 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 16 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |
| 32 | 1.000 | 0.250 | 0.250 | 0.250 | 1.000 |

Qwen3 model-produced helper-line packets also passed:

| Condition | Correct | Accuracy | Mean bytes | Packet valid rate |
|---|---:|---:|---:|---:|
| target_only | 16/64 | 0.250 | 0.00 | n/a |
| matched_model_packet | 64/64 | 1.000 | 2.00 | 1.000 |
| zero_source | 16/64 | 0.250 | 0.00 | n/a |
| shuffled_model_packet | 16/64 | 0.250 | 2.00 | n/a |
| random_same_byte | 16/64 | 0.250 | 2.00 | n/a |
| answer_only | 16/64 | 0.250 | 2.00 | n/a |
| answer_masked | 16/64 | 0.250 | 0.00 | n/a |
| target_derived_sidecar | 16/64 | 0.250 | 2.00 | n/a |
| full_diag_oracle | 64/64 | 1.000 | 2.00 | n/a |

The model-packet gate reports matched minus best no-source `0.750` and matched
minus best source-destroying control `0.750`.

## Interpretation

This is the first positive surface in the pivot where the private source signal
comes from actual code execution artifacts. The matched 2-byte packet recovers
the hidden repair candidate, and the gain disappears under zero, shuffled,
random, answer-only, answer-masked, and target-derived controls.

The result is still protocol-assisted. The private log contains a helper-line
repair diagnostic and the candidate pool exposes matching diagnostic metadata.
The claim is therefore narrow: compact source-private hidden-test packets can
mediate repair-candidate selection under a controlled protocol. It is not yet a
general code-repair, no-helper extraction, or latent communication claim.

## Decision

Promote the hidden-repair packet branch from deterministic smoke to
model-mediated smoke on Qwen3. The next gate should test whether the same
protocol survives cross-model source emitters and then remove or weaken the
helper-line dependency.

## Next Gate

`source_private_hidden_repair_packet_cross_model_20260428`:

- run the same frozen `64` examples on Qwen2.5-0.5B, Qwen3-0.6B, Phi-3-mini,
  and TinyLlama as a negative capability row
- keep target-only, zero-source, shuffled-source, random same-byte,
  answer-only, answer-masked, and target-derived controls
- promote only if at least two capable instruction models pass and controls
  remain at no-source
- record packet validity, bytes, token count, and p50 source latency
