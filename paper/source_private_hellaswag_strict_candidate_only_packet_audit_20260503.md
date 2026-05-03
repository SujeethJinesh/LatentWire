# HellaSwag Strict Candidate-Only Packet Audit

Date: 2026-05-03

## Status

This is a systems/privacy compaction result, not a positive receiver-fusion
result.

Current paper readiness remains: COLM workshop plausible; ICLR full still
blocked. The paper story is that source-private fixed-byte packets can carry
useful task evidence under strict destructive controls. The exact ICLR blocker
is unchanged: we still need a packet-beating receiver/common-basis method or a
native systems row that is strong enough to stand without claiming cross-model
latent reasoning.

## Artifact

- `results/source_private_hellaswag_strict_candidate_only_packet_audit_20260503_validation0_9216/`
- `scripts/build_source_private_hellaswag_strict_candidate_only_packet_audit.py`
- `tests/test_build_source_private_hellaswag_strict_candidate_only_packet_audit.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_strict_candidate_only_packet_audit.py \
  --output-dir results/source_private_hellaswag_strict_candidate_only_packet_audit_20260503_validation0_9216 \
  --run-date 2026-05-03
```

## Result

The audit recomputes all nine `1024`-row strict Qwen HellaSwag slices directly
from each saved `predictions.jsonl` file. The candidate-only decoder exactly
reproduces the prior selected-packet row.

| Quantity | Value |
| --- | ---: |
| Eval rows | `9216` |
| Candidate-only accuracy | `0.525499` |
| Source-rank/index-only control | `0.479384` |
| Score-only control | `0.479384` |
| Best label-copy control | `0.483941` |
| Min delta vs source-rank/index-only | `0.037109` |
| Min CI95 low vs source-rank/index-only | `0.018555` |
| Previous packet | `2B` raw / `5B` framed |
| Candidate-only packet | `1B` raw / `4B` framed |

The pass gate is true. The packet contract is now only:

```text
selected candidate id packed into 2 bits
```

No source text, source KV cache, raw hidden vector, raw score vector, or source
logits are exposed.

## Interpretation

This strengthens the systems accounting row: the strongest strict HellaSwag
surface can be represented as a `1B` raw / `4B` framed candidate-only packet,
while preserving the original positive separation from label-copy,
source-rank/index-only, score-only, zero-hidden, candidate-roll, score-channel
roll, and corrupted-hidden controls.

It also sharpens the reviewer limitation. On this multiple-choice surface, the
positive row is still selected-candidate communication. It should not be
described as a general latent language, a learned receiver, or proof that a
target model reasons from rich source latents.

Lay explanation: we checked whether the previous hint needed two bytes or
whether it was enough to send only which of the four answer choices the source
model picked. Sending just that choice gives exactly the same answers on the
large frozen slice, so the packet can be smaller. This helps the systems table,
but it does not solve model-to-model latent reasoning.

## Decision

- Promote the `1B` raw / `4B` framed strict HellaSwag row as a systems/privacy
  contribution only.
- Keep the source-index/source-rank criticism in the main paper, not as an
  appendix-only caveat.
- Do not spend more Mac time on acceptor variants that merely recover
  candidate-only behavior.
- Next exact ICLR gate: a receiver/common-basis method must beat packet-only
  and explicit source-index/source-rank/score quantization on strict HellaSwag,
  or the systems branch needs native NVIDIA/vLLM/SGLang throughput and memory
  rows against C2C/KVComm/QJL/TurboQuant-style baselines.
