# HellaSwag Qwen Hybrid-To-Phi Cross-Family Gate

Date: 2026-05-03

## Status

This is a cross-family packet-policy survival result, not a learned receiver or
common-basis result.

Current paper readiness remains: COLM workshop plausible; ICLR full still
blocked. The current story is that source-private fixed-byte packets can carry
task evidence under strict controls. The exact ICLR blocker is still a learned
receiver/common-basis method that beats packet-only, broader cross-family
stability, or native systems evidence against KV/state-transfer baselines.

## Artifact

- `results/source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate_20260503_validation1024_2048/`
- `scripts/build_source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate.py \
  --output-dir results/source_private_hellaswag_qwen_hybrid_to_phi_cross_family_gate_20260503_validation1024_2048 \
  --bootstrap-samples 5000 \
  --train-prefix-rows-per-slice 128 \
  --run-date 2026-05-03
```

## Result

The gate evaluates the fixed Qwen hybrid vote-on-score-agreement packet policy
on the cached Phi-3 HellaSwag receiver-family surface. It uses the same heldout
rows as the previous Qwen-strict-to-Phi receiver scout: two contiguous
`512`-row slices, reserving the first `128` rows per slice as train-prefix
rows for comparability with the failed receiver gate and scoring the remaining
`768` heldout rows.

| Row | Heldout accuracy | Delta vs Qwen candidate-only | CI95 low |
| --- | ---: | ---: | ---: |
| Qwen hybrid packet | `0.467448` | `+0.011719` | `+0.001302` |
| Qwen candidate-only packet | `0.455729` | reference | reference |
| trained-label control | `0.414062` | `-0.041667` | `-0.066406` |
| source-label / source-rank / score-only controls | `0.411458` | `-0.044271` | `<= -0.065104` |
| Phi target-only | `0.263021` | `-0.192708` | `-0.238281` |
| score-channel-roll hidden control | `0.257812` | `-0.197917` | `-0.250000` |

Additional readout:

- Qwen hybrid vs Phi target-only: delta `+0.204427`, CI95 low `+0.160156`.
- Qwen candidate-only vs Phi target-only: delta `+0.192708`, CI95 low
  `+0.145833`.
- Target-or-candidate-only oracle: `0.593750`.
- Target-or-hybrid oracle: `0.604167`.

Slice readout:

| Slice | Phi target | Candidate-only | Hybrid | Hybrid - candidate-only |
| --- | ---: | ---: | ---: | ---: |
| `1024:1536` | `0.270833` | `0.473958` | `0.486979` | `+0.013021` |
| `1536:2048` | `0.255208` | `0.437500` | `0.447917` | `+0.010417` |

Pass gate: `True`.

## Interpretation

This resolves the immediate cross-family pressure on the new hybrid policy:
the fixed Qwen hybrid packet improves over Qwen candidate-only on cached Phi
heldout rows while retaining the same receiver-visible one-candidate packet
contract. The result also stays above source-label, source-rank/index,
score-only, wrong-example hidden, candidate-roll, zero-hidden, and
score-channel-roll controls.

This should be framed narrowly. Phi does not consume a learned latent, soft
prefix, SAE feature, score simplex, or KV/cache state. It receives one final
candidate id emitted by Qwen's source-side policy. The result therefore
strengthens the packet-policy and cross-family transfer story but does not
solve model-to-model latent reasoning.

Scientifically, this says the Qwen hybrid rule is not merely overfit to a
same-family Qwen receiver surface. But the large target-or-hybrid oracle still
shows room for a real Phi receiver to use target evidence without destroying
packet utility.

Lay explanation: we checked whether the improved Qwen hint still helps when
the receiving model is Phi instead of Qwen. On the cached Phi rows, the fixed
hybrid hint beats both Phi's own answer and the older Qwen candidate-only hint.
The caveat is that Phi is still receiving an answer-choice hint, not a rich
hidden thought.

## Decision

- Promote this as cross-family survival for the fixed hybrid packet policy.
- Do not promote it as learned receiver fusion or a common latent language.
- Keep the failed Qwen-strict-to-Phi receiver gate in the paper as an important
  limitation: the generic Phi receiver still trails packet-only.
- The next exact Mac-local method gate should try a packet-preserving anti-harm
  veto or target-loss query/soft-prefix connector that can beat this hybrid row
  without exposing source text, source KV, raw hidden vectors, raw scores, or
  logits.
