# HellaSwag Qwen Hybrid-To-Phi Conditional Acceptor Gate

Date: 2026-05-03

## Status

Current paper readiness: COLM workshop plausible; ICLR full still blocked.
Current story: fixed-byte source-private packets can carry HellaSwag task
evidence under strict destructive controls. Exact ICLR blocker: a learned
receiver/common-basis method must beat packet-only, or the paper needs native
systems evidence against KV/state-transfer baselines.

This gate is a negative target-aware receiver result. It tests whether Phi's
own public score simplex can safely override the fixed Qwen hybrid packet on
the cached Qwen-to-Phi surface.

## Artifact

- `results/source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate_20260503_validation1024_2048/`
- `scripts/build_source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.py`
- `tests/test_build_source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_qwen_hybrid_to_phi_conditional_acceptor_gate.py \
  --run-date 2026-05-03
```

## Result

The gate reserves the first `64` rows per cached Phi slice for fitting one-rule
target-side override candidates, the next `64` rows per slice for selecting a
single frozen rule, and the remaining `384` rows per slice for held-out
evaluation. The selected rule was:

```text
selected_margin <= 0.213526913511
```

| Row | Heldout accuracy | Delta vs fixed hybrid | CI95 low | Helps | Harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| target-or-hybrid oracle | `0.604167` | `+0.136719` | `+0.111979` | `105` | `0` |
| fixed hybrid vote-on-score-agreement | `0.467448` | reference | reference | reference | reference |
| Qwen candidate-only | `0.455729` | `-0.011719` | `-0.023438` | `5` | `14` |
| conditional target acceptor | `0.454427` | `-0.013021` | `-0.028646` | `14` | `24` |
| random same-coverage target override | `0.436198` | `-0.031250` | `-0.046875` | `8` | `32` |
| best source-destroying control | `0.420573` | `-0.046875` | `-0.070313` | `26` | `62` |
| Phi target-only | `0.263021` | `-0.204427` | `-0.251302` | `105` | `262` |

Slice readout:

| Slice | Rows | Conditional acceptor | Fixed hybrid | Delta |
| --- | ---: | ---: | ---: | ---: |
| `1024:1536` | `384` | `0.473958` | `0.486979` | `-0.013021` |
| `1536:2048` | `384` | `0.434896` | `0.447917` | `-0.013021` |

Pass gate: `False`.

## Interpretation

This kills the simple target-score conditional acceptor branch. There is real
oracle headroom: if an oracle could choose between the fixed Qwen hybrid packet
and Phi's target-side answer, accuracy would rise from `0.467448` to
`0.604167`. The selected rule cannot access that headroom. It overrides `60`
held-out rows, creating `14` helps but `24` harms versus fixed hybrid, and is
negative on both cached Phi slices.

Scientifically, the failure is useful: public target confidence and simple
source-packet margins are not enough to decide when a weaker Phi target answer
should replace the stronger Qwen hybrid packet. Future receiver work needs a
new feature source or a genuinely learned common-basis/conditional-innovation
mechanism, not another shallow one-rule acceptor.

Lay explanation: Phi sometimes has an answer that would improve the Qwen hint.
This test trained a small rule for when Phi should override that hint. On new
examples the rule was wrong more often than it was helpful, so simply trusting
Phi under a margin rule makes the system worse.

## Decision

- Keep fixed hybrid vote-on-score-agreement as the current best cross-family
  packet-policy row.
- Mark shallow target-score acceptors as saturated unless a new feature source
  is introduced.
- Use the remaining oracle gap to motivate a stronger receiver/common-basis
  branch, but require it to beat fixed hybrid under the same held-out slices and
  destructive controls.
- Before broad reasoning claims, add an option-order/candidate-permutation
  audit for HellaSwag fixed-hybrid rows, because candidate-id packet claims are
  vulnerable to multiple-choice selector bias.
