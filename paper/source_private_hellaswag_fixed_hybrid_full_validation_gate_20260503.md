# HellaSwag Fixed Hybrid Full-Validation Gate

Date: 2026-05-03

## Status

This is an evaluation-strengthening result for the current packet policy, not
a learned receiver/common-basis result.

Current paper readiness: COLM workshop plausible; ICLR full still blocked.
Current story: source-private fixed-byte packets can carry HellaSwag task
evidence under strict destructive controls. Exact ICLR blocker: a receiver or
common-basis method must beat packet-only, or we need native systems evidence
against KV/state-transfer baselines.

## Artifact

- `results/source_private_hellaswag_fixed_hybrid_full_validation_gate_20260503_validation0_10042/`
- `scripts/build_source_private_hellaswag_fixed_hybrid_full_validation_gate.py`
- `tests/test_build_source_private_hellaswag_fixed_hybrid_full_validation_gate.py`

## Command

```bash
./venv_arm64/bin/python scripts/build_source_private_hellaswag_fixed_hybrid_full_validation_gate.py \
  --run-date 2026-05-03
```

## Result

The fixed hybrid vote-on-score-agreement packet policy extends from the prior
strict `0:9216` surface to the full cached HellaSwag validation range
`0:10042`, including the terminal tail `9216:10042`.

| Row | Accuracy | Delta vs candidate-only | CI95 low | Helps | Harms |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed hybrid vote-on-score-agreement | `0.532464` | `+0.005776` | `+0.002888` | `139` | `81` |
| candidate-only packet | `0.526688` | reference | reference | reference | reference |
| trained-label control | `0.484565` | `-0.042123` | `-0.049592` | `502` | `925` |
| source-label / source-rank / score-only / zero-hidden controls | `0.480880` | `-0.045808` | `<= -0.051583` | `257` | `717` |
| wrong-example hidden control | `0.452599` | `-0.074089` | `-0.082354` | `571` | `1315` |
| candidate-roll hidden control | `0.416152` | `-0.110536` | `-0.119199` | `487` | `1597` |
| score-channel-roll hidden control | `0.253933` | `-0.272754` | `-0.286696` | `1523` | `4262` |

Slice readout:

| Slice | Rows | Candidate-only | Fixed hybrid | Delta |
| --- | ---: | ---: | ---: | ---: |
| `0:1024` | `1024` | `0.512695` | `0.518555` | `+0.005859` |
| `1024:2048` | `1024` | `0.454102` | `0.465820` | `+0.011719` |
| `2048:3072` | `1024` | `0.479492` | `0.485352` | `+0.005859` |
| `3072:4096` | `1024` | `0.531250` | `0.537109` | `+0.005859` |
| `4096:5120` | `1024` | `0.538086` | `0.541992` | `+0.003906` |
| `5120:6144` | `1024` | `0.555664` | `0.562500` | `+0.006836` |
| `6144:7168` | `1024` | `0.555664` | `0.556641` | `+0.000977` |
| `7168:8192` | `1024` | `0.556641` | `0.558594` | `+0.001953` |
| `8192:9216` | `1024` | `0.545898` | `0.553711` | `+0.007812` |
| `9216:10042` | `826` | `0.539952` | `0.547215` | `+0.007264` |

Pass gate: `True`.

## Interpretation

This closes the cached full-validation evaluation gap for the current fixed
hybrid packet policy. The same deterministic policy improves candidate-only on
all ten contiguous validation slices, including the terminal tail that had
previously been discussed as unresolved for stricter hidden-innovation
jackknife criteria.

The result strengthens the current HellaSwag packet contribution:

- receiver-visible payload remains `1B` raw / `4B` framed;
- no source text, source KV, raw hidden vectors, raw score vector, or source
  logits are exposed;
- destructive/source-score controls remain well below the fixed hybrid packet.

It does not solve the main ICLR blocker. The receiver still sees one final
source candidate id emitted by a fixed source-side policy. This should be
presented as full-validation packet-policy evidence, not as model-to-model
latent reasoning.

Lay explanation: we checked the final cached HellaSwag examples that were not
part of the previous large strict surface. The same tiny hybrid answer hint
still helps on that tail and on the full validation set, so the packet result
is not just from the first `9216` examples.

## Decision

- Promote fixed hybrid vote-on-score-agreement as the strongest HellaSwag
  packet-policy row over full cached validation `0:10042`.
- Keep the current ICLR gap explicit: this is not learned receiver fusion or a
  common latent language.
- Next method gate should target the remaining oracle gap with a target-loss
  score-simplex conditional innovation receiver or another common-basis method
  that must beat packet-only under source-destroying controls.
