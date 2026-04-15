# ARC Control Findings - 2026-04-15

## Goal

Test whether the small ARC-Challenge gain comes from source-to-target KV
communication or from target-side cache perturbation.

## Sanity Fix

The MCQ RotAlign prefix path tokenized answer choices as `"A"` while
`target_alone` scored `" A"` via `context + " " + choice`. This made the
gate-0 RotAlign control differ from target-alone. The evaluation path now
uses the same space-prefixed choice boundary for prefix-cache scoring.

Sanity check after the fix:

- Target-alone: `0.4286`
- Zero translated KV, gate `0.00`: `0.4286`

This means gate zero is now a valid target-equivalence control.

## Results On `data/arc_challenge_eval_35.jsonl`

All rows use Qwen2.5-0.5B-Instruct as source, Qwen3-0.6B as target, and
`cka_half_seed1`.

| Condition | Best Accuracy | Best Gate | Interpretation |
|---|---:|---:|---|
| Target alone | `0.4286` | n/a | Baseline |
| Text-to-text brief analysis | `0.3429` | n/a | Weaker than target |
| Zero translated KV | `0.4857` | `0.05` / `0.10` | Target-cache attenuation helps |
| Real translated KV | `0.4857` | `0.05` / `0.20` / `0.25` | No gain beyond zero-translated control |

Quick check on `data/arc_challenge_50.jsonl`:

- Target-alone: `0.4200`
- Real translated KV: best `0.4400`
- Zero translated KV: best `0.4600`

This larger-but-still-small check points in the same direction: target-only
attenuation is at least as strong as real source-KV transfer.

Updated phase-3 rerun after explicit zero-byte accounting:

| Condition | Best Accuracy | Best Gate | Latent Bytes | Interpretation |
|---|---:|---:|---:|---|
| Target alone | `0.4200` | n/a | `0.0` | Baseline |
| Text-to-text brief analysis | `0.3600` | n/a | `0.0` | Weaker than target |
| Target attenuation | `0.4600` | `0.05` | `0.0` | Best current ARC control |
| Real translated KV | `0.4400` | `0.25` | `14448.0` | Below target attenuation |

Paired real-vs-target-attenuation comparison on the shared gates:

| Gate | Real Acc | Attenuation Acc | Delta | Real-only | Attenuation-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| `0.15` | `0.4000` | `0.4000` | `+0.0000` | `3` | `3` | `[-0.1000, +0.1000]` | `1.0000` |
| `0.25` | `0.4400` | `0.3400` | `+0.1000` | `7` | `2` | `[-0.0200, +0.2200]` | `0.1824` |

Best-real versus best-attenuation paired check:

| Comparison | Real Acc | Attenuation Acc | Delta | Real-only | Attenuation-only | 95% Bootstrap Delta | McNemar p |
|---|---:|---:|---:|---:|---:|---:|---:|
| real gate `0.25` vs attenuation gate `0.05` | `0.4400` | `0.4600` | `-0.0200` | `3` | `4` | `[-0.1200, +0.0800]` | `1.0000` |

The gate-0.25 real-KV row is directionally interesting but not publishable
evidence yet because the confidence interval still crosses zero and the best
zero-byte attenuation result is higher overall.

## Interpretation

The current ARC improvement is real as a small paired-split effect, but it is
not yet evidence for source-to-target latent communication. A target-only
control that fuses zero translated KV, equivalent to attenuating selected
target KV layers by a small gate, matches the real translated-KV result.

This shifts the immediate project goal:

- Keep RotAlign-KV as the candidate communication method.
- Add target-cache attenuation as a mandatory baseline.
- Claim source communication only when real translated KV beats the
  zero/random translated-KV controls on paired examples.

## Next Experiments

1. Use `target_attenuation_brief` as the explicit zero-byte "target KV
   attenuation" baseline in tables.
2. Rerun ARC with real translated KV versus zero translated KV on a larger
   split and report paired flips.
3. Rerun ARC with held-out gate selection, because choosing the best gate on
   the same 50 examples can still overstate both real and zero-byte controls.
4. Rerun any old MCQ headline numbers because pre-fix ARC MCQ results used
   the wrong answer-boundary scoring path.
5. Continue using GSM8K generation as a separate reasoning check; that path is
   not affected by the MCQ answer-boundary fix.
