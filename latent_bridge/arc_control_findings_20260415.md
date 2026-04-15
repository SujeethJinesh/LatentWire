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
3. Rerun any old MCQ headline numbers because pre-fix ARC MCQ results used
   the wrong answer-boundary scoring path.
4. Continue using GSM8K generation as a separate reasoning check; that path is
   not affected by the MCQ answer-boundary fix.
