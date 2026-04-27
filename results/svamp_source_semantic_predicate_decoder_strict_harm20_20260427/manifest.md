# SVAMP Source Semantic Predicate Decoder Strict Harm20

- date: `2026-04-27`
- status: `semantic_predicate_decoder_fails_smoke`
- live branch entering run: learned semantic predicates with erasure-aware abstention
- scale-up rung: CPU smoke / branch falsification
- device: CPU-only artifact replay; no MPS jobs were started
- MPS blocker: PID `31103` remained present with `STAT=UE`

## Command

```bash
./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --mode learned_logodds \
  --outer-folds 5 \
  --accept-penalty 0.75 \
  --harm-weight 20.0 \
  --min-live-correct 25 \
  --min-live-clean-source-necessary 2 \
  --min-holdout-correct 10 \
  --min-holdout-clean-source-necessary 1 \
  --max-control-clean-union 0 \
  --max-accepted-harm 0 \
  --date 2026-04-27 \
  --output-dir results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427 \
  --output-predictions-jsonl results/svamp_source_semantic_predicate_decoder_strict_harm20_20260427/predictions.jsonl
```

## Result

- Status: `semantic_predicate_decoder_fails_smoke`.
- Live: `25/70` correct, `5` accepted source sidecars, `3` clean source-necessary IDs, accepted harm `0`, control clean union `0`.
- Holdout: `9/70` correct, `2` accepted source sidecars, `0` clean source-necessary IDs, accepted harm `0`, control clean union `0`.
- Clean live IDs recovered: `2de1549556000830`, `41cce6c6e6bb0058`, `4d780f825bb8541c`.

## Decision

Do not promote. The stricter erasure rule proves target-safe live recovery is
possible on the consumed surface, but the signal does not transfer to the
holdout surface. Treat this as a live-only trace/predicate clue, not a positive
method. Revive only with stronger source surfaces or target-side likelihood /
uncertainty features collected after MPS is safe.

## Artifact Hashes

- `semantic_predicate_decoder.json`: `97a8cd1ba95c1239f0055a82a8bc461c99070bb8aba744bbc47f6b5d2567b53f`
- `semantic_predicate_decoder.md`: `c57a791fa9a137d96ea4a951e76883610cba257076f041b6b73ba86ef16ce46a`
- `predictions.jsonl`: `2fa404726a3cb654ec2de6cba75cfceac3e20fccf7ca2b5b873a687020aa6aed`

## Next Gate

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is absent, run the stronger-source scout in
`paper/postkill_historical_cpu_audit_20260427.md`. If it remains present, MPS
work remains blocked by OS/session state.
