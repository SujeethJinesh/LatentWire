# Query-Memory Answer-Likelihood CPU Sweeps - 2026-04-26

## Status

- ICLR readiness: not ready
- estimated distance: one deployable source-derived method plus strict small,
  medium, seed-repeat, systems, and cross-family gates
- current story: historical RotAlign/query-memory rows provide mechanism
  clues, but the live target-memory/query-memory family remains
  control-explained under gold-answer continuation likelihood
- exact blocker: matched source does not robustly beat zero-source,
  shuffled-source, target-only, and slots-only controls
- current live branch: killed as a strict positive method candidate
- highest-priority next gate: clear MPS PID `31103`, then reset to source
  surface/interface discovery rather than tuning another Perceiver memory
  checkpoint

## Runs

All runs used `latent_bridge/evaluate.py` on CPU with `--methods rotalign`,
fixed gate `0.15`, `--kv-transport k_only`, chat templates enabled, thinking
disabled, and `max_new_tokens=8`. Controls were matched, zero-source,
shuffled-source salt `1`, `target_only`, and `slots_only` whenever the
checkpoint supported target-conditioned memory.

| Run | n | Status | Matched Mean | Best-Control Delta | Best Wins |
|---|---:|---|---:|---:|---:|
| SVAMP32 delta-memory | 4 | fail | -7.673776 | -0.141012 | 0/4 |
| SVAMP70 Perceiver answer-teacher | 4 | fail | -7.261671 | -0.112360 | 0/4 |
| Qwen2.5-Math Perceiver 4 clean IDs | 4 | pass | -7.989116 | +0.080362 | 3/4 |
| Qwen2.5-Math Perceiver 6 clean IDs | 6 | fail | -8.195434 | -0.090384 | 4/6 |

## Evidence

The SVAMP32 delta-memory checkpoint beats `target_only` and `slots_only`, but
loses to `zero_source` on mean answer likelihood. That is a decisive source
attribution failure.

The SVAMP70 Perceiver answer-teacher checkpoint has no hidden answer-likelihood
signal: matched loses to shuffled-source, target-only, and slots-only controls,
and wins `0/4` against the best runnable control.

The Qwen2.5-Math Perceiver checkpoint produced the only positive clue: on a
4-clean-ID CPU smoke, matched source beat every control and won `3/4` against
the best control. The required clean6 expansion failed immediately. Matched
still beats zero-source, but shuffled-source, target-only, and slots-only match
or beat it on mean answer likelihood.

## Decision

Kill the target-memory/query-memory Perceiver family as the current live
positive-method branch. The clean6 failure means the 4-ID pass is not
promotable, and the repeated pattern across delta-memory, SVAMP70 Perceiver,
and Qwen2.5-Math Perceiver is target/slot/shuffle control dominance.

This does not kill all future source-memory ideas. It does kill continuing to
tune fixed gate, answer-teacher weight, anti-memory weight, query count, or
bridge rank on these exact Perceiver/delta-memory checkpoints without a new
source-interface hypothesis.

## Artifacts

- `results/svamp32_deltamem_answer_likelihood_cpu_smoke_20260426/manifest.md`
- `results/svamp70_perceiver_answer_likelihood_cpu_smoke_20260426/manifest.md`
- `results/qwen25math_svamp32_perceiver_answer_likelihood_cpu_smoke_20260426/manifest.md`
- `results/qwen25math_svamp32_perceiver_answer_likelihood_clean6_cpu_20260426/manifest.md`

## Next Gate

Hard blocker first: PID `31103` is still an orphaned MPS calibration process in
`STAT=UE` and ignores `SIGKILL`. Do not start more MPS jobs until it is cleared.

I ran the existing-artifact source surface re-scan locally because it does not
need MPS. Exact command:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_source_headroom_surfaces.py \
  --surface svamp70_live_source=target_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_live_source \
  --surface svamp70_holdout_source=target_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_holdout_source \
  --surface svamp70_chal171_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal171_240_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal171_240_source \
  --surface svamp70_chal241_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal241_310_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal241_310_source \
  --surface svamp70_chal311_source=target_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_svamp70_surface_scout_chal311_380_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp70_chal311_380_source \
  --surface gsm70_math_source=target_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/target_alone.jsonl,source_path=results/qwen25math_qwen3_gsm70_source_surface_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_gsm70_source \
  --surface svamp32_math_chat_source=target_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/target_alone.jsonl,source_path=results/surface_scout_qwen25math_qwen3_svamp32_chat_20260426/source_alone.jsonl,target_method=target_alone,source_method=source_alone,note=qwen25math_qwen3_svamp32_chat_source \
  --min-source-only 6 \
  --output-json results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.json \
  --output-md results/source_headroom_surface_scan_20260426/scan_after_query_memory_prune.md
```

Result:

- `svamp70_live_source`: strong, target `21/70`, source `13/70`,
  source-only `9`, oracle `30/70`
- `svamp70_holdout_source`: strong, target `8/70`, source `8/70`,
  source-only `6`, oracle `14/70`
- all adjacent SVAMP70 scouts, GSM70, and SVAMP32 remain below the source-only
  threshold

Next branch selection: the only reusable decision surface is still
`svamp70_live_source` with immediate `svamp70_holdout_source` validation. Do
not reuse fixed decoded guards, shallow source-text routers, tiny prefix
emitters, or Perceiver target-memory checkpoints. The next method branch must
be a materially different rate-capped source interface on this live/holdout
surface, or a new source/target scout once MPS is clear.

Immediate resume command before any new MPS work:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

Proceed only if PID `31103` is gone or no longer the stuck
`scripts/calibrate.py --device mps` process.
