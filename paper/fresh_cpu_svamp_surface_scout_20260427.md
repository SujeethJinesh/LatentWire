# Fresh CPU SVAMP Surface Scout

- date: `2026-04-27`
- readiness: not ICLR-ready
- scale-up rung: smoke / source-surface discovery
- decision: no usable CPU-only fresh surface found in two adjacent SVAMP8
  scouts

## Question

While PID `31103` blocks MPS work, can a tiny fresh CPU-only SVAMP surface
produce clean source-needed examples that survive text-relay control and
answer-masking audits?

## Commands

First range:

```bash
./venv_arm64/bin/python scripts/materialize_jsonl_range.py \
  --source data/svamp_1000.jsonl \
  --output results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl \
  --start-index 381 \
  --count 8 \
  --manifest-json results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.manifest.json \
  --manifest-md results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.manifest.md \
  --run-date 2026-04-27
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_generation_baselines.py \
  --eval-file results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl \
  --results-dir results/fresh_cpu_svamp8_answernull_20260427/baselines \
  --methods source target t2t \
  --limit 8 \
  --device cpu \
  --max-new-tokens 64 \
  --source-reasoning-mode brief_analysis \
  --use-chat-template \
  --no-enable-thinking \
  --continue-on-error
```

```bash
./venv_arm64/bin/python scripts/build_source_contrastive_target_set.py \
  --target target=path=results/fresh_cpu_svamp8_answernull_20260427/baselines/target_alone.jsonl,method=target_alone \
  --source source=path=results/fresh_cpu_svamp8_answernull_20260427/baselines/source_alone.jsonl,method=source_alone \
  --control text=path=results/fresh_cpu_svamp8_answernull_20260427/baselines/text_to_text.jsonl,method=text_to_text \
  --min-source-only 1 \
  --date 2026-04-27 \
  --output-json results/fresh_cpu_svamp8_answernull_20260427/source_contrastive_target_set.json \
  --output-md results/fresh_cpu_svamp8_answernull_20260427/source_contrastive_target_set.md
```

```bash
./venv_arm64/bin/python scripts/audit_source_surface_answer_masking.py \
  --results-root results/fresh_cpu_svamp8_answernull_20260427 \
  --date 2026-04-27 \
  --output-json results/fresh_cpu_svamp8_answernull_20260427/answer_masking_audit.json \
  --output-md results/fresh_cpu_svamp8_answernull_20260427/answer_masking_audit.md
```

Second range used the same commands over rows `389..396`, writing under
`results/fresh_cpu_svamp8b_answernull_20260427/`.

## Evidence

| Range | Target | Source | Text Relay | Source-Only | Clean Source-Only | Answer-Unexplained Clean In Pool |
|---|---:|---:|---:|---:|---:|---:|
| SVAMP rows 381-388 | `1/8` | `1/8` | `4/8` | `1` | `0` | `0` |
| SVAMP rows 389-396 | `2/8` | `1/8` | `3/8` | `0` | `0` | `0` |

The first range has one source-only ID, `e38cfdf451eb3fd5`, but it is solved by
the text-relay control and is answer-explained in the masking audit. The second
range has no source-only IDs.

CPU feasibility:

- range 381-388 source elapsed `45.58s`, target `63.42s`, text relay `88.38s`
- range 389-396 source elapsed `49.79s`, target `61.77s`, text relay `88.99s`

## Decision

Two adjacent CPU-only fresh SVAMP8 scouts do not produce a usable clean
answer-masked source surface. Continuing this by brute-force CPU range scanning
is possible but low expected value: each eight-example scout costs about
`3.3` minutes and the first two produce no promotable clean signal.

The next evidence-bearing gate is therefore MPS-gated: clear PID `31103`, then
run a larger fresh same-family surface scout with source-final masking and
strict controls.

## Artifacts

- `results/fresh_cpu_svamp8_answernull_20260427/`
- `results/fresh_cpu_svamp8b_answernull_20260427/`

Hashes:

- `results/fresh_cpu_svamp8_answernull_20260427/svamp_rows381_388.jsonl`:
  `7cc0e9d8388778fca31b7dde69293d5400e0c645f03c21ec5c677a482c50daf1`
- `results/fresh_cpu_svamp8_answernull_20260427/baselines/manifest.json`:
  `e7c755be971dbcbd8b29b0c28296d10edd934911b0b3f5ce30bd496de4d32e32`
- `results/fresh_cpu_svamp8_answernull_20260427/source_contrastive_target_set.json`:
  `bc74b715e6f6fef21aee644fa1bfe3ec925764031d7a0c6f46ac96dcb9cbfacc`
- `results/fresh_cpu_svamp8_answernull_20260427/answer_masking_audit.json`:
  `ce20056a6ca59b27f08ee9ed02c2494d997344c4f7e58ce2d16fc5638c03ed5c`
- `results/fresh_cpu_svamp8b_answernull_20260427/svamp_rows389_396.jsonl`:
  `753b2778348768e3ff6b72cd0c070454ce5baf52a142f8fc0ed3a0db78138280`
- `results/fresh_cpu_svamp8b_answernull_20260427/baselines/manifest.json`:
  `ebb82ae03a766c35bb5f47dfad8bdd742e436cafadc0f0b2c553a504baee5f18`
- `results/fresh_cpu_svamp8b_answernull_20260427/source_contrastive_target_set.json`:
  `f1ab04c2313fd70c452b41076a881f1f1066c20a3ebc6aad8f9473cb3943cf23`
- `results/fresh_cpu_svamp8b_answernull_20260427/answer_masking_audit.json`:
  `843885e38e760b2951e9260e6ee6d9e6112835cc590c0b471de21d465afb29fe`

## Next Gate

First check:

```bash
ps -p 31103 -o pid,ppid,stat,etime,command
```

If PID `31103` is gone, run the stronger-source SVAMP70 scout or KVComm strict
source-control MPS smoke recorded in the ledger. If it remains in `STAT=UE`,
OS/session-level cleanup is required before the next evidence-bearing gate.
