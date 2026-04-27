# Source Candidate Sidecar Materializer

- date: `2026-04-27`
- status: `failed_smoke`
- scale-up rung: `smoke / strict-small preparation`
- live branch decision: kill this heuristic materializer; keep the no-leak sidecar tooling
- base commit before run: `2deef3b445219036016ac4092617a1208f0fb70a`

## Question

Can a frozen, CPU-only source-derived candidate-score sidecar over target-side
candidate values recover clean source-necessary SVAMP examples while preserving
target-correct examples and surviving sidecar-shaped controls?

## Implementation

Added `scripts/materialize_svamp_source_candidate_sidecars.py`.

The materializer emits `candidate_scores` JSONL sidecars compatible with
`scripts/analyze_svamp_source_semantic_predicate_decoder.py`. It uses source
trace features, but only scores values already present in the target-side
candidate pool; it never adds source-only answers to the receiver pool.

The manifest records command, git commit, input paths, hashes for sidecar JSONL
files, target-set summaries, and byte budget.

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_materialize_svamp_source_candidate_sidecars.py tests/test_analyze_svamp_source_semantic_predicate_decoder.py -q
./venv_arm64/bin/python -m py_compile scripts/materialize_svamp_source_candidate_sidecars.py scripts/analyze_svamp_source_semantic_predicate_decoder.py tests/test_materialize_svamp_source_candidate_sidecars.py
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/materialize_svamp_source_candidate_sidecars.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --output-dir .debug/source_candidate_sidecars_20260427 \
  --sidecar-bits 8 \
  --date 2026-04-27
```

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp_source_semantic_predicate_decoder.py \
  --live-target-set results/qwen25math_qwen3_svamp70_source_surface_20260426/source_contrastive_target_set.json \
  --holdout-target-set results/qwen25math_qwen3_svamp70_holdout_source_surface_20260426/source_contrastive_target_set.json \
  --live-sidecar-jsonl .debug/source_candidate_sidecars_20260427/live_candidate_sidecars.jsonl \
  --holdout-sidecar-jsonl .debug/source_candidate_sidecars_20260427/holdout_candidate_sidecars.jsonl \
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
  --output-dir .debug/source_candidate_sidecar_gate_20260427 \
  --output-predictions-jsonl .debug/source_candidate_sidecar_gate_20260427/predictions.jsonl
```

## Result

Sidecar materialization:

- live: `70` examples, `1` byte/example, source final in target pool `43/70`,
  source-mentioned target-pool hits `59/70`, top labels `target=62`, `t2t=8`
- holdout: `70` examples, `1` byte/example, source final in target pool
  `45/70`, source-mentioned target-pool hits `56/70`, top labels `target=63`,
  `t2t=7`

Decoder gate:

- live matched: `21/70`, accepted `0`, clean source-necessary `0`, accepted
  harm `0`, control clean union `0`
- holdout matched: `11/70`, accepted `7`, clean source-necessary `0`,
  accepted harm `1`, control clean union `0`
- status: `semantic_predicate_decoder_fails_smoke`

The heuristic mostly ranks fallback target-side candidates. It does not produce
clean source-necessary recovery under the hardened target-pool decoder.

## Artifact Hashes

- `.debug/source_candidate_sidecars_20260427/manifest.json`:
  `47c58449496b4983923879dbe466effe83e996ef5a13019be342c21e548dd722`
- `.debug/source_candidate_sidecars_20260427/manifest.md`:
  `d936d08113654d1dbae76ab4d944f4a0f0316ec4a1f68bda45eb1a38173e5150`
- `.debug/source_candidate_sidecars_20260427/live_candidate_sidecars.jsonl`:
  `0378c86287b88edacfe2ae1fd61418d24e10180d99d562b57fb8b2a1574f06dd`
- `.debug/source_candidate_sidecars_20260427/holdout_candidate_sidecars.jsonl`:
  `267abcfd7efa67dd702e9a362a2fdae357084e03b308d3496548141d4ce54b83`
- `.debug/source_candidate_sidecar_gate_20260427/semantic_predicate_decoder.json`:
  `ac5e5e2a9ac60453504fac7c9139fcc788e7e53d8704f4eb7f476b08d072a792`
- `.debug/source_candidate_sidecar_gate_20260427/semantic_predicate_decoder.md`:
  `ec4d14f179c671db63a6f0d8f888c68ab58813301cb81f9c9404156c92b436e2`
- `.debug/source_candidate_sidecar_gate_20260427/predictions.jsonl`:
  `43b4824e0d82a40b2a77e8833dce7943ec9f929e934d0e9ba4479eec6ffec30a`

## Artifact Audit Addendum

The artifact audit found that strict target-mentioned headroom on the canonical
SVAMP70 target pool is small:

- live target baseline `21/70`; perfect strict selector over target-mentioned
  values adds only `2` IDs, for `23/70`
- holdout target baseline `8/70`; perfect strict selector adds `4` IDs, for
  `12/70`

The broader text-to-text/C2C candidate pool has much larger oracle headroom,
but it is not clean target-only evidence. Future sidecars over that pool need
frozen source-derived scoring plus zero-source, shuffled-source, target-only,
slots-only, random sidecar, and same-byte controls.

## Decision

Kill this heuristic source-candidate materializer as a method branch. It is
useful tooling only.

Next branch: a frozen model-scored sidecar producer that scores target-side
candidate values under condition-specific source prompts and writes the same
`candidate_scores` schema. The next gate must run live and holdout with
source-destroying controls and must not count broader C2C/text candidate oracle
headroom as communication unless the source-derived sidecar survives controls.

Current blocker: MPS remains blocked by orphaned PID `31103` in `STAT=UE`, so
the next full model-scored gate should wait for OS/session cleanup or run a
very small CPU smoke only for plumbing.

## Frozen Model-Scored Follow-Up

Added `scripts/collect_svamp_frozen_candidate_score_sidecar.py` and
`tests/test_collect_svamp_frozen_candidate_score_sidecar.py`.

This stricter producer scores only target-side candidate-pool values with a
source model and emits no gold answers, correctness fields, or source-only
candidate values. Fake-model unit tests verify next-token scoring, leakage
exclusion, source-only value exclusion, and deterministic schema output.

Two-example CPU plumbing smoke:

- output JSONL:
  `.debug/frozen_candidate_score_sidecar_smoke_20260427/live_limit2.jsonl`
- JSONL sha256:
  `15227350a56e5bf9d26143c108fcae598b5cdffa78a07301f44f6c6ed852ce7c`
- markdown sha256:
  `9f9c03a7b364d51096990525c7de94476295fdfc82b0b1ad7a441f58f2f79cf8`
- decoder schema smoke: `semantic_predicate_decoder_passes_smoke` on a
  two-ID `.debug/` target-set copy, but accepted `0` sidecar rows and is not
  scientific evidence

Full live CPU collection:

- result manifest:
  `results/frozen_candidate_score_sidecar_20260427/manifest.md`
- live JSONL:
  `results/frozen_candidate_score_sidecar_20260427/live_candidate_score_sidecar_cpu.jsonl`
- live JSONL sha256:
  `3734e4884c87bc14d3bc74317a47c195bbac85253927ae799e3eaa717cf2e771`
- rows: `70`
- elapsed: `351.12s`
- top labels: `target=44`, `t2t=26`

Live gate:

- status: `semantic_predicate_decoder_fails_smoke`
- matched correct: `21/70`
- accepted: `1`
- clean source-necessary: `0`
- accepted harm: `0`
- control clean union: `0`

Decision: kill the frozen model-scored target-side candidate sidecar producer
on the canonical live SVAMP70 surface. It is cleaner than the older source
likelihood sketch, but it recovers no clean source-necessary examples and does
not improve target accuracy. Do not spend another CPU pass on holdout for this
producer.
