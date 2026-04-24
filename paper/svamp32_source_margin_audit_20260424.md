# SVAMP32 Source Margin Audit - 2026-04-24

## Status

- status: `no_source_margin_signal`
- readiness impact: negative for same-pair Qwen2.5-0.5B to Qwen3-0.6B as the
  current positive-method surface
- gate: source must show positive gold-vs-target-wrong answer margins with an
  advantage over target-alone on at least `2/6` clean residual IDs
- outcome: `0/6` clean residual IDs had positive source-margin advantage

## Motivation

The previous teacher-forced connector diagnostic and answer-teacher microfit
both failed the source-necessity gate: matched source produced `0/6` clean
matched-only positives, and apparent positives leaked to zero-source,
target-only, or slots-only controls. Before writing another connector variant,
this audit asks whether the frozen source model itself contains recoverable
answer evidence on the six clean C2C-only residual IDs.

Primary local reference anchors:

- `paper/svamp32_source_oracle_bound_20260424.md`
- `paper/svamp32_teacher_forced_connector_diagnostic_20260424.md`
- `paper/svamp32_answer_teacher_microfit_20260424.md`
- `references/451_answer_teacher_microfit_refs.md`

## Decision Surface

Top moves considered:

- Source margin audit on six clean residual IDs. It matters because it directly
  tests whether the source assigns higher probability to the gold answer than
  to the target-alone wrong answer. It might fail if final-answer margins miss
  cache-level C2C signal. It is low compute and helps same-pair,
  interpretability, and reproducibility.
- Standalone differentiable answer-margin sidecar. It matters because it can
  optimize matched-vs-control margins directly. It might fail by memorizing six
  IDs or learning target-cache shortcuts. It is medium code/compute and helps
  same-pair and robustness only if the source has signal.
- Latent arithmetic-syndrome sidecar. It matters because it would transmit
  compact numeric residue checks with the target candidate pool as decoder side
  information. It might fail because source/text final answers and source
  margins are weak on the clean IDs. It helps efficiency and interpretability
  but is a new branch.

I executed the source margin audit because it is the cheapest decisive gate
before more same-pair connector tuning.

## Implementation

Added `scripts/analyze_svamp32_source_margin_audit.py`.

The script:

- loads exact-ID SVAMP32 eval examples and existing target/source/text/C2C rows
- enforces unique stable example IDs
- scores the same clean residual IDs and optional target-self-repair IDs
- compares gold numeric continuation logprob against the target-alone wrong
  numeric distractor
- reports final-answer source/text correctness separately from source answer
  margins
- writes JSON and Markdown artifacts with exact IDs, predictions, margins, and
  source-vs-target margin deltas

Unit tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_source_margin_audit.py -q
```

Result: `3 passed in 0.02s`.

Full suite:

```bash
./venv_arm64/bin/python -m pytest -q
```

Result: `657 passed in 25.85s`.

## Command

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_source_margin_audit.py \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --source-jsonl results/svamp_exactid_baselines32_20260423/source_alone.jsonl \
  --text-jsonl results/svamp_exactid_baselines32_20260423/text_to_text.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_source_margin_audit_20260424/source_margin_clean_self.json \
  --output-md results/svamp32_source_margin_audit_20260424/source_margin_clean_self.md \
  --device mps \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --score-target-self
```

## Evidence

Summary:

- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- source/text final clean correct: `0/6`
- source-margin positive clean IDs: `2/6`
- target-margin positive clean IDs: `2/6`
- source-margin positive+advantage clean IDs: `0/6`
- mean source margin: `-3.065174`
- mean target margin: `-3.624139`
- mean source-minus-target margin: `0.558965`

Clean residual rows:

| Example ID | Gold | Distractor | Source Pred | Text Pred | Source Margin | Target Margin | Source - Target | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `13cb77b698eeadb5` | 8142 | 46 | n/a | 4 | -13.759378 | -13.017838 | -0.741541 | no_source_margin_advantage |
| `1d50b408c8f5cd2c` | 949 | 1 | 25 | 25 | -8.573317 | -11.604153 | 3.030836 | no_source_margin_advantage |
| `2de1549556000830` | 39 | 33 | 33 | 33 | -2.990372 | -9.340767 | 6.350395 | no_source_margin_advantage |
| `6e9745b37ab6fc45` | 61 | 600 | 3 | 661 | -1.279598 | -8.293890 | 7.014292 | no_source_margin_advantage |
| `aee922049c757331` | 1 | 17 | 5 | 3 | 5.874361 | 14.522760 | -8.648399 | no_source_margin_advantage |
| `e3ab8666238a289e` | 1 | 4 | 5 | 5 | 2.337257 | 5.989050 | -3.651793 | no_source_margin_advantage |

The source has three negative-margin advantages where target is even more
wrong, but those are not useful source signal because the source still prefers
the target wrong answer over gold. The two source-positive rows are also
target-positive and target margins are much stronger, matching the prior
target-prior/control-leak pattern.

## Artifacts

- `results/svamp32_source_margin_audit_20260424/source_margin_clean_self.json`
- sha256: `16ab06a97024d61cbb6efb3b1cfbebacc9f542bab25862e1671b5e4ec7a919ff`
- `results/svamp32_source_margin_audit_20260424/source_margin_clean_self.md`
- sha256: `9bb5a859eff08ff0727ab6fd8d57bc37b38dbd8dae0f072b725e6fb609f9b91b`
- `.debug/svamp32_source_margin_audit_20260424/logs/source_margin_clean_self.log`
- sha256: `ce606e110ab7f35096ea799d52bedeeed070d8590d0b3db1af789f11b5ab03c3`

## Subagent Synthesis

- ablation agent recommended source candidate-pool plus answer-margin oracle
  as the strongest gate before connector tuning
- creative internet agent recommended a latent arithmetic-syndrome sidecar
  grounded in decoder-side-information coding, but only after source signal is
  demonstrated
- repo-audit agent emphasized provenance-first exact-ID matrices and hard
  failure on artifact/ID/numeric drift
- source-audit agent independently selected this same source-margin audit and
  the `>=2/6` positive-advantage threshold

## Hypothesis Update

- killed: source/text final-answer relay is informative on the six clean
  SVAMP32 residual IDs
- killed: source answer-token margins justify more same-pair connector tuning
  on this exact Qwen2.5-0.5B to Qwen3-0.6B surface
- weakened: standalone answer-margin sidecar for this pair, because its
  strongest prerequisite source margin signal is absent
- still alive: C2C/cache-residual distillation, because C2C remains `16/32`
  while source/text final answers and source margins fail on clean IDs
- still alive: latent syndrome sidecar, but only as a new branch with explicit
  source-control residue checks
- promoted: either switch source pair/strength for a strict cross-family or
  stronger-source falsification, or implement a C2C-residual distillation gate
  that does not depend on source final answers

## Next Exact Gate

Do not run another same-pair Qwen2.5-0.5B to Qwen3-0.6B connector calibration
proxy. The next exact gate should be one of:

- strict stronger-source or cross-family source-informativeness falsification
  on the same frozen SVAMP32 IDs
- C2C-residual distillation sidecar that must clear the same matched-vs-control
  `>=2/6` clean residual source-necessity threshold
- latent syndrome sidecar only if residue predictions beat zero/shuffle
  controls on at least `2/6` clean residual IDs without eroding target-self
