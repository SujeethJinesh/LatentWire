# SVAMP32 Teacher-Forced Connector Diagnostic - 2026-04-24

## Status

- status: `no_teacher_forced_source_signal`
- readiness impact: negative for the current live connector branch; useful as a source-necessity falsification
- gate: matched answer margin must be positive and better than zero-source, shuffled-source, target-only, and slots-only controls on at least `2/6` clean residual IDs
- outcome: `0/6` clean residual IDs had matched-only positive answer margins; the `2/6` matched-positive IDs were also positive under source-destroying controls

## Motivation

The previous Perceiver-query generation sweep reached only `10/32` and `0/6`
clean residual IDs. Before spending another full generation sweep on a stronger
objective or wider slice, this diagnostic asks whether the trained connector
already contains source-specific answer evidence below greedy decoding.

The diagnostic is deliberately stricter than a matched-only logprob probe:
positive evidence must disappear when source K/V is zeroed, when the source
prompt is shuffled, when target-only memory is forced, or when only learned
slots remain. This separates real source communication from target-prior or
cache-shape effects.

Local reference anchors:

- `references/443_query_resampler_connector_refs.md`
- `references/447_qformer_connector_followup_refs.md`
- `references/450_lateral_connector_repair_refs.md`

## Decision Surface

Top moves considered before execution:

- Teacher-forced source-control answer-margin diagnostic. It matters because it
  is the cheapest way to decide whether the failed generation checkpoint hides
  usable source-specific signal. It could fail if answer-token margins are too
  brittle or if the gold-vs-target distractor contrast is not representative.
  It gains direct matched-vs-control evidence at low compute cost and helps
  same-pair, interpretability, and reproducibility.
- Source correctness and provenance audit on existing SVAMP32 predictions. It
  matters because it checks whether the source model is informative on the six
  residual IDs. It could fail to distinguish latent transfer from text answer
  copying. It is cheap and helps reproducibility and source-pair design.
- Full seed repeat or larger frozen slice. It matters for eventual ICLR
  rigor, but it is premature because the live branch has not cleared the
  residual gate. It is higher compute and helps robustness only after a live
  method row exists.

The teacher-forced diagnostic was the highest-value move because it could
quickly kill or revive the current connector before any broader benchmark work.

## Implementation

Added `scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py`.

The script:

- loads an existing `RotAlignKVTranslator` checkpoint
- reuses the same prefix-state path as `latent_bridge/evaluate.py`
- scores teacher-forced target continuations for the gold numeric answer and a
  target-alone wrong numeric distractor
- compares controls `matched`, `zero_source`, `shuffled_source`, `target_only`,
  and `slots_only`
- writes JSON and Markdown artifacts with exact example IDs, margins,
  controls, source IDs, bytes, and gate status

Unit tests:

```bash
./venv_arm64/bin/python -m pytest tests/test_analyze_svamp32_teacher_forced_connector_diagnostic.py -q
```

Result: `2 passed in 0.01s`.

## Commands

Clean residual IDs only:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py \
  --translator .debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean.json \
  --output-md results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean.md \
  --device mps \
  --fixed-gate 0.15 \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1
```

Clean residual plus target-self-repair IDs:

```bash
PYTHONUNBUFFERED=1 ./venv_arm64/bin/python scripts/analyze_svamp32_teacher_forced_connector_diagnostic.py \
  --translator .debug/svamp32_perceiver_query_connector_20260424/checkpoints/qwen25_to_qwen3_svamp32_perceiver_queries_w030_m010_r16_q8_seed1.pt \
  --source-model Qwen/Qwen2.5-0.5B-Instruct \
  --target-model Qwen/Qwen3-0.6B \
  --eval-file results/svamp_exactid_baselines32_20260423/_artifacts/svamp_eval_70_32.jsonl \
  --target-jsonl results/svamp_exactid_baselines32_20260423/target_alone.jsonl \
  --teacher-jsonl results/svamp_exactid_baselines32_20260423/c2c_generate.jsonl \
  --target-set-json results/svamp32_query_innovation_query_pool_transport_20260423/svamp32_innovation_target_set_20260423.json \
  --output-json results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean_self.json \
  --output-md results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean_self.md \
  --device mps \
  --fixed-gate 0.15 \
  --kv-transport k_only \
  --position-selection-ratio 0.5 \
  --position-selection-metric attention \
  --source-reasoning-mode brief_analysis \
  --source-use-chat-template \
  --target-use-chat-template \
  --source-enable-thinking false \
  --target-enable-thinking false \
  --random-salt 1 \
  --score-target-self
```

## Evidence

Summary:

- clean residual IDs scored: `6`
- target-self-repair IDs scored: `3`
- matched-positive clean IDs: `2`
- matched-only clean IDs: `0`
- control-leak clean IDs: `2`
- mean matched margin: `-3.764944`
- mean best-control margin: `-2.272635`
- mean matched-minus-control margin: `-1.492309`

Clean residual rows:

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| `13cb77b698eeadb5` | 8142 | 46 | -13.347810 | -12.956595 | -0.391215 | zero_source | control_or_negative |
| `1d50b408c8f5cd2c` | 949 | 1 | -10.996808 | -11.238231 | 0.241423 | zero_source | control_or_negative |
| `2de1549556000830` | 39 | 33 | -9.659750 | -8.681938 | -0.977812 | zero_source | control_or_negative |
| `6e9745b37ab6fc45` | 61 | 600 | -7.854908 | -1.341321 | -6.513587 | shuffled_source | control_or_negative |
| `aee922049c757331` | 1 | 17 | 14.129086 | 14.568278 | -0.439192 | zero_source | control_or_negative |
| `e3ab8666238a289e` | 1 | 4 | 5.140525 | 6.013994 | -0.873469 | zero_source | control_or_negative |

Target-self-repair rows:

| Example ID | Gold | Distractor | Matched Margin | Best Control Margin | Matched - Control | Best Control | Status |
|---|---:|---:|---:|---:|---:|---|---|
| `4c84ebf42812703b` | 10 | 2 | -5.167335 | -5.577995 | 0.410661 | shuffled_source | control_or_negative |
| `4d780f825bb8541c` | 26 | 1 | -10.419864 | -10.148226 | -0.271638 | shuffled_source | control_or_negative |
| `de1bf4d142544e5b` | 57 | 2 | -2.471706 | -2.605896 | 0.134190 | zero_source | control_or_negative |

The two matched-positive clean rows are not usable evidence because
zero-source controls score higher than matched source. That is target-prior or
cache-shape leakage, not source communication.

## Artifacts

- `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean.json`
- sha256: `3e67de34ca7121cc803bc10bad78b1b3aab4e2857efd8654eab6655132f693a9`
- `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean.md`
- sha256: `098cd43ddddc9e269f357699260880cacab3cc4925851035db90323107ccb48d`
- `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean_self.json`
- sha256: `47443a71295606330e26911777ed4b496f390506538f943695e2e1d6df746c0c`
- `results/svamp32_teacher_forced_connector_diagnostic_20260424/perceiver_queries_gate015_answer_margin_clean_self.md`
- sha256: `6bf0367b38a34621508ebb8c4e40209462ac7107c432914d650a0ea584be6903`
- `.debug/svamp32_teacher_forced_connector_diagnostic_20260424/logs/perceiver_queries_gate015_answer_margin_clean_self.log`
- sha256: `ebecf85e36ff89b93ba15b947eaf889ecb30af1ac728c43b01ae691c23d182b0`

## Subagent Synthesis

- literature and internet-creative agents recommended answer-margin diagnostics
  before another generation sweep, grounded in bottleneck connector designs
  such as Q-Former/Perceiver-style querying and source-control repair
- ablation agent recommended matched, zero-source, shuffled-source, target-only,
  and slots-only controls as the minimal decisive set
- repo-audit agent recommended a standalone script rather than adding more
  special cases to `latent_bridge/evaluate.py`

## Hypothesis Update

- killed: this Perceiver-query checkpoint contains hidden teacher-forced
  source-specific answer evidence on the six clean residual IDs
- killed: another matched generation sweep from this checkpoint is likely to
  recover a source-necessary positive row
- weakened: K-only Perceiver-query transport under the current fit objective
  creates a usable residual channel before answer-token supervision
- still alive: receiver-conditioned connectors if trained directly against an
  answer-token or C2C residual objective with source-destroying controls
- promoted: controlled answer-token microfit as the next architecture gate

## Next Exact Gate

Do not widen benchmarks or run seed repeats for this killed checkpoint. The
next gate is a controlled microfit, not another full sweep:

- train a small answer-token or C2C residual objective on the six clean residual
  IDs plus target-self-preserve IDs
- evaluate with this same diagnostic surface before greedy generation
- promote only if matched-only positive margins appear on at least `2/6` clean
  IDs and disappear under zero-source, shuffled-source, target-only, and
  slots-only controls
- if the microfit cannot clear that gate, kill the current same-pair connector
  architecture and move to a source-informativeness or stronger source-pair gate
