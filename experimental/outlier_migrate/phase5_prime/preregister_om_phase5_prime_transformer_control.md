# OutlierMigrate Phase 5' Pure-Transformer Control Preregistration

**Frozen on**: 2026-05-12
**Frozen by**: Codex GPU swarm, under `swarm/goal.md` Phase 5' authorization
**Pivot depth**: 1 from OutlierMigrate Phase 0
**Status**: Frozen before any Phase 5' inference, migration rows, or activation
statistics were collected.

## Purpose

OutlierMigrate Phase 0/1/2 established strong decode-time top-channel
migration in Mamba-2 hybrid reasoning models. Phase 5' is a
pure-Transformer control. It tests whether the same migration metric is a
broad property of reasoning LLMs or whether it is substantially sharper in
hybrid sequence-mixer architectures.

## Hypothesis

The Phase 0/1/2 migration metric, measured on a pure-Transformer reasoning
model with RoPE and no Mamba components, will fall into one of two
predefined regimes:

- **Dynamic transformer regime**: pure Transformers also show top-channel
  migration above the 5% shelf, broadening the finding beyond hybrid
  Mamba-2 models.
- **Static transformer regime**: pure Transformers stay below the 5% shelf,
  sharpening the claim that the observed effect is hybrid-specific.

This preregistration is intentionally non-directional. The result must be
reported as measured.

## Model selection and compatibility precheck

Primary model:

- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
- HuggingFace snapshot commit observed before inference:
  `ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562`
- `AutoConfig` architecture: `Qwen2ForCausalLM`

Fallback model, only if the primary fails vLLM inference compatibility:

- `Qwen/Qwen2.5-1.5B-Instruct`
- HuggingFace snapshot commit observed before inference:
  `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`
- `AutoConfig` architecture: `Qwen2ForCausalLM`

Compatibility checks performed before freezing this preregistration:

- `AutoConfig.from_pretrained(..., trust_remote_code=True)` succeeded for
  both primary and fallback.
- `AutoTokenizer.from_pretrained(..., trust_remote_code=True)` succeeded for
  both primary and fallback.
- Local vLLM version `0.10.2` reports support for `Qwen2ForCausalLM`.

Because the primary model passed the precheck, Phase 5' uses
`deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`. The fallback may be used only
for a documented inference-infrastructure failure, not for a scientific or
post-hoc result-selection reason.

## Trace set

- Source: AIME-2025
- Count: 24 traces
- Selection: deterministic, prompt indices 0-23
- Prompt SHA-256: `aa038b29332b6d137d558205ee441163e7ea4cb3cc323eb705a2f5928fd2fe4e`

The prompt set must match OutlierMigrate Phase 1.

## Decode position grid

Measure at decode positions:

- `{100, 500, 1000, 5000, 10000, 20000}`

This grid is identical to Phase 1 for direct comparability.

## Measurement procedure

1. Run deterministic greedy decoding for the fixed 24-prompt AIME-2025
   trace set.
2. Capture residual-stream channel magnitudes at every transformer block at
   decode positions `{100, 500, 1000, 5000, 10000, 20000}`.
3. For each layer and trace, identify the top-1% channels by magnitude at
   position 100.
4. Compute the original migration metric: the fraction of top-1% channels
   at position 100 whose rank at position 20000 differs by more than 2
   positions.
5. Aggregate across layers and traces with the same weighting as Phase 0/1/2.
6. Bootstrap over the 24 traces with seed `20260511` to compute a 95% CI.

The result packet must also report the set-leaving / within-set
rank-shuffling decomposition used in later analysis, but the Phase 5'
decision is based only on the preregistered migration metric and 95% CI.

## Decision rule

### DYNAMIC_REGIME_TRANSFORMER

Return this decision if:

- migration fraction > 0.05, and
- bootstrap 95% CI lower bound > 0.05.

Implication: the migration finding broadens to pure-Transformer reasoning
LLMs.

### STATIC_REGIME_TRANSFORMER

Return this decision if:

- migration fraction < 0.05, and
- bootstrap 95% CI upper bound < 0.05.

Implication: the finding is hybrid-specific under this metric.

### AMBIGUOUS_TRANSFORMER

Return this decision for any in-between result or wide confidence interval.

### FAIL_INFRA_TRANSFORMER

Return this decision for infrastructure failure, missing artifacts, model
load failure not resolved by the preregistered fallback, incomplete trace
set, or incomplete packet.

## Required artifacts

The result packet must include:

- `metadata/environment.json`
- `metadata/environment.txt`
- `metadata/model_provenance.json`
- `metadata/prompt_manifest.json`
- `metadata/command.json`
- full stdout/stderr logs
- per-layer/per-trace activation or reduced-rank evidence sufficient to
  recompute the metric
- `profiler_metrics.json` or equivalent metrics file
- `checker_result.json`
- `artifact_check.json`
- artifact hashes tying the reduced metrics back to packet files

## Forbidden actions

- Choosing the direction of the decision rule after observing the data.
- Switching from the primary to the fallback model after observing any
  migration row.
- Changing the trace set, prompt ordering, decode grid, top-1% threshold,
  rank-move threshold, or bootstrap seed after observing Phase 5' data.
- Excluding prompts, layers, or decode positions post-hoc.
- Modifying this preregistration after Phase 5' inference begins.

## Integration rule

If Phase 5' returns `DYNAMIC_REGIME_TRANSFORMER`, the paper may state that
the migration metric is not confined to Mamba-2 hybrids, while keeping the
Lineage-2 layer-uniformity claims scoped to measured hybrid models.

If Phase 5' returns `STATIC_REGIME_TRANSFORMER`, the paper should sharpen
the claim that high migration is characteristic of the measured hybrid
Mamba-2 reasoning models rather than pure-Transformer reasoning models.

If Phase 5' returns `AMBIGUOUS_TRANSFORMER` or `FAIL_INFRA_TRANSFORMER`,
the paper must report the limitation without using Phase 5' to broaden or
narrow the main claim.
