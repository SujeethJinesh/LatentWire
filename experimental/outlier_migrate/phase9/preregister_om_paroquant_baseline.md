# Phase 9 ParoQuant Baseline Preregistration

**Frozen on**: 2026-05-20
**Frozen by**: Codex GPU swarm, after KL accumulation landed and before any
ParoQuant scoring run.
**Status**: Frozen baseline preregistration. This is a descriptive SOTA
comparison, not a new proposed method and not a pass/kill gate.

## Terminology

This preregistration uses **decode-position channel-set drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a defined terminology note distinguishing this
channel-across-decode-position usage from SmoothQuant's activation-to-weight
difficulty transfer and MoBiQuant's precision-dependent token-sensitivity
shift.

## Purpose

ParoQuant is the closest current W4A16 reasoning-quantization baseline that
directly threatens a negative-result framing. The ParoQuant paper reports a
weight-only PTQ pipeline using scaled pairwise Givens rotations plus
channel-wise scaling to suppress outliers, with reasoning-task gains over AWQ
and an ICLR 2026 implementation release.

This baseline asks whether an algorithmic ParoQuant reproduction recovers the
BF16-vs-static-top-1% W4A16 gap on the same Granite-4-H-Small long-decode
reasoning slice used by the Phase 9 intervention methods.

## Prior-Art Sources

- ParoQuant paper: `arXiv:2511.10645`, "ParoQuant: Pairwise Rotation
  Quantization for Efficient Reasoning LLM Inference".
- OpenReview ICLR 2026 PDF: `https://openreview.net/pdf?id=1USeVjsKau`.
- Official implementation: `https://github.com/z-lab/paroquant`.

The upstream repository describes ParoQuant as state-of-the-art INT4
quantization for LLMs using learned pairwise rotations to suppress weight
outliers. The paper's Section 4 describes scaled pairwise rotation as a series
of independent Givens rotations combined with channel-wise scaling, optimized
layer-wise to reduce quantization-induced output loss. The official README
notes that the main branch is under active development and recommends a legacy
branch for reproducing paper results; this packet therefore records the exact
upstream commit or package version used if any upstream code is invoked.

## Baseline Scope

This run implements an **algorithmic ParoQuant baseline**:

- include channel-wise scaling;
- include independent Givens pair rotations inside channel groups;
- use W4A16 weight-only quantization for evaluated linear weights;
- compute long-decode perplexity recovery on the frozen Granite-Small traces;
- do not require ParoQuant's fused CUDA kernel;
- do not claim throughput or systems-overhead parity with ParoQuant.

If the upstream package can quantize `ibm-granite/granite-4.0-h-small` in this
environment without upgrading vLLM or modifying model source code, the runner
may use it and must record the package version, commit SHA, invocation, and
configuration. If upstream execution is incompatible with Granite-Small, the
runner must use a local algorithmic reproduction of the above transform and
must label the result as `algorithmic_reproduction_not_full_upstream`.

If the local algorithm omits any ParoQuant component beyond fused kernels, the
runner must write that omission to `paroquant_limitations.json`, and the paper
must not present the number as a full reproduction of ParoQuant.

## Model

Primary model:

- `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`

This preregistration covers Granite-Small only. DeepSeek-R1-Distill-Qwen-1.5B
and DeepSeek-R1-Distill-Llama-3.1-8B are follow-up ParoQuant baselines after
the Granite-Small packet lands.

## Trace Set

- Source: AIME-2025
- Count: 12 traces
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- The runner must verify the canonical prompt file hash and write the prompt
  payload SHA-256 over indices `0-11`.

The 12-trace slice matches M2, M10, M11, M11b, M18, M26, DecDEC, and KL
accumulation on Granite-Small.

## Quantization and Scoring

- BF16 baseline target traces are deterministic greedy decode traces.
- Weight quantization: W4A16 weight-only PTQ.
- Activation dtype: FP16/BF16 according to the existing Granite-Small runner
  path; no activation quantization is allowed.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260601`.

The baseline may reuse BF16 traces and activation evidence already produced by
previous Granite-Small packets if the reused artifact SHA is recorded in
`source_artifacts.json`.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static top-1% protected W4A16 baseline from position `100`.
3. ParoQuant W4A16 algorithmic baseline.
4. Random channel-matched W4A16 control if the local implementation exposes a
   channel-selection/protection component; otherwise this control is marked
   `not_applicable_no_channel_selection` and the reason is written to
   `control_metrics.json`.

ParoQuant is a baseline, not an intervention proposed by this project, so no
ParoQuant pass/kill decision string is assigned.

## Metric

For each trace:

`recovery = 1 - (perplexity_ParoQuant - perplexity_BF16) / (perplexity_static_top1 - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The packet must still report the count and fraction
of no-recoverable-static-gap traces.

Report:

- per-trace perplexity for all regimes;
- per-trace recovery for ParoQuant and any controls;
- median recovery and bootstrap 95% CI;
- ParoQuant minus static-top-1% median perplexity delta;
- ParoQuant minus M11b top-5 median recovery where comparable;
- exact implementation mode:
  `upstream_paroquant`, `local_algorithmic_reproduction`, or
  `algorithmic_reproduction_not_full_upstream`;
- any omitted upstream components.

## Descriptive Interpretation Rule

The checker returns `PASS_PAROQUANT_BASELINE_REPORTED` if the packet is
complete and the descriptive statistics can be audited.

Interpretation bands for paper framing:

- If ParoQuant median recovery is `> 0.50`, scope the negative-result framing
  tightly: ParoQuant substantially addresses the gap on Granite-Small.
- If ParoQuant median recovery is between `0.30` and `0.50`, report it as a
  meaningful but incomplete SOTA baseline.
- If ParoQuant median recovery is `< 0.30`, report that this W4A16 SOTA
  baseline does not recover the Granite-Small long-decode gap under the
  algorithmic reproduction used here.

These are interpretation bands, not preregistered pass/kill thresholds.

## Required Artifacts

Each ParoQuant packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA);
- model provenance with HuggingFace snapshot commit;
- prompt manifest and prompt SHA;
- exact command line and stdout/stderr logs;
- upstream ParoQuant source/version metadata, if used;
- transform configuration and learned/selected pair metadata;
- channel-wise scaling metadata;
- quantization configuration;
- source artifacts manifest for reused BF16 traces or activation data;
- per-trace perplexity table;
- per-trace recovery table;
- bootstrap CI table;
- descriptive checker result and artifact check;
- artifact hashes.

## Forbidden Actions

- Modifying earlier preregistration files.
- Changing trace indices after observing ParoQuant data.
- Dropping the static top-1% baseline.
- Reporting ParoQuant as a full upstream reproduction if fused kernels,
  QAT-like second-stage optimization, or upstream conversion is omitted.
- Upgrading vLLM for this run.
- Downloading Qwen3.6 or Kimi Linear weights for this run.
- Modifying model source code.
- Cherry-picking only traces with favorable recoveries.

## Paper Integration Rule

This packet is mandatory reviewer context. The paper must report ParoQuant as
a SOTA W4A16 reasoning-quantization baseline and must distinguish:

- full upstream ParoQuant results reported by its authors;
- this repo's Granite-Small algorithmic baseline;
- the paper's own M11b budget-scaling result.

If ParoQuant recovers the Granite-Small gap, the paper's contribution shifts
toward diagnosis and SOTA-contextualization rather than claiming that W4A16
reasoning PTQ is broadly unsolved. If it does not, the paper may state that
the gap persists for this small hybrid reasoning model under the audited
algorithmic ParoQuant baseline.
