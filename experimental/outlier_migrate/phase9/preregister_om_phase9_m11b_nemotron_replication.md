# Phase 9 M11b Nemotron Replication Preregistration

**Frozen on**: 2026-05-20
**Frozen by**: Codex GPU swarm, after Granite-Small M11b and ParoQuant
Granite-Small landed and before any M11b Nemotron scoring run.
**Status**: Frozen cross-model replication preregistration.

## Terminology

This preregistration uses **decode-position channel-set drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a defined terminology note distinguishing this
channel-across-decode-position usage from SmoothQuant's activation-to-weight
difficulty transfer and MoBiQuant's precision-dependent token-sensitivity
shift.

## Purpose

Granite-Small M11b produced the first mechanical positive signal for an
inference-time channel-protection method:

- `m11b_top5` median recovery `0.4492840911245966`;
- CI95 `[-1.3009000187907436, 1.00079062654749]`;
- static-top10 matched-budget median `-0.05176717655558488`;
- checker decision `PASS_M11B_BUDGET_MATTERS`.

The CI is wide and the result is single-model. This replication tests whether
the budget-scaling signal transfers to the Phase 2 same-family scale-out model
`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`.

## Prior-Art Differentiation

DecDEC (Park, Hyun, Kim, and Lee, OSDI 2025; arXiv 2412.20185) identifies
salient channels dynamically at each decode step and fetches residual
compensation. M11b is not reactive per-step DecDEC; it keeps an
EMA-smoothed protected-set decision surface and tests whether budget scaling
to top-5% or top-10% transfers across a second hybrid reasoning model.

ParoQuant (arXiv 2511.10645; ICLR 2026) is now a required W4A16 SOTA baseline.
This replication is not a ParoQuant comparison and must not be framed as
beating ParoQuant. It asks whether the project-owned M11b mechanism generalizes
beyond Granite-Small.

PMPD (arXiv 2410.13461) changes bit width across decode positions; M11b keeps
the quantization format fixed and changes only the channel protection budget
for an EMA-smoothed set.

## Model

Primary replication model:

- `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
- HuggingFace snapshot commit:
  `cbd3fa9f933d55ef16a84236559f4ee2a0526848`

This is the same model snapshot used by the completed Phase 2 partial
cross-model validation packet.

## Trace Set

- Source: AIME-2025
- Count: 12 traces
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- The runner must verify the canonical prompt file hash and write the prompt
  payload SHA-256 over indices `0-11`.

The 12-trace slice matches the Granite-Small Phase 9 intervention slice.

## Quantization and Scoring

- BF16 baseline target traces are deterministic greedy decode traces.
- Weight quantization: simple symmetric per-channel INT4 represented as
  dequantized tensors for framework compatibility.
- Activations: FP16 under the existing scoring autocast path.
- Protected channels remain in the unquantized/dequantized high-precision path.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260604`.

AWQ-style activation-aware scaling, SmoothQuant scale folding, ParoQuant
rotations, and post-hoc threshold tuning are forbidden for this replication.

## M11b Protected-Set Update Rule

For each layer independently, maintain a protected-set score vector `p_l(t)`
with one score per channel.

Initialization:

- `p_l(0)` is the calibration top-1% indicator at decode position `100`.

At each decode position that is a multiple of `100`:

`p_l(t+1) = 0.3 * current_top_budget_indicator_l(t) + 0.7 * p_l(t)`

Budget levels:

- top-1%
- top-5%
- top-10%

Protected channels at position `t` are the highest-scoring channels in
`p_l(t)` up to that budget. There is no top-3% cap in the 5% and 10% arms
because the purpose is a budget-scaling replication.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static top-1% protected set from position `100`.
3. M11b EMA-smoothed top-1%.
4. M11b EMA-smoothed top-5%.
5. M11b EMA-smoothed top-10%.
6. Static top-10% matched-budget control from position `100`.

## Metric

For each trace and M11b/control regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_top1 - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The packet must still report the count and fraction
of no-recoverable-static-gap traces.

Report:

- per-trace perplexity for all regimes;
- per-trace recovery for M11b top-1%, top-5%, top-10%, and static top-10%;
- median recovery and bootstrap 95% CI for every non-BF16 regime;
- M11b top-5% and top-10% separation from M11b top-1%;
- M11b top-5% and top-10% separation from static top-10%;
- effective protected-channel count statistics for every budget level.

## Decision Rule

### PASS_M11B_NEMOTRON_REPLICATES

Return pass if either M11b top-5% or M11b top-10% satisfies both:

1. median recovery is at least `0.30`; and
2. median recovery beats static top-10% matched-budget control by at least
   `0.15`.

This supports cross-model transfer of the budget-scaling mechanism.

### KILL_M11B_NEMOTRON_BUDGET_INSUFFICIENT

Return this kill if both M11b top-5% and M11b top-10% are within `0.05`
median recovery of M11b top-1%.

### AMBIGUOUS_M11B_NEMOTRON

Return ambiguous for intermediate outcomes, including cases where high-budget
M11b improves over top-1% but misses the `0.30` recovery shelf, fails to beat
static top-10% by `0.15`, or has highly overlapping CIs.

### FAIL_INFRA_M11B_NEMOTRON

Return infrastructure failure for model load failure, incomplete packet, OOM
that cannot be fixed by batch-size reduction, missing required artifacts, or
checker failure that prevents applying the mechanical decision rule.

## Required Artifacts

Each packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA);
- model provenance with HuggingFace snapshot commit;
- prompt manifest and prompt SHA;
- exact command line and stdout/stderr logs;
- BF16 target traces;
- activation/top-channel evidence used for EMA updates;
- protected-set trajectory for each budget level;
- quantization configuration;
- per-trace perplexity table;
- per-trace recovery table;
- bootstrap CI table;
- checker result and artifact check;
- artifact hashes.

## Forbidden Actions

- Modifying prior preregistration files.
- Changing trace indices after observing Nemotron M11b data.
- Adjusting alpha, budgets, or decision thresholds after observing data.
- Dropping the static top-10% matched-budget control.
- Using AWQ-style scaling, SmoothQuant scale folding, or ParoQuant rotations.
- Selectively reporting only the best budget arm.
- Modifying model source code.

## Paper Integration Rule

If this replication passes, the paper may describe M11b as replicated across
two hybrid reasoning models, while still reporting ParoQuant as the stronger
Granite-Small SOTA baseline unless M11b also beats ParoQuant in a directly
matched comparison. If this replication kills or is ambiguous, M11b remains a
single-model positive signal and the paper must not claim a general positive
method.
