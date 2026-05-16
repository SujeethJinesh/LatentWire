# Phase 9 DecDEC Algorithmic Baseline Preregistration

**Frozen on**: 2026-05-16
**Frozen by**: Codex GPU swarm, under information-value queue
restructuring authorization
**Status**: Frozen while M18 Granite-Small was running and before any DecDEC
baseline scoring run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
"outlier migration" only in a defined terminology note distinguishing this
channel-across-decode-position usage from SmoothQuant's activation-to-weight
difficulty transfer and MoBiQuant's precision-dependent token-sensitivity
shift.

## Purpose

This is a **baseline**, not a proposed method. It has no PASS/KILL decision
rule. It measures how much of the BF16-vs-static-protection gap can be
recovered by an algorithmic DecDEC-style reactive channel-selection policy on
the same deterministic traces and W4A16-style scoring surface used by Phase 9.

The result is required for reviewer pre-emption because DecDEC is the closest
dynamic-channel prior art. It also informs the Phase 1 mechanism split:

- If DecDEC recovers quality while M2/M10/M11 fail, the evidence supports
  stale future-channel signal (hypothesis 3a): per-step current activation
  evidence is needed.
- If DecDEC also fails, that weakens pure signal-staleness explanations and
  shifts attention toward budget insufficiency (3b) or long-horizon compound
  error (3c).

## Prior-Art Anchor

DecDEC (Park, Hyun, Kim, and Lee, OSDI 2025; arXiv 2412.20185) stores the
full-precision-minus-quantized weight residual on CPU and fetches only the
residual rows corresponding to dynamically selected salient channels. The
paper states: "Salient channels are identified dynamically at each decoding
step by analyzing the input activations." The official repository describes
the implementation as bucketed Top-K channel selection, residual fetch, a
residual GEMV, and merge into the base GEMV result.

This baseline implements the **algorithmic selection and protection effect**
only. It does not implement DecDEC's CUDA residual GEMV kernel, PCIe zero-copy
CPU staging, autotuner, or throughput claims. Therefore:

- We may compare recovery/perplexity behavior against DecDEC-style reactive
  selection.
- We may not claim DecDEC latency, memory, kernel efficiency, or systems
  overhead.
- If our methods fail to beat this baseline, the paper must report that
  honestly.

Primary sources:

- USENIX OSDI 2025 paper page:
  `https://www.usenix.org/conference/osdi25/presentation/park-yeonhong`
- arXiv preprint:
  `https://arxiv.org/abs/2412.20185`
- official code:
  `https://github.com/SNU-ARC/decdec`

## Models

Run in this order:

1. `ibm-granite/granite-4.0-h-small`
2. `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`

The Granite-Small run is the Phase 1 priority-2 baseline and must land before
Phase 2 positive-method branching. Nemotron and DeepSeek runs should follow
for cross-model comparison unless an infrastructure failure makes the baseline
unavailable on those models.

## Trace Set

- Source: AIME-2025
- Count: 12 traces under vacation-mode V4 standing decision
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Prompt payload SHA-256 over indices `0-11`: computed by the runner and
  written to each result packet before analysis.

The 12-trace slice matches M2/M10/M11/M18 on Granite-Small.

## Quantization and Scoring

- BF16 baseline target traces are deterministic greedy decode traces.
- Weight quantization: simple symmetric per-channel INT4 represented as
  dequantized tensors for framework compatibility.
- Activations: FP16 unless a channel is explicitly protected.
- Protected channels remain in the unquantized/dequantized high-precision
  path.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260528`.

AWQ-style activation-aware scaling, SmoothQuant scale folding, and any
post-hoc threshold tuning are forbidden for this baseline.

## Algorithmic DecDEC Baseline

At each decode step and for each layer independently:

1. Observe the current layer-output activation magnitudes.
2. Select the top `1%` channels by current absolute activation magnitude.
3. Protect those selected channels in BF16-equivalent precision for the
   corresponding weight rows/columns during the next-token computation.
4. Do **not** smooth the selection over time.
5. Do **not** use future positions, offline union sets, or position bins.

This is deliberately reactive and discontinuous because that is the DecDEC
decision surface. It differs from M2/M10, which use precomputed bins, and M11,
which smooths channel membership over time.

If the framework cannot apply per-step protection without source modification,
the runner may use the strongest algorithmic oracle proxy: protect the
top-1% channels measured at the same decode step when scoring the target
sequence. The packet must label this as `bf16_oracle_decdec_proxy` and must
not make systems-throughput claims.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static top-1% protected set from position `100`.
3. DecDEC algorithmic reactive top-1% protection.
4. Static top-10% matched-budget reference.
5. Random reactive top-1% control with the same per-step cadence and channel
   count as DecDEC, seed `20260528`.

The static top-10% reference is included because DecDEC's per-step Top-K can
touch more total unique channels over a long trace than any fixed top-1% set.
It is a budget sanity check, not a pass/fail control.

## Metric

For each trace and baseline/control regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_1pct - perplexity_BF16)`

Only traces with a positive recoverable static top-1% gap are included in the
primary recovery median. The packet must still report the count and fraction
of no-recoverable-static-gap traces.

Report:

- per-trace perplexity for all regimes
- per-trace recovery for DecDEC, static top-10%, and random reactive control
- median recovery and bootstrap 95% CI for every non-BF16 regime
- median separation from static top-1%
- median separation from random reactive control
- total unique channels protected by DecDEC over the trace
- average protected channels per decode step

## Descriptive Readout

This baseline has no pass/kill decision. The checker returns
`PASS_DECDEC_BASELINE_REPORTED` when the packet is artifact-complete.
It returns `FAIL_INFRA_DECDEC_BASELINE` only if the packet is incomplete or
the runner cannot apply the mechanical descriptive rule.

The paper must report the recovery value regardless of whether it helps or
hurts our proposed methods.

## Required Artifacts

Each DecDEC baseline packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA)
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- BF16 target traces or cited cache source with SHA-256
- activation evidence used for per-step Top-K channel selection
- protected-channel trajectory for DecDEC and random reactive control
- quantization configuration
- per-trace perplexity table
- per-trace recovery table
- bootstrap CI table
- checker result and artifact check
- artifact hashes

## Forbidden Actions

- Modifying prior preregistration files.
- Claiming DecDEC CUDA kernel, PCIe, CPU-staging, latency, or memory results.
- Using future decode positions to choose the current reactive protected set.
- Smoothing DecDEC channel membership.
- Dropping the random reactive control.
- Dropping the static top-10% budget reference.
- Using AWQ-style scaling or SmoothQuant scale folding.
- Modifying model source code.

## Paper Integration Rule

After Granite-Small lands, the paper must add a DecDEC comparison paragraph
before any positive-method claim is made. If DecDEC outperforms M2/M10/M11,
frame it as evidence that reactive current-step channel evidence carries
information that predeclared bins and smoothed offline sets miss. If DecDEC
also fails, frame it as evidence against simple signal-staleness explanations
and shift Phase 1 interpretation toward budget insufficiency or compound
error.
