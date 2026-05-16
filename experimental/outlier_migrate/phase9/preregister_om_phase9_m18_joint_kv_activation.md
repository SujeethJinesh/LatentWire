# Phase 9 M18 Joint KV-Cache + Activation Protection Preregistration

**Frozen on**: 2026-05-16
**Frozen by**: Codex GPU swarm, under vacation-mode 72-hour sprint
authorization
**Status**: Frozen after M2, M10, and M11 landed on Granite-4-H-Small and
before any M18 scoring run.

## Terminology

This preregistration uses **decode-position channel drift** and
**long-decode channel drift** for the measured phenomenon. The paper may use
the phrase "outlier migration" only in a defined context that distinguishes
this channel-set shift across decode positions from SmoothQuant's
activation-to-weight difficulty transfer and MoBiQuant's precision-dependent
token-sensitivity shift.

## Preconditions and Motivation

M2 position-conditional protected-set switching landed with
`KILL_M2_RANDOM_CONTROL_BEATS` on Granite-4-H-Small. M10 hard position-binned
scale tables also landed with `KILL_M10_RANDOM_CONTROL_BEATS`. M11
EMA-smoothed drift protection landed with `KILL_M11_AMBIGUOUS`: it did not
pass the positive-method bar, but unlike M2/M10 it was not beaten by its
random-walk control.

The current evidence suggests that changing only the activation/weight
protected set is insufficient. M18 tests the next structurally different
mechanism: channels that are high-magnitude in linear-layer activations may
also require high-precision treatment in the attention key cache along the
same channel axis. If the quality loss comes from a cross-tensor coupling
between activation channel `c` and key-cache channel `c`, activation-only
protection can fail even when it tracks the right activation channels.

M18 is run before M12 in this window because the human explicitly authorized
skipping M17 after M10 and M11 failed to produce a positive signal, and
because M18 tests a different tensor rather than another set-update smoothing
policy.

## Prior-Art Differentiation

KIVI (arXiv 2402.02750) studies KV-cache quantization and finds that key
cache entries should be quantized per-channel while value cache entries should
be quantized per-token. M18 differs by linking the key-cache precision choice
to the activation channel that is drifting during decode. KIVI is a KV-cache
method with no activation cross-reference; M18 explicitly asks whether
activation channel `c` and key-cache channel `c` must be protected jointly.

PM-KVQ (arXiv 2505.18610) studies progressive mixed-precision KV-cache
quantization for long-CoT LLMs. Its closest mechanism is KV-cache degradation
under long reasoning, including position-dependent key-cache behavior. M18 is
different because the decision unit is a joint activation/KV channel selected
from activation-channel evidence, not a KV-only precision schedule.

PMPD (Progressive Mixed-Precision Decoding, arXiv 2410.13461) states:
"PMPD builds upon the observation that the prefill phase and the initial
tokens of the decoding phase are more sensitive to approximations than later
tokens." M18 tests the opposite pressure in long reasoning traces: later
decode positions may remain sensitive because the channel identity requiring
high precision shifts with decode position and can propagate through both
activation and key-cache tensors.

DecDEC (Park, Hyun, Kim, and Lee; OSDI 2025) uses per-step reactive channel
identification: "Salient channels are identified dynamically at each decoding
step by analyzing the input activations." M18 is not a CUDA-kernel throughput
baseline and does not claim DecDEC systems speed. It is an algorithmic
mechanism test: if activation evidence identifies a channel, does protecting
the corresponding key-cache channel improve over activation-only and KV-only
baselines?

## Model

Primary model:

- `ibm-granite/granite-4.0-h-small`
- HuggingFace snapshot commit:
  `b8c0982bab7fde4eb48110f5a069527c008fab39`

M18 replication on additional models is not part of this primary gate. The
72-hour sprint prioritizes M12 and DecDEC baseline after this Granite-Small
M18 run unless a later explicit instruction changes the order.

## Trace Set

- Source: AIME-2025
- Count: 12 traces under vacation-mode V4 standing decision
- Selection: deterministic prompt indices `0-11`
- Prompt file:
  `experimental/shared/prompts/aime_2025_indices_0_23.jsonl`
- Prompt payload SHA-256 over indices `0-11`: computed by the runner and
  written to the result packet before analysis.

The 12-trace slice is chosen before M18 data exists and matches the
vacation-mode slice used by M2/M10/M11 on Granite-Small. Decision thresholds
are not relaxed for the smaller slice.

## Quantization and Scoring

- BF16 baseline target traces are deterministic greedy decode traces.
- Weight quantization: simple symmetric per-channel INT4 represented as
  dequantized tensors for framework compatibility.
- Activations: FP16 unless a channel is explicitly protected.
- Static activation protected channels remain in the unquantized/dequantized
  high-precision path, as in prior Phase 4/Phase 9 runners.
- Key-cache protected channels remain FP16/BF16-equivalent in the key cache
  for the matching attention head/channel axis.
- Value-cache protected channels are only used in the secondary `K+V` arm.
- Scoring position: decode position `10000`.
- Scoring window: 512 tokens ending at position `10000`.
- Bootstrap samples: `1000`.
- Bootstrap seed: `20260527`.

AWQ-style activation-aware scaling, SmoothQuant scale folding, and any
post-hoc threshold tuning are forbidden for M18.

## Channel Selection

For each attention-containing layer independently:

1. Identify the activation static-1% channel set from decode position `100`.
2. Identify the activation drift-aware union channel set from positions
   `{100, 1000, 5000, 10000}`.
3. For M18 `activation+K`, protect the activation union channel `c` and the
   corresponding key-cache channel `c` whenever the layer exposes an attention
   key-cache tensor with that channel axis.
4. For M18 `activation+K+V`, also protect value-cache channel `c` when the
   value-cache tensor exposes the same channel axis.

For layers without an attention key/value cache, M18 falls back to
activation-only protection and records the layer as `no_kv_axis` in the
artifact packet. The runner must not modify model source code to expose
hidden tensors. If key/value hooks are inaccessible without source
modification, the runner must return `FAIL_INFRA_M18` rather than silently
substituting a different tensor.

## Regimes

Each packet must evaluate:

1. BF16 baseline.
2. Static activation-only top-1% protected set from position `100`.
3. KIVI-style key-cache-only per-channel protection: protect top-1% key-cache
   channels by key-cache magnitude, with no activation cross-reference.
4. M18 joint activation+K protection: protect activation union channels and
   corresponding key-cache channels.
5. M18 joint activation+K+V protection: same as M18 activation+K, plus
   corresponding value-cache channels.
6. Random-coupled activation+K control: matched activation/K protected counts
   and matched layer coverage, but channel indices are generated from seed
   `20260527` independent of activation magnitudes.

The user-facing sprint prompt specified five regimes. This preregistration
adds the random-coupled control because the active vacation-mode rule requires
a random-channel negative control for every method. Adding this control makes
the gate stricter and does not relax any threshold.

## Metric

For each trace and each M18/control regime:

`recovery = 1 - (perplexity_regime - perplexity_BF16) / (perplexity_static_activation_1pct - perplexity_BF16)`

Only traces with a positive recoverable static activation-only gap are
included in the primary recovery median. The packet must still report the
count and fraction of no-recoverable-static-gap traces.

Report:

- per-trace perplexity for all regimes
- per-trace recovery for KIVI-style K-only, M18 activation+K,
  M18 activation+K+V, and random-coupled control
- median recovery and bootstrap 95% CI for every non-BF16 regime
- median separation from static activation-only
- median separation from KIVI-style K-only
- median separation from random-coupled control
- number and fraction of layers with accessible key/value-cache channel axes

## Decision Rule

The primary decision is based on the M18 `activation+K` arm. The
`activation+K+V` arm is a secondary diagnostic and cannot rescue a failed
primary `activation+K` gate.

### PASS_M18_JOINT_KV_ACTIVATION

Return pass if M18 `activation+K` satisfies all of:

1. median recovery is at least `0.30`;
2. bootstrap 95% CI lower bound is greater than `0.10`;
3. median recovery beats static activation-only by at least `0.15`;
4. median recovery beats KIVI-style K-only by at least `0.10`;
5. median recovery beats random-coupled activation+K control by at least
   `0.20`;
6. key-cache channel axes were accessible for at least `80%` of attention
   layers in the evaluated model.

### KILL_M18_NO_IMPROVEMENT

Return this kill if M18 `activation+K` median recovery is within `0.05` of
static activation-only.

### KILL_M18_KV_ONLY_NOT_BEATEN

Return this kill if KIVI-style K-only recovery is within `0.05` of, or higher
than, M18 `activation+K` recovery. This means the joint activation/KV coupling
did not add measurable value over a KV-only channel baseline.

### KILL_M18_RANDOM_CONTROL_BEATS

Return this kill if the random-coupled activation+K control beats M18
`activation+K` by more than `0.10` median recovery.

### KILL_M18_AMBIGUOUS

Return ambiguous kill for middle outcomes that are neither pass nor the three
specific kill modes, including outcomes where M18 improves over static but
fails CI or control-separation thresholds.

### FAIL_INFRA_M18

Return infrastructure failure for model load failure, incomplete packet, OOM
that cannot be fixed by batch-size reduction, inaccessible key/value-cache
hooks, missing required artifacts, or checker failure that prevents applying
the mechanical decision rule.

## Required Artifacts

Each M18 packet must contain:

- environment snapshot (`pip freeze`, `nvidia-smi`, CUDA/driver, git SHA)
- model provenance with HuggingFace snapshot commit
- prompt manifest and prompt SHA
- exact command line and stdout/stderr logs
- BF16 target traces or cited cache source with SHA-256
- activation-channel evidence used for static and union channel sets
- key/value-cache channel evidence used for KIVI-style K-only selection
- hook coverage manifest for attention key/value tensors
- protected-channel manifests for all regimes
- quantization configuration
- per-trace perplexity table
- per-trace recovery table
- bootstrap CI table
- checker result and artifact check
- artifact hashes

## Forbidden Actions

- Modifying prior preregistration files.
- Adjusting M18 thresholds after observing M18 data.
- Dropping the KIVI-style K-only baseline.
- Dropping the random-coupled negative control.
- Using the `activation+K+V` arm to claim pass if the primary
  `activation+K` arm fails.
- Using AWQ-style scaling or SmoothQuant scale folding.
- Replacing activation-selected channels with key-selected channels in the
  M18 joint arm.
- Modifying model source code to expose hooks.
- Selectively reporting only the best arm.

## Paper Integration Rule

If M18 passes, the paper may frame joint activation/KV channel protection as
the first positive Phase 9 method and position M2/M10/M11 as diagnostic
failures that showed why activation-only set updates were insufficient.

If M18 kills, the paper should pivot toward the negative-result framing:
strict set-leaving is robust, discontinuous set-update methods fail, smooth
activation-only updates do not clear the bar, and cross-tensor activation/KV
coupling also fails to recover the gap. Under that outcome, M12 and DecDEC
baseline remain important for comparison, but new method invention is
forbidden during the vacation window.
