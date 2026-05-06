# ThoughtFlow-FP8 Pre-Registration: Recurrence-Distance Utility

Status: **SUPERSEDED: frozen probe and reproduction checks were run; current
decision is diagnostic-only.**

This artifact defines exactly one new utility signal for a future one-shot
successor evaluation. It is intentionally bounded because the current
anchor/recent/phase/math policy family has stopped on the available saved
traces.

## Paper Readiness Gate

- Current paper readiness: not ICLR-ready; still missing a positive method that
  survives frozen sparse-cache quality against ThinKV-like and R-KV-like.
- Current story: retrofit sparse-KV compression may need to protect tokens that
  become useful again after low-attention intervals, rather than only protecting
  anchors, recent tokens, phase markers, or math-state labels.
- Blocking gap: no pre-registered successor has yet shown matched-budget
  continuation-quality gains with paired uncertainty on the frozen sparse-cache
  surface.

## Prior Evidence Used

The frozen 74-trace CPU sparse-cache probe weakened the saved-trace
anchor/recent/phase/math family:

- `thin_kv_like`: NLL 3.900
- `tf_sparse_r0.55_p0.05_m0.12_a2`: NLL 3.908
- `thoughtflow_saliency_recent`: NLL 3.920
- `rkv_like`: NLL 3.939

Promotion required a frozen ThoughtFlow row to beat both R-KV-like and
ThinKV-like by at least 0.03 NLL with paired CIs below zero. No row cleared it.

Saturated:

- synthetic marker retention
- text-prefix-only policy tuning
- anchor/recent/phase/math sparse-cache sweeps on these saved traces
- phase-marker recall without continuation-quality improvement

Still alive:

- CPU sparse-cache pruning as a falsification harness
- hidden/KV telemetry for explaining eviction bias
- one pre-registered successor signal evaluated once

## New Utility Signal

Name: **recurrence-distance utility** (`rdu_topk`)

Hypothesis: a useful sparse cache should retain prefix tokens that later prefix
queries repeatedly re-attend to after nontrivial gaps. This targets latent
future-use recurrence, not token class labels. It is motivated by the
recurrence failure mode emphasized by LazyEviction and by future-attention
framing in ForesightKV, while staying training-free and prefill-only.

Primary source anchors:

- LazyEviction identifies token-importance recurrence and uses lagged eviction
  to preserve recurring tokens during low-attention intervals:
  https://openreview.net/pdf?id=Mac3RzkEQu
- ForesightKV frames eviction around long-term contribution using future
  attention, but trains a predictor; this pre-registration tests a training-free
  prefix-only proxy instead: https://arxiv.org/abs/2602.03203
- StreamingLLM motivates why pure recent-window cache retention is insufficient
  without stable retained tokens, but this signal does not hard-code sink or
  recent-token rules: https://arxiv.org/abs/2309.17453

## Signal Definition

Allowed inputs:

- tokenized prefix ids
- the same model's prefill attentions for that prefix with
  `output_attentions=True`
- prefix length and cache budget

Forbidden inputs:

- continuation tokens or continuation loss
- trace outcome/NLL from any previous frozen sparse-cache run
- token labels such as `anchor`, `phase`, or `math_state`
- hand-tuned recent, phase, math, or anchor weights

For a prefix of length `L`, budget `B = ceil(0.20 * L)`, layer count `M`, head
count `H`, and attention tensor `A[m,h,q,i]` from query position `q` to key
position `i`, define lag buckets:

- `b0 = [8, 15]`
- `b1 = [16, 31]`
- `b2 = [32, 63]`
- `b3 = [64, infinity)`

For each token position `i` and bucket `b`, let:

```text
mass(i,b) =
  sum_{m,h,q: q > i and q - i in b} A[m,h,q,i]
  / max(1, sqrt(count_{q: q > i and q - i in b} * M * H))
```

Then:

```text
rdu(i) = max_b mass(i,b) + 0.5 * second_largest_b mass(i,b)
```

Tie break:

```text
sort by (-rdu(i), i)
```

The policy keeps the first `B` sorted token positions. It does not add sink,
recent, phase, math-state, or hidden-norm bonuses.

## Exact Policy Transformation

Add one frozen policy named `rdu_topk` to the frozen sparse-cache probe. The
policy must compute `rdu(i)` from the full prefix prefill attentions before
pruning `past_key_values`, then keep exactly the top-budget positions.

No sweep is allowed. The lag buckets and second-bucket multiplier above are
fixed. Any implementation bug fix must preserve this formula or invalidate this
pre-registration.

## Frozen Evaluation Command

Historical command, retained for reproducibility of the stopped branch:

```bash
cd /Users/sujeethjinesh/Desktop/LatentWire
./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/frozen_sparse_cache_probe.py \
  --model-name distilgpt2 \
  --keep-fraction 0.20 \
  --max-traces 74 \
  --max-length 96 \
  --continuation-tokens 24
```

The run must report `rdu_topk` beside the existing frozen baselines and must not
select among variants.

## Promotion Rule

Promote the branch only if `rdu_topk` satisfies all conditions on the frozen
74-trace sparse-cache probe:

- mean continuation NLL beats both `thin_kv_like` and `rkv_like` by at least
  0.03 at matched keep fraction
- paired delta CI high versus `thin_kv_like` is below 0
- paired delta CI high versus `rkv_like` is below 0
- per-span telemetry is reported separately for anchors, phase markers,
  math-state tokens, and recurrence-distance buckets

## Failure Rule

If `rdu_topk` does not clear the promotion rule in the first frozen run, this
specific signal is ruled out on the current Mac-local distilgpt2 saved-trace
surface. Do not tune lag buckets, add recency reserves, or combine it with the
stopped anchor/recent/phase/math family on these traces.

## Bounds

- Exactly one signal: recurrence-distance utility.
- Exactly one frozen evaluation after implementation.
- No further saved-trace tuning before that evaluation.
- No GPU, FP8, kernel, long-context, or paper-claim work until the promotion
  rule clears.
