# ThoughtFlow-FP8 Pre-Registration: Prefix-Surprisal Utility

Status: **PRE-REGISTERED BEFORE ONE-SHOT FRESH-SURFACE RUN.**

This artifact defines one successor utility signal after the `rdu_topk` branch was
demoted. It is intentionally different from recurrence-distance utility and from
the stopped anchor/recent/phase/math policy family.

## Paper Readiness Gate

- Current paper readiness: diagnostic only; no positive sparse-cache method is
  alive.
- Current story: a sparse-cache policy may need to retain prefix tokens that the
  model itself finds informative or hard to predict, not merely tokens with
  delayed recurrence, phase labels, hand-coded math labels, or recent position.
- Blocking gap: no fresh utility signal has survived a no-retune sparse-cache
  quality gate against R-KV-like and ThinKV-like.

## New Utility Signal

Name: **prefix-surprisal utility** (`psi_topk`)

Hypothesis: prefix tokens with high model self-surprisal carry information that
is costly to reconstruct from the remaining prefix. Retaining those tokens may
preserve details such as quantities, entity names, and unusual operators without
using labels or continuation loss.

## Signal Definition

Allowed inputs:

- tokenized prefix ids;
- the same model's prefill logits for the prefix;
- prefix length and cache budget.

Forbidden inputs:

- continuation tokens or continuation loss;
- prior sparse-cache NLL rows;
- token labels such as `anchor`, `phase`, or `math_state`;
- hand-coded recent, phase, anchor, or math-state reserves;
- attention recurrence buckets from `rdu_topk`.

For prefix token ids `x_0 ... x_{L-1}`, define token self-surprisal for position
`i` as:

```text
surprisal(i) = 0                                    if i = 0
             = -log p_model(x_i | x_0 ... x_{i-1})  otherwise
```

The policy keeps `B = ceil(0.20 * L)` positions sorted by:

```text
(-surprisal(i), i)
```

It does not add recency, sink, phase, math-state, anchor, or RDU bonuses.

## Fresh Evaluation Surface

The one-shot run uses a saved-trace surface not used by the original
ThoughtFlow/RDU frozen gates:

```text
results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl
```

The run may score fewer than 70 rows after token-length filtering. This is still
a Mac-local CPU sparse-cache quality gate, not CUDA, FP8, latency, throughput, or
systems evidence.

## Exact Run Command

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/psi_fresh_sparse_cache_check.py \
  --model-name distilgpt2 \
  --keep-fraction 0.20 \
  --max-traces 70 \
  --max-length 96 \
  --continuation-tokens 24 \
  --input-jsonl results/c2c_gsm70_20260418/qwen_gsm70_c2c.jsonl
```

## Promotion Rule

Promote `psi_topk` only if all conditions hold on the one-shot fresh surface:

- mean continuation NLL beats both `thin_kv_like` and `rkv_like` by at least
  0.03 at matched keep fraction;
- paired delta CI high versus `thin_kv_like` is below 0;
- paired delta CI high versus `rkv_like` is below 0;
- `psi_topk` is the best compressed row among stopped ThoughtFlow rows and
  cross-family baselines.

## Failure Rule

If `psi_topk` does not clear the promotion rule in this one run, this specific
signal is ruled out for the current Mac-local sparse-cache harness. Do not tune
surprisal normalization, add recency, mix with RDU, or switch trace inputs after
seeing the result.

