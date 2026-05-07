# ThoughtFlow-FP8 Pre-Registration: Value-Weighted Attention Contribution

Status: **SUPERSEDED / KILLED AFTER ONE-SHOT FRESH-SURFACE RUN**. This was
pre-registered before measurement and is preserved for audit only; the
`vwac_topk` branch failed its fresh-surface gate and is not a live utility.

This artifact defines one successor utility after the stopped `rdu_topk` and
`psi_topk` branches. It is intentionally different from RDU lag-bucket
recurrence, prefix self-surprisal, and the anchor/recent/phase/math policy
family.

## New Utility Signal

Name: **value-weighted attention contribution** (`vwac_topk`)

Hypothesis: a retained token should be useful when later prefix queries attend
to it and its value vector has nontrivial magnitude. `rdu_topk` counted delayed
attention recurrence; `vwac_topk` instead estimates cache-state contribution:

```text
vwac(i) =
  sum_{m,h,q>i} A[m,h,q,i] * ||V[m,h,i]||_2
  / max(1, sqrt((L-i-1) * M * H))
```

where `A[m,h,q,i]` is full-prefix prefill attention and `V[m,h,i]` is the cached
value vector for layer `m`, head `h`, token `i`.

The policy keeps `B = ceil(0.20 * L)` positions sorted by:

```text
(-vwac(i), i)
```

## Inputs and Forbidden Inputs

Allowed inputs:

- tokenized prefix ids;
- same-model prefill attentions;
- same-model prefill cached value vectors;
- prefix length and cache budget.

Forbidden inputs:

- continuation tokens or continuation loss;
- previous sparse-cache NLL rows;
- token labels such as `anchor`, `phase`, or `math_state`;
- hand-coded recent, phase, anchor, math-state, RDU, or PSI bonuses;
- any sweep over normalization, value norm, or attention variants.

## Fresh Evaluation Surface

The one-shot run uses a saved-trace surface not used by the RDU or PSI promotion
gates:

```text
results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl
```

This is still a Mac-local CPU sparse-cache quality gate, not CUDA, FP8, latency,
throughput, or systems evidence.

## Exact Run Command

```bash
TRITON_CPU_BACKEND=1 TRITON_INTERPRET=1 TRITON_HOME="$PWD/.debug/triton_home" \
  ./venv_arm64/bin/python experimental/thoughtflow_fp8/phase2/vwac_fresh_sparse_cache_check.py \
  --model-name distilgpt2 \
  --keep-fraction 0.20 \
  --max-traces 70 \
  --max-length 96 \
  --continuation-tokens 24 \
  --input-jsonl results/c2c_svamp70_20260418/qwen_svamp70_c2c.jsonl
```

## Promotion Rule

Promote `vwac_topk` only if all conditions hold on the one-shot fresh surface:

- mean continuation NLL beats both `thin_kv_like` and `rkv_like` by at least
  0.03 at matched keep fraction;
- paired delta CI high versus `thin_kv_like` is below 0;
- paired delta CI high versus `rkv_like` is below 0;
- `vwac_topk` is the best compressed row among stopped ThoughtFlow rows and
  cross-family baselines.

## Failure Rule

If `vwac_topk` does not clear the promotion rule in this one run, this exact
signal is ruled out for the current Mac-local sparse-cache harness. Do not tune
the formula or rerun it on a different surface after seeing the result.
