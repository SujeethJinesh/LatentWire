# Code Audit: Correctness

Date: 2026-05-25

Scope: `release/src/outlier_migrate/`, method implementations, metrics, and tests.
This audit was performed locally after the planned subagent failed due usage limits.

## Findings

| Severity | Finding | Status |
|---|---|---|
| CRITICAL | Reproduction scripts previously returned frozen claim values even when run without `--fast-verify`, which could be mistaken for full GPU reproduction. | Fixed by `build_reproduction_payload`: dry-run validates config, fast-verify replays frozen claims, and full mode raises a clear error. |
| SUBSTANTIAL | `release/` implements minimal mask builders and metric helpers, not the full experimental GPU packet pipeline. | Documented in README, architecture overview, reproducing guide, and verification log. |
| MINOR | `m26.stable_core_mask` uses top stability counts rather than a literal all-position intersection when padding is needed. | Acceptable for the release helper; the paper claim is sourced to experimental packet outputs, not this helper. |

## Method Checks

- `static_topk` enforces `1 <= budget <= channel_count`, returns a boolean mask,
  and protects exactly `budget` channels.
- `decdec.reactive_topk` delegates to `static_topk`, matching the paper's
  algorithmic proxy: current-step top-k with no temporal smoothing.
- `m11b.ema_scores` implements `alpha * current + (1-alpha) * previous` with
  shape validation; `m11b_topk` applies the top-k budget to EMA scores.
- `m26.stable_core_mask` protects the most stable channels by count, matching
  the padded stable-core release abstraction.
- `paroquant.givens_pair_rotation` applies a standard two-column Givens
  rotation. The release does not claim to reproduce upstream optimized kernels.

## Metric Checks

- `recovery_fraction` implements `1 - (candidate - bf16)/(static - bf16)` and
  raises on zero BF16-vs-static gap, matching the positive-gap-only convention.
- `set_leaving_rate` computes `|base \ later| / |base|` and raises on empty
  base set.
- `kl_divergence` normalizes clipped probability vectors before `KL(P||Q)`.
- `bootstrap_ci` computes percentile bootstrap intervals for the median with a
  configurable seed.

## Manual Edge Cases

- Position-zero / first-step behavior: represented by direct score arrays; no
  stateful release method silently initializes hidden state.
- Uniform inputs: top-k methods use `np.argpartition`, so ties are deterministic
  for a given NumPy build but not semantically meaningful; tests only require
  exact budget, not tie order.
- Single-channel cases: methods reject zero or overlarge budgets and work for
  `budget == channel_count`.

## Tests

`./.venv_gpu/bin/python -m pytest release/tests` passed with `7 passed`.
