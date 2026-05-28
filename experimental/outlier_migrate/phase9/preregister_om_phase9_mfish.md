# Phase 9 M-FISH Preregistration

Created: 2026-05-28T03:05Z

## Hypothesis

Magnitude-only protected-channel selection misses channels that are small but
loss-sensitive. A diagonal Fisher estimate at the decode-time layer-output
surface can improve channel protection on dense Transformer models where M11b
is positive but ambiguous.

## Scope

Primary gate model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`.

Hybrid Mamba/MoE models are excluded from this gate because scratch probes
showed their cache implementations mutate state tensors in-place during
gradient-enabled decode steps, breaking autograd. That infrastructure issue is
documented in `docs/mfish_gradient_feasibility.md`.

## Method

For each calibration prompt and selected decode positions near the scoring
window, build a detached prefix cache under `torch.no_grad()`. For the sampled
decode step, run one gradient-enabled forward pass with frozen model weights
and retain gradients on layer-output activations. Accumulate

`F_{\ell,c} = E[(d loss / d x_{\ell,c})^2]`

per layer and channel.

M-FISH uses the same budget-tuned indicator EMA as M11b, but replaces the
per-position magnitude ranking with a Fisher-weighted ranking:

`score_{\ell,c}(t) = F_{\ell,c} * E[|x_{\ell,c}(t)|]`.

Protected sets are the final EMA snapshots at top-5% and top-10% budgets.

## Regimes

- `bf16` reused from the DeepSeek M11b packet.
- `static_1pct` reused from the DeepSeek M11b packet.
- `m11b_top5` reused from the DeepSeek M11b packet.
- `m11b_top10` reused from the DeepSeek M11b packet.
- `static_top10` reused from the DeepSeek M11b packet.
- `mfish_top5` newly scored.
- `mfish_top10` newly scored.
- `random_fisher_top10` newly scored after shuffling Fisher weights per layer.

## Decision Criteria

- `PASS_MFISH_ARCHITECTURE_FILL`: `mfish_top10` beats `m11b_top10` by at
  least 0.05 median recovery, has median recovery above 0.30, and has CI lower
  bound above 0.
- `PASS_MFISH_STRONG`: `mfish_top10` median recovery exceeds 0.55 with CI lower
  bound above 0.20.
- `AMBIGUOUS_MFISH`: M-FISH is positive but overlaps M11b or random-Fisher.
- `KILL_MFISH`: M-FISH is no better than M11b or loses to random-Fisher.

## Scoop Check

Fresh web search found FGMP (arXiv:2504.14152), which uses Fisher-weighted
perturbation to select mixed-precision weight and activation blocks, and CMPQ
(arXiv:2410.13056), which allocates channel-wise mixed precision from
activation distributions. M-FISH is narrower: decode-time, W4A16,
layer-output-channel protection under long reasoning traces, using
one-step Fisher gradients to re-rank protected channel sets.
