# Source Surface Reselection And SVAMP70 Cross-Attention Follow-Up

- date: `2026-04-26`
- status: `no_live_learned_prefix_branch`
- base commit: `e5a7be224d2ce53bfca747a5106add17a4f74b03`

## Surface Reselection

After SVAMP32 summary-prefix and cross-attention gates failed, I rescanned the
available exact-ID surfaces.

Top surfaces:

| Surface | Status | Source-only | Target/source oracle |
|---|---|---:|---:|
| `svamp70_live` | strong source-complementary | 9 | 30/70 |
| `svamp70_holdout` | strong source-complementary | 6 | 14/70 |
| `svamp32_qwen25math` | weak source-complementary | 5 | 13/32 |

GSM70 and the alternate SVAMP32 surfaces remain weak immediate candidates.

## SVAMP70 Cross-Attention Rescue

I then reran the token-local cross-attention logprob gate on `svamp70_live`,
the highest-ranked source surface.

Result:

- clean IDs scored: `6`
- matched-only clean IDs: `0/6`
- matched-positive clean IDs: `3/6`
- clean control leaks: `3/6`
- mean matched-minus-best-control clean margin: `-0.443233`
- target-preservation IDs scored: `22`
- target-preservation matched-positive count: `13/22`

Clean rows:

| Example ID | Matched Margin | Best Control | Best Control Margin | Delta |
|---|---:|---|---:|---:|
| `14bfbfc94f2c2e7b` | -3.733 | label_shuffled | -3.540 | -0.193 |
| `2de1549556000830` | -1.381 | zero_source | -0.884 | -0.497 |
| `4d780f825bb8541c` | 0.591 | target_only_prefix | 1.628 | -1.037 |
| `41cce6c6e6bb0058` | 2.749 | label_shuffled | 3.392 | -0.643 |
| `ce08a3a269bf0151` | 0.807 | shuffled_source | 0.919 | -0.111 |
| `bd9d8da923981d69` | -4.560 | zero_source | -4.383 | -0.177 |

## Decision

Do not spend more compute on tiny learned prefix emitters for this story unless
a new mechanism explains the persistent target/control dominance. The
SVAMP70 surface is stronger than SVAMP32, but the same source-control failure
appears there.

## Next Gate

The next highest-value branch is not another prefix-emitter variant. It should
be either:

- a discrete source-derived candidate/routing stack evaluated on `svamp70_live`
  and immediately validated on `svamp70_holdout`, with process repair only as a
  target-side baseline/confound; or
- new source-surface discovery looking for stronger source-only headroom before
  method spend.

Artifacts:

- `results/source_surface_reselection_20260426/manifest.md`
- `results/source_surface_reselection_20260426/source_headroom_surfaces.md`
- `results/qwen25math_svamp70_source_cross_attention_logprob_20260426/manifest.md`
- `results/qwen25math_svamp70_source_cross_attention_logprob_20260426/live_smoke.md`
