# HellaSwag Qwen-To-Phi Harm-Controlled Bucket Gate

## Status

- Paper readiness: COLM workshop remains plausible; ICLR full paper remains
  blocked.
- Current paper story: source-private fixed-byte packets can transfer useful
  candidate evidence, but learned cross-family receivers have not yet produced
  a positive method.
- Exact blocker: no train-only receiver has beaten fixed Qwen hybrid on the
  frozen Qwen-to-Phi HellaSwag `1024:2048` surface with paired uncertainty and
  destructive controls.

## Gate

This gate tests the most conservative follow-up to the near-miss
official-train receiver-calibrated run. It defaults to fixed Qwen hybrid and
accepts a Qwen-rival or Phi-top1 override only when an official-train bucket has
enough support, positive lower-bound mean benefit, and low observed harm.

Unlike the previous linear receiver, this gate does not expose raw Qwen score
vectors to the receiver. The source-side encoder may use Qwen scores only to
emit quantized packet fields:

- hybrid candidate id;
- rival candidate id;
- Qwen margin bin;
- Qwen rival-advantage bin;
- selected-margin bin;
- Qwen top1 relation;
- Qwen mean relation.

Phi scores remain receiver-local side information. The packet is `3B` raw /
`6B` framed. It transmits no source text, KV cache, hidden vector, raw scores,
or logits.

## Result

The gate fails by selecting no safe override buckets.

| metric | value |
|---|---:|
| official-train calibration rows | `1487` |
| official-train fit/dev rows | `1115 / 372` |
| eval rows | `768` |
| fixed Qwen hybrid | `0.467448` |
| harm-controlled bucket packet | `0.467448` |
| delta vs fixed hybrid | `0.000000` |
| CI95 low vs fixed hybrid | `0.000000` |
| overrides / helps / harms | `0 / 0 / 0` |
| selected scheme | `no_op` |
| selected eligible buckets | `0` |
| hybrid/rival/Phi oracle | `0.766927` |
| oracle helps over fixed hybrid | `230` |

The strongest non-noop official-dev configurations were not promotable: the
top configurations each made one override on official dev and harmed it
(`0` helps, `1` harm, delta `-0.002688`). This is the key evidence: even after
restricting to low-harm buckets, official-train calibration does not identify
reliable complementarity.

Slice rows are exactly tied to fixed hybrid because the selected rule is no-op:

| slice | rows | fixed | method | delta |
|---:|---:|---:|---:|---:|
| `1024` | `384` | `0.486979` | `0.486979` | `0.000000` |
| `1536` | `384` | `0.447917` | `0.447917` | `0.000000` |

Controls remain useful as a sanity check. The source-score-row-shuffle control
ties fixed hybrid only because the selected model is no-op. Label permutation,
source-row shuffle, code-value permutation, random same-byte source, and
candidate-roll source all collapse below fixed hybrid.

## Decision

Weaken the current receiver-family branch. The sequence now looks consistent:

1. protected-rival/top-2 packet has large oracle but learned decoders harm;
2. official-train source dictionary harms badly;
3. receiver-calibrated linear model nearly ties but still harms;
4. harm-controlled buckets select no safe overrides.

The live problem is no longer "find a slightly better shallow switcher over
this packet." The next highest-value branch should be a different interface:

- target self-resonance / self-compression first, proving Phi can be steered by
  compact learned tokens relative to its own full-text state; or
- decision-supervised sparse/common-dictionary intervention with atom-shuffle,
  wrong-row, target-cache-only, and top-atom knockout controls.

For ICLR, this result is useful only as a rigorous negative diagnostic. It
should not be framed as the positive method.

## Lay Explanation

We asked whether Phi could learn a very cautious rule: keep Qwen's safe answer
unless the training data shows that a specific kind of tiny Qwen message almost
always helps. The cautious rule found no safe message types. That means this
particular receiver strategy is not strong enough; we need a richer way for the
models to communicate than a shallow candidate-switch rule.
