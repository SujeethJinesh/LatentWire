# HellaSwag Switch Decomposition

## Status

This gate kills top-2 trust-or-switch as a standalone contribution for the
current paper. The source top-2 oracle has large headroom, but the train-only
switch model cannot reliably identify when to leave source top-1 for source
top-2.

Current paper story: LatentWire should keep the candidate-wise hidden-innovation
packet as the main HellaSwag branch. The switch decomposition becomes a reviewer
diagnostic showing that the positive dense/hybrid packet is not merely a
simple top-2 switch policy.

## Lay Explanation

The sender often has the correct answer as its second guess. This experiment
asks whether we can learn a tiny rule that says, "trust the first guess here,
but use the second guess there." It failed: the learned rule switches too often
and is wrong too often.

## Artifact

`results/source_private_hellaswag_switch_decomposition_20260502/hellaswag_switch_decomposition.json`

## Gate Definition

The selected switch view is chosen only on the 512-row train/dev split. It must
pass both validation-first1024 and terminal-tail:

- beat best label-copy by at least `+0.02`;
- paired CI95 low vs best label-copy must be positive;
- use hidden-private evidence, not only scores;
- beat score-only switch by at least `+0.01`;
- beat matched-rate random switching by at least `+0.02`;
- switch precision at least `0.60`;
- recall over gold-source-top2 rows at least `0.25`;
- capture at least `0.20` of top-2 oracle headroom;
- wrong-hidden and label-permuted controls lag selected by at least `0.01`.

## Results

Selected train-dev switch: `hidden_score_switch`, ridge `100.0`, internal-dev
accuracy `0.523438`.

### Validation[0:1024]

- source label-copy: `0.461914`
- source top-2 oracle: `0.715820`
- always switch to top-2: `0.253906`
- matched-rate random switch: `0.378906`
- score switch: `0.430664`
- selected hidden+score switch: `0.449219`
- selected delta vs best label-copy: `-0.012695`
- selected switch precision: `0.333333`
- selected recall over gold-top2 rows: `0.453846`
- selected headroom capture: `-0.050000`
- dense hidden-innovation reference: `0.512695`
- hybrid hidden-vote reference: `0.518555`

### Validation[9216:10042] Terminal Tail

- source label-copy: `0.497579`
- source top-2 oracle: `0.756659`
- always switch to top-2: `0.259080`
- matched-rate random switch: `0.429782`
- score switch: `0.498789`
- selected hidden+score switch: `0.485472`
- selected delta vs best label-copy: `-0.012107`
- selected switch precision: `0.353175`
- selected recall over gold-top2 rows: `0.415888`
- selected headroom capture: `-0.046729`
- dense hidden-innovation reference: `0.539952`
- hybrid hidden-vote reference: `0.547215`

## Decision

Promoted:

1. Top-2 oracle headroom is large: roughly `+0.254` on first1024 and `+0.259`
   on the terminal tail.
2. The candidate-wise dense/hybrid hidden-innovation packet remains the stronger
   method branch: it improves both slices while the pure switch fails.
3. The decomposition is useful reviewer evidence: the current HellaSwag gain is
   not explained by an obvious top-2 switching shortcut.

Ruled out:

1. Top-2 trust-or-switch as one of the three paper contributions.
2. A selective-classification framing for the current method.
3. More investment in switch-only policies without a new representation or
   training target.

## Reviewer-Framing Boundary

Selective classification and learning-to-defer already study confidence-based
accept/switch policies. Our failed switch gate means we should not present this
as a new selective prediction method. Sources:
https://arxiv.org/abs/1705.08500, https://arxiv.org/abs/1901.09192, and
https://arxiv.org/abs/1711.06664.

Prefix/prompt/adapters learn persistent conditioning or model parameters, while
this gate transmits only a fixed-byte decision packet and does not modify the
receiver. Sources: https://arxiv.org/abs/2101.00190,
https://arxiv.org/abs/2104.08691, https://arxiv.org/abs/2110.07602, and
https://arxiv.org/abs/1902.00751.

C2C and KVComm remain the close non-text communication baselines because they
share or reuse source-side compute state. This gate only uses a `2B` raw /
`5B` framed packet, but native systems comparisons remain pending. Sources:
https://arxiv.org/abs/2510.03215 and https://arxiv.org/abs/2510.03346.

## Next Gate

Cut trust-or-switch from the contribution list. The next highest-value gate is
candidate-wise hidden-innovation stability:

1. rerun dense/hybrid candidate-wise denoising with a stricter no-eval-leak
   train/dev selection surface;
2. require terminal-tail jackknife stability;
3. then run one strict cross-family falsification pair;
4. only after that widen to native NVIDIA systems rows.
