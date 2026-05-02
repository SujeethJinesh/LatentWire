# HellaSwag Receiver Acceptance Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains defensible under the fixed-byte
  source-private packet story; ICLR still needs a positive receiver method,
  benchmark diversity, and native NVIDIA systems rows.
- Current story: Qwen and TinyLlama each support a full-validation HellaSwag
  `2B` raw / `5B` framed hidden-innovation packet. The previous headroom
  decomposition showed a TinyLlama/Qwen oracle at `0.692947`, but no fair
  receiver had captured that headroom.
- Exact remaining blocker: a train-only receiver must beat TinyLlama
  packet-only with positive paired uncertainty.

## Lay Explanation

This experiment asks whether a receiver can learn when to ignore the
TinyLlama packet and use Qwen's own candidate instead. The receiver sees only
an early prefix during training, chooses thresholds on a prefix dev split, and
then applies the frozen rule to later heldout rows. The ridge row is a learned
error predictor. The relative-kNN row is the common-basis version: it asks
whether rows near similar training anchors should make the same override
decision.

## Artifact

`results/source_private_hellaswag_receiver_acceptance_gate_20260502/hellaswag_receiver_acceptance_gate.json`

Supporting files:

- `results/source_private_hellaswag_receiver_acceptance_gate_20260502/hellaswag_receiver_acceptance_gate.md`
- `results/source_private_hellaswag_receiver_acceptance_gate_20260502/manifest.json`

## Method

- Source packet: TinyLlama full-validation selected hidden-innovation packet.
- Receiver alternatives:
  - Qwen target-score top-1;
  - Qwen mean-zscore hidden packet;
  - Qwen hybrid vote-on-score-agreement hidden packet.
- Train prefixes: `2048`, `4096`, and `6144` rows.
- Each prefix uses the first `75%` for fitting and the remaining `25%` for
  dev threshold/model selection.
- Heldout suffix begins after the train prefix.
- Feature views:
  - score-only packet/target rank and confidence features;
  - score plus Qwen hidden-confidence features.
- Receiver families:
  - benefit ridge: predict whether overriding the packet helps or harms;
  - relative-kNN benefit: estimate override value from train-only nearest
    anchors in normalized feature space.

Strict promotion requires the predeclared default receiver to beat packet-only
by at least `+0.005` with positive paired CI95 lower bound, beat target-only
by at least `+0.02`, and stay positive across contiguous heldout blocks.

## Result

Gate outcomes:

- pass gate: `false`;
- best-scout pass gate: `false`;
- predeclared default pass gate: `false`;
- target-transfer gate: `true`;
- block-stability gate: `false`.

Headline rows:

| Row | Train prefix | Accuracy | Packet-only | Delta | CI95 low |
|---|---:|---:|---:|---:|---:|
| predeclared default | `2048` | `0.645234` | `0.646110` | `-0.000876` | `-0.001876` |
| best scout | `4096` | `0.672217` | `0.671376` | `+0.000841` | `-0.001345` |

Best scout details:

- method: benefit ridge;
- alternative: Qwen hybrid vote-on-score-agreement packet;
- feature view: score-only;
- eval override rate: `0.010764`;
- heldout improvement is too small and uncertain to promote.

Default heldout block deltas versus packet-only:

| Block | Rows | Delta |
|---:|---:|---:|
| `0` | `1599` | `-0.001251` |
| `1` | `1599` | `0.000000` |
| `2` | `1599` | `-0.001251` |
| `3` | `1599` | `+0.000625` |
| `4` | `1598` | `-0.002503` |

Systems/accounting row:

- packet remains `2B` raw / `5B` framed;
- no source text, source KV, raw hidden vectors, or raw score vectors are
  transmitted;
- receiver selector is Mac-local and cache-based;
- native GPU serving claims remain disabled.

## Interpretation

This is a branch-kill result for simple receiver acceptance. The oracle
headroom is real, but score geometry, hidden-confidence features, ridge
benefit prediction, and nearest-anchor relative selectors do not recover it
under train-only selection. The best observed scout lift is only `+0.000841`
and has a negative CI95 lower bound, so claiming receiver improvement would be
indefensible.

The next receiver branch should change the information structure, not tune
thresholds harder. The highest-value next steps are:

1. official-train calibration so the receiver has more supervised rows without
   consuming the validation decision surface;
2. a sparse/crosscoder or relative-representation common basis that learns
   reusable disagreement atoms rather than local confidence thresholds;
3. a tiny learned query-bottleneck receiver, Q-Former/Perceiver style, with
   source-destroying controls and heldout-only model selection.

## Contribution Status

Promoted:

1. A reviewer-facing negative result that rules out simple selective
   prediction as the missing common-language mechanism.
2. A stricter receiver decision surface with train/dev/eval separation and
   paired uncertainty.
3. A Mac-local systems-side receiver artifact preserving the fixed-byte
   sideband boundary.

Still blocked:

1. Positive receiver improvement over packet-only.
2. Second benchmark under the same packet discipline.
3. Native NVIDIA/vLLM/SGLang systems rows and direct C2C/KVComm/KV-quant
   comparisons.

## Decision

Do not continue tuning score/hidden-confidence selectors on this decision
surface. The next exact gate should generate official-train receiver
calibration artifacts or implement a learned sparse/query-bottleneck receiver
with explicit source-destroying controls.
