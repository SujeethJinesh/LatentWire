# HellaSwag Receiver-Family Packet Gate

Date: 2026-05-02

## Readiness

- Current paper readiness: COLM remains strong under a bounded fixed-byte
  source-private packet framing; ICLR remains gated by true receiver-family
  improvement, benchmark diversity, and native NVIDIA systems rows.
- Current story: a `2B` raw / `5B` framed hidden-innovation packet improves
  full HellaSwag validation for both Qwen and TinyLlama source families. This
  scout asks whether a different-family target-score receiver can use the
  TinyLlama packet.
- Exact remaining blocker: the target-aware receiver must beat both
  target-only and packet-only, not merely trust the source packet.

## Lay Explanation

The previous results showed that the tiny hidden hint is useful by itself.
This experiment asks a stricter question: if a Qwen receiver has its own
answer scores, can it combine those scores with a TinyLlama packet better than
either source alone? In plain terms, this tests whether the receiver is learning
when to listen to the sender, rather than always copying the sender's tiny
message.

## Artifact

`results/source_private_hellaswag_receiver_family_packet_gate_20260502/hellaswag_receiver_family_packet_gate.json`

Supporting files:

- `results/source_private_hellaswag_receiver_family_packet_gate_20260502/hellaswag_receiver_family_packet_gate.md`
- `results/source_private_hellaswag_receiver_family_packet_gate_20260502/manifest.json`

## Method

- Source packet: TinyLlama full-validation hidden-innovation packet.
- Target receiver: Qwen2.5 target score cache from the full-validation global
  stability artifact.
- Train split: HellaSwag validation rows `0:1024`.
- Heldout eval split: HellaSwag validation rows `1024:10042`.
- Candidate receiver rules:
  - target-margin accept-packet threshold;
  - ridge candidate receiver using target score features plus packet candidate
    indicators.
- Strict ICLR pass requires:
  - receiver beats target-only by at least `+0.02` with positive paired CI;
  - destructive packet controls stay at least `0.02` below the receiver;
  - receiver also beats packet-only by at least `+0.005` with positive paired
    CI.

## Result

Heldout rows `1024:10042` (`9018` examples):

| Row | Accuracy |
|---|---:|
| Qwen target-only | `0.483034` |
| TinyLlama packet-only | `0.629741` |
| selected receiver | `0.627190` |
| target-or-packet oracle | `0.683744` |

Paired deltas:

| Comparison | Delta | CI95 low | CI95 high |
|---|---:|---:|---:|
| receiver vs target-only | `+0.144156` | `+0.134509` | `+0.154025` |
| receiver vs packet-only | `-0.002550` | `-0.004103` | `-0.000998` |

Destructive controls under the same selected receiver rule:

| Control packet | Accuracy | Receiver minus control | CI95 low |
|---|---:|---:|---:|
| wrong-example hidden | `0.472167` | `+0.155023` | `+0.144486` |
| candidate-roll hidden | `0.396873` | `+0.230317` | `+0.219447` |
| zero-hidden | `0.569639` | `+0.057552` | `+0.050566` |
| source-label packet | `0.569639` | `+0.057552` | `+0.050119` |

Gate outcome:

- target-family transfer subgate: `true`;
- receiver-improvement subgate: `false`;
- strict ICLR receiver-family gate: `false`.

## Interpretation

This is useful but not submission-closing. A TinyLlama hidden packet is clearly
usable by a Qwen target-score receiver relative to Qwen target-only scoring,
and the destructive packet controls collapse. However, the receiver does not
beat packet-only. That means the current branch still behaves like a strong
source-side task packet rather than a receiver-side common-language interface.

The positive contribution remains alive, but the next method branch should be
receiver/common-basis oriented: the oracle says that target-or-packet selection
could reach `0.683744`, leaving about `+0.054` absolute accuracy above the
packet-only row if the receiver can learn when Qwen's own evidence should
override TinyLlama's packet.

## Contribution Status

Promoted:

1. Receiver-family target utility: a non-Qwen source packet gives a large
   heldout lift over a Qwen target-only receiver under destructive controls.
2. Reviewer boundary: this artifact separates source-packet transfer from the
   stronger claim that the receiver learns a cross-family latent language.
3. Next-method headroom: target-or-packet oracle headroom is quantified on
   `9018` heldout rows.

Still blocked:

1. A receiver-improvement method that beats packet-only.
2. A second benchmark with the same source-private packet discipline.
3. Native NVIDIA/vLLM/SGLang systems rows.

## Decision

Do not claim solved receiver-family latent communication yet. Promote this as
a falsification/diagnostic card and make the next live branch a receiver
acceptance/common-basis method that explicitly targets the `0.683744`
target-or-packet oracle headroom.
