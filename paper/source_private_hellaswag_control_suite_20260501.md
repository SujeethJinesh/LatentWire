# HellaSwag Label-Copy Control Suite

Date: 2026-05-01

## Status

This branch is weakened, not promoted. The `2B` HellaSwag packet still beats
target-only, destructive controls, and the old same-byte choice-prefix text
relay, but it does not beat the stricter one-byte source-label-copy control.

## Why We Ran This

The previous HellaSwag result looked strong because a two-byte private packet
beat a two-byte text prefix from the source-selected ending. A reviewer would
ask a sharper question: is the packet doing anything more than saying "the
source model picked option B"?

In plain terms, we tested whether the fancy tiny hint beats simply sending the
letter of the source model's favorite answer. It does not yet.

## Control Suite

Artifact:
`results/source_private_hellaswag_control_suite_20260501/`

Frozen validation slice: first `1024` official HellaSwag validation rows.

| Condition | Accuracy |
|---|---:|
| matched source-private packet | `0.460` |
| source-label text copy | `0.462` |
| same-activity shuffled source packet | `0.245` |
| same-split-type shuffled source packet | `0.245` |
| activity-label train-majority prior | `0.237` |
| split-type train-majority prior | `0.233` |

Readout:

- matched minus best metadata/activity control: `+0.215`
- matched minus source-label copy: `-0.002`
- paired CI95 versus source-label copy: `[-0.005, 0.000]`
- metadata/activity controls clean: `true`
- strict non-label-copy pass gate: `false`

Interpretation: the HellaSwag slice is not explained by activity labels,
split-type labels, or same-activity packet leakage. The remaining problem is
more direct: copying the source model's selected option is enough to match the
packet.

## Score-Packet Probe

Artifact:
`results/source_private_hellaswag_score_packet_headroom_20260501_qwen05_validation1024/`

The score probe asked whether a tiny confidence packet can repair source
top-label mistakes by switching to the runner-up when the source model is
uncertain.

| Readout | Heldout accuracy |
|---|---:|
| source-label text | `0.465` |
| simple top-2 margin threshold packet | `0.467` |
| best rank-bin score packet | `0.465` |
| top-2 oracle | `0.734` |

Readout:

- score-packet minus source-label text: `+0.002`
- best rank-bin minus source-label text: `0.000`
- selected margin threshold: `0.002`
- source scoring latency on Mac CPU: `505.24s`
- pass gate: `false`

Interpretation: the source model often has the right answer somewhere in its
top two choices, but the top-vs-runner-up margin is not enough to decide when
to switch. This keeps the method branch alive as a research surface, but the
next method must carry a better error-repair signal than the top label plus one
confidence number.

## Scratch Repair Selector

Artifact:
`results/source_private_hellaswag_repair_selector_scratch_20260501/`

As a final cheap DFS probe, I fit a shared linear repair selector on even rows
of the frozen validation slice and evaluated it on odd rows. It used source
top-2 score features plus public context/choice lexical features. This is not a
valid final benchmark protocol because it calibrates on validation labels; it is
only a quick test of whether the obvious feature family has signal.

Result:

- source-label heldout accuracy: `0.465`
- best repair-selector heldout accuracy: `0.467`
- delta: `+0.002`
- pass gate: `false`

Interpretation: margin, entropy, rank, choice length, and simple context
overlap do not explain enough source mistakes. The next HellaSwag method branch
needs a different source-error signal.

## Train-Only Public Receiver Repair Probe

Artifact:
`results/source_private_hellaswag_public_receiver_repair_probe_20260501_qwen05_validation1024/`

This follow-up moved the repair probe off validation-label calibration. A
hashed lexical receiver scorer was trained only on official HellaSwag train
rows, selected by an internal train/dev split, and evaluated once on the frozen
`1024` validation rows using the cached source top-2 scores.

| Condition | Accuracy |
|---|---:|
| source-label copy | `0.462` |
| public target-only receiver | `0.260` |
| top-2 public rerank | `0.364` |
| public-if-in-source-top-2 gate | `0.413` |

Readout:

- selected internal dev accuracy: `0.309`
- best repair minus source-label copy: `-0.049`
- source top-2 oracle accuracy: `0.716`
- pass gate: `false`

Interpretation: public train-only lexical repair does not recover the top-2
headroom. The next valid branch needs train-split source scores or source
hidden summaries for calibrated source-error repair, not more eval-only
thresholding of public features.

## Train-Source-Score Repair Probe

Artifact:
`results/source_private_hellaswag_train_source_score_repair_probe_20260501_qwen05_train512_validation1024/`

This follow-up scored `512` official train rows with the same source model and
selected a repair policy only on an internal `384/128` train/dev split. The
selected policy was `top2_margin_8bin`, a two-byte packet that sends source
rank order plus a quantized score-shape code.

| Condition | Accuracy |
|---|---:|
| source-label copy | `0.462` |
| trained choice-bias label-copy | `0.459` |
| selected train-source-score repair | `0.447` |

Readout:

- train source scoring latency on Mac CPU: `412.72s`
- selected internal dev accuracy: `0.461`
- selected minus best label-copy: `-0.015`
- source top-2 oracle accuracy: `0.716`
- source top-4 oracle accuracy: `1.000`
- pass gate: `false`

Interpretation: source-score shape learned from train rows still does not
identify the source model's top-choice errors. This weakens score-only repair
and pushes the next HellaSwag gate toward source hidden summaries or residual
codes rather than larger margin bins.

## Decision

Promote only as:

- HellaSwag-specific diagnostic surface
- metadata/activity leakage falsification
- evidence of large top-2 repair headroom
- systems evidence that the packet interface remains byte-small

Do not promote as:

- strict non-label-copy HellaSwag result
- final ICLR headline benchmark
- proof of cross-model latent reasoning
- superiority to C2C, KVComm, KVCOMM, QJL, TurboQuant, or prompt compression

## Next Gate

Build a calibrated source-error repair packet on HellaSwag:

- train only on official HellaSwag train rows
- send top-2 or top-4 source choices plus a learned repair bit or small
  quantized repair sketch
- evaluate on the same frozen `1024` validation slice
- require heldout accuracy to beat source-label text by at least `0.02`
- then rerun five projection/packet seeds and the metadata/source-label
  controls before widening to full validation

The most promising branch is not "more bytes"; it is a better decision rule
for when the source model's top answer should be distrusted.
