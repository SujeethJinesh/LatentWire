# ARC Tokenwise Repair Readout Preflight

- created UTC: `2026-05-05T00:04:50Z`
- COLM_v2 readiness effect: useful negative/guardrail row
- ICLR readiness effect: still blocked
- gate: `arc_tokenwise_repair_readout_preflight`
- status: `failed`

## Current Story

The ICLR direction is to distill useful dense C2C-style transfer into sparse,
source-private packets. Before attempting another target soft-prefix receiver,
this gate asks whether Qwen2.5 source token traces contain enough row-specific
signal for a tiny held-out repair readout to fix Qwen3 target errors.

## What Ran

Artifact:
`results/source_private_arc_tokenwise_repair_readout_preflight_20260504_n32_qwen05_to_qwen3/arc_tokenwise_repair_readout_preflight.json`

The gate reused the existing n32 ARC/OpenBookQA target-score audit and avoided
the Qwen3/MPS prefix path. It extracted answer-key-forbidden Qwen2.5 token
hidden pools on CPU, trained a ridge candidate readout on 16 fit rows, and
evaluated on 16 audit-aligned rows.

Strict controls included target-only, public-only readout, zero-source,
wrong-row source, source-row shuffle, same-source-choice wrong-row, atom
shuffle, coefficient shuffle, candidate roll, source-index, and same-byte
visible text.

## Result

| condition | accuracy | mean margin | matched delta CI |
|---|---:|---:|---:|
| matched tokenwise repair readout | 0.3125 | -0.1177 | - |
| target only | 0.2500 | -0.1784 | +0.0625 [-0.2500, +0.3750] |
| public candidate readout | 0.3125 | -0.1359 | +0.0000 [+0.0000, +0.0000] |
| zero source | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| wrong-row source | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| source-row shuffle | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| same-source-choice wrong-row | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| atom shuffle | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| coefficient shuffle | 0.3125 | -0.1177 | +0.0000 [+0.0000, +0.0000] |
| same-byte visible text | 0.3750 | -0.1951 | -0.0625 [-0.4375, +0.3125] |
| packet-only source index | 0.4375 | -0.1250 | -0.1250 [-0.4375, +0.1875] |

Headline:

- pass gate: `False`;
- best strict control: `packet_only_source_index` at `0.4375`;
- matched minus best control: `-0.1250`;
- dense diagnostic feature bytes per row: `57,344`;
- hypothetical top-k sparse proxy bytes per row: `44.0`;
- source extraction latency: `12.09s`;
- end-to-end wall time: `14.33s`.

## Interpretation

This fails as a positive ICLR branch. The most important diagnostic is that
matched, zero-source, wrong-row, source-row-shuffle, same-source-choice
wrong-row, atom-shuffle, and coefficient-shuffle conditions are identical. The
readout is not using row-specific source structure; it is behaving like a
public/candidate-index prior.

This does not rule out source token evidence broadly. It does rule out this
specific row-level token-pool representation plus ridge repair readout as a
source-causal packet precursor.

In lay terms: we gave the receiver detailed token clues from the source model,
but when we erased or shuffled those clues it made the same decisions. That
means this version was not truly listening to the source.

## Decision

- `ruled out`: row-level Qwen2.5 token hidden pool plus ridge candidate repair
  as a source-causal preflight.
- `weakened`: another always-on lightweight receiver over pooled source
  summaries.
- `alive`: candidate-local source evidence, because this gate flattened the
  token pool into a row-level vector and did not preserve candidate-local source
  structure.
- `alive`: dense-teacher/C2C-effect distillation, but it still needs either a
  proxy teacher on Mac or native GPU comparisons later.
- `not promoted`: Sparse Resonance Packets as a broad positive method.

## Next Exact Gate

Run `arc_candidate_local_source_evidence_repair_preflight`: use candidate-local
source hidden innovations, not row-level token pools, and train a tiny
train-only behavior repair readout. The pass bar remains beating target-only,
public-only, source-index/rank/score, same-byte text, wrong-row,
same-source-choice wrong-row, atom/coeff shuffles, and candidate-roll controls
with positive paired CI.

If candidate-local evidence also collapses, stop the Mac-local Qwen2.5 to Qwen3
ARC token-hidden branch and move the ICLR method search to a dense-teacher
proxy or NVIDIA-backed C2C/KV distillation surface.

## Commands

```bash
./venv_arm64/bin/python -m pytest tests/test_build_source_private_arc_tokenwise_repair_readout_preflight.py
./venv_arm64/bin/python scripts/build_source_private_arc_tokenwise_repair_readout_preflight.py
```
